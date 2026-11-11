# 中文导读：特征物化作业。复用在线推理和控制器，但消费者换成保存 actor，不执行草稿反向传播。
# 与离线训练相对：本文件负责“生成离线数据”，offline_replay_engine 负责“读取离线数据供训练”。

"""Materialize live target-model outputs for offline training."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.controller.inference_manager import AsyncInferenceManager
from torchspec.controller.loop import _safe_training_cleanup
from torchspec.controller.setup import build_mooncake_config
from torchspec.controller.training_controller import AsyncTrainingController
from torchspec.inference.factory import create_inference_engines
from torchspec.offline.dataset import OfflineDataset
from torchspec.offline.saving_actor import OfflineSavingActor
from torchspec.ray.placement_group import _ensure_ray_initialized, create_placement_groups
from torchspec.train_entry import (
    _get_draft_model_config,
    _resolve_batch_size,
    _validate_and_configure_dflash,
)
from torchspec.transfer.mooncake.utils import launch_mooncake_master
from torchspec.utils.env import get_torchspec_env_vars
from torchspec.utils.logging import logger


# 解析生成目录、覆盖开关和训练配置；借用 batch 推导但明确禁止当前未支持的 USP 物化。
def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--overwrite", action="store_true")
    cli, unknown = parser.parse_known_args()
    args = config_to_flat_args(load_config(cli.config, cli_args=unknown or None))
    args.rank = 0
    args.world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        raise ValueError("Offline materialization does not support USP")
    _resolve_batch_size(args)
    return cli, args


# 若草稿裁剪词表，连同数据保存 d2t/t2d 映射，回放时保持输出列与 token 的相同对应。
def _save_vocab_mapping(controller, output_dir: str, draft_config) -> None:
    draft_vocab_size = getattr(draft_config, "draft_vocab_size", None)
    if draft_vocab_size is None or draft_vocab_size == draft_config.vocab_size:
        return
    path = Path(output_dir) / "vocab_mapping.pt"
    d2t, t2d = ray.get(
        controller.compute_vocab_mapping.remote(draft_config.vocab_size, draft_vocab_size)
    )
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"d2t": d2t.cpu(), "t2d": t2d.cpu()}, temporary)
    os.replace(temporary, path)


# 建立真实推理引擎、单消费者队列和磁盘保存器，逐条持久化训练/验证样本。
# 每 dispatch 一条可保留数据尾部；有界输出池降低磁盘慢于 GPU 时的积压，不是严格的物理内存上限。
# 循环同时跟踪预期数和已处理数：若所有输入/任务/输出都耗尽却未达到预期，显式失败而非静默少样本。
def materialize(args, output_dir: str, *, overwrite: bool = False) -> dict[str, int]:
    if getattr(args, "inference_engine_type", None) == "offline":
        raise ValueError("Materialization requires a live inference engine")

    draft_config = _get_draft_model_config(args)
    args.draft_model_config_obj = draft_config
    _validate_and_configure_dflash(args, draft_config)
    # A one-item dispatch lets the saving actor persist every source sample,
    # including a final partial inference batch.
    # 物化要保存每一条原始样本，不希望像优化器训练那样丢掉不足 global batch 的尾部。
    # 控制器用 DP=1 且每次一条，因此无论引擎批大小如何，最后剩余样本也可逐条消费。
    # 例如 5 条输入、引擎 batch=2，产出可以是 2+2+1，但消费仍按五次单样本提交。
    # 这里改变的是控制器 dispatch 粒度，不把 teacher 引擎的 inference_batch_size 改成 1。
    args.per_dp_rank_batch_size = 1
    # 先建立磁盘格式元信息，尤其记录最后隐状态是否 pre-norm；后续回放必须复用这一语义。
    # overwrite 参数会沿构造路径删除已有目录，阅读/使用这个入口时要区分普通续写和明确覆盖。
    OfflineDataset(
        output_dir,
        create=True,
        last_hidden_states_prenorm=args.last_hidden_states_prenorm,
        overwrite=overwrite,
    )

    _ensure_ray_initialized()
    driver_node_id = ray.get_runtime_context().get_node_id()
    controller = AsyncTrainingController.options(
        runtime_env={"env_vars": get_torchspec_env_vars()},
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=driver_node_id, soft=False),
    ).remote(args, 1)
    # 这里仍从配置加载全部源数据，没有用已有 manifest 的 IDs 从源数据中剔除已完成项。
    # 重启续写可能重新跑 teacher；节省的是重复落盘，不能理解成从最后一次 GPU 进度继续计算。
    train_size, eval_size = ray.get(
        [controller.load_dataset.remote(args), controller.load_eval_dataset.remote(args)]
    )
    # 裁剪词表的映射是数据语义的一部分，一并保存使回放训练不用重新根据其他语料选词。
    # 若重新选词，同一 lm_head 列可能对应不同 token，即便所有张量维度看起来正确。
    # 映射按本次源数据计算并替换，未与旧文件做内容一致性比较；同目录续写应保持源数据/模型语义。
    # dataset.json 的 norm 检查无法发现词表映射或教师辅助层配置变化。
    _save_vocab_mapping(controller, output_dir, draft_config)

    # 这里只分配推理 GPU；保存 actor 使用 CPU 并从 Mooncake 下载，不创建草稿训练进程。
    # 因此物化与训练可以在不同时间使用硬件，而不是要求两组模型同时驻留。
    inference_pg = create_placement_groups(args, roles={"inference"})["inference"]
    launch_mooncake_master(args)
    mooncake_config = build_mooncake_config(args)
    engines = create_inference_engines(args, inference_pg, mooncake_config)

    # 单写者 actor 维护 manifest，避免多个进程同时追加同一清单导致索引竞争。
    # Ray 队列里仍是 TrainSample，生产者无须知道下游究竟做反向传播还是写磁盘。
    saver = OfflineSavingActor.remote(output_dir, mooncake_config)
    train_queues, eval_queues = ray.get(
        [controller.get_train_queues.remote(), controller.get_eval_queues.remote()]
    )
    ray.get(saver.set_queues.remote(train_queues[0], eval_queues[0]))

    # 默认给待消费结果池一个有限条数阈值，降低磁盘慢于 GPU 时的积压。
    # 它不统计所有在途输出的物理字节，因此不构成“内存绝不超限”的硬保证。
    if getattr(args, "max_sample_pool_size", 0) <= 0:
        args.max_sample_pool_size = max(
            8,
            getattr(args, "inference_batch_size", 1)
            * getattr(args, "max_concurrent_batches", 1)
            * 4,
        )
    manager = AsyncInferenceManager.remote(
        args,
        controller,
        inference_engines=engines,
        max_concurrent_batches=getattr(args, "max_concurrent_batches", 1),
    )
    # 先提交验证条目，再提交训练条目；下面消费循环也先尝试 eval。
    # 两条队列仍然分开，写入清单时 split 不会因处理顺序而丢失。
    if eval_size:
        ray.get(controller.submit_eval_chunk.remote(0, eval_size))
    ray.get(controller.submit_training_dataset.remote())

    manager_future = manager.run.remote()
    # processed 计本次已消费的队列条目，dataset.count 则计去重后的磁盘记录。
    # 例如 ID 已在旧数据集里，append 返回 False，但这条推理结果已经被消费，所以 processed 仍加一。
    expected = {"train": train_size, "eval": eval_size}
    processed = {"train": 0, "eval": 0}
    counts = None
    try:
        # 这个 try/finally 从 manager 启动后的处理阶段开始；更早的初始化失败不在同一个清理范围内。
        # 不要因为看见 finally 就推断任意阶段异常都必然释放了此前创建的全部资源。
        while processed != expected:
            made_progress = False
            for split, method in (
                ("eval", controller.try_dispatch_eval_batch),
                ("train", controller.try_dispatch_batch),
            ):
                if processed[split] < expected[split] and ray.get(method.remote()):
                    # dispatch 成功只说明 TrainSample 已放入队列；等 saver RPC 正常返回才算消费完成。
                    # append 因重复 ID 返回 0 仍算一次 processed；RPC 抛错则本轮不会加一。
                    ray.get(saver.save_from_queue.remote(split))
                    processed[split] += 1
                    made_progress = True

            if made_progress:
                logger.info("Materialization progress: %s / %s", processed, expected)
                continue

            manager_status, controller_status = ray.get(
                [manager.get_status.remote(), controller.get_full_status.remote()]
            )
            # 只有管理器输入/任务、控制器输入/训练结果、评估结果五处都空，才判断生产流已经耗尽。
            # 若此时 processed 未达到 expected，就说明有样本未成功产出；应报错而不是返回一个悄悄变小的数据集。
            # 这是多个 actor 的状态快照组合，不是跨 actor 的原子快照或对每个 ID 的完成确认。
            # 它用于发现流水线已无工作却少样本的情况，不提供失败样本的重试/补齐机制。
            drained = (
                manager_status["prompt_buffer_size"] == 0
                and manager_status["pending_tasks"] == 0
                and controller_status["prompt_buffer_size"] == 0
                and controller_status["sample_pool_size"] == 0
                and ray.get(controller.get_eval_pool_size.remote()) == 0
            )
            if drained:
                raise RuntimeError(
                    f"Inference drained early: processed={processed}, expected={expected}"
                )
            time.sleep(0.05)

        if eval_size:
            ray.get(controller.finalize_eval_dispatch.remote())
        # 返回值来自去重后的实际磁盘索引，不把本次 processed 当作最终数据集大小。
        counts = ray.get(saver.counts.remote())
    finally:
        try:
            # 先关闭读取/保存客户端，再复用通用引擎停机流程，最后向控制器队列投递结束信号。
            # 逐段捕获清理异常并记录，尽量不让次生清理错误遮盖原始失败。
            ray.get(saver.close.remote())
        except Exception as exc:
            logger.warning("Failed to close offline saver: %s", exc)
        _safe_training_cleanup(args, manager, manager_future, engines)
        try:
            ray.get(controller.shutdown.remote())
        except Exception as exc:
            logger.warning("Failed to stop materialization controller: %s", exc)

    logger.info("Offline dataset ready at %s: %s", output_dir, counts)
    return counts


# 命令行入口：解析后启动物化，overwrite 原样传递给数据目录创建逻辑。
def main() -> None:
    cli, args = _parse_args()
    materialize(args, cli.output, overwrite=cli.overwrite)


if __name__ == "__main__":
    main()
