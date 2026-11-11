# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# 中文导读：评估使用固定 teacher 特征缓存，避免每次评估都重复运行昂贵目标模型。
# 构建缓存仍经过同一推理→Mooncake→训练端链路，但有独立队列并按小块推进。
# 这里记录的 simulated_acc_len 是训练评估指标；不能直接当成部署引擎测得的加速倍数。

"""Eval-specific setup and runtime helpers for the controller loop."""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import ray
import wandb
from tqdm import tqdm

from torchspec.training.checkpoint import _read_checkpoint_metadata, _write_checkpoint_metadata
from torchspec.utils.logging import logger

EVAL_CACHE_IDLE_TIMEOUT = 300.0


# 封装评估调度状态：是否启用、缓存路径/命中、历史最好分数和已提交的首块数量。
@dataclass
class EvalSetupState:
    eval_interval: int
    eval_enabled: bool
    eval_cache_loaded: bool
    eval_cache_path: str | None
    best_eval_score: float
    eval_dispatch_bs: int
    eval_dataset_size: int
    dp_size: int
    initial_eval_submit_count: int = 0


# 按“距上次实际进展”计超时，而非评估总用时；长任务只要持续收到数据就不会被误判停滞。
def _check_idle_timeout(
    dispatched_samples: int, last_progress_at: float, total_samples: int
) -> None:
    # monotonic 不受系统时间校准影响；此检查只在 try_dispatch 返回 False 后被调用。
    # 若某个 ray.get/cache_eval_samples 本身挂住，本函数没有机会运行，并非 RPC 的全局 deadline。
    idle_for = time.monotonic() - last_progress_at
    if idle_for >= EVAL_CACHE_IDLE_TIMEOUT:
        raise TimeoutError(
            "Timed out while waiting for eval cache generation "
            f"(no progress during eval for {idle_for:.1f}s, "
            f"idle_timeout={EVAL_CACHE_IDLE_TIMEOUT:.1f}s, "
            f"dispatched={dispatched_samples}/{total_samples} samples)"
        )


# 将评估结果写入已有 checkpoint 元数据，并按 simulated_acc_len 更新最佳指示文件。
# 这里用 step+1 构造目录编号，是与 checkpoint 保存侧配对的约定；阅读时不要擅自改成 step。
def update_checkpoint_eval_meta(
    checkpoint_dir: str | None,
    step: int,
    eval_metrics: dict,
    current_best: float,
) -> float:
    """Append eval metrics to checkpoint meta.json and track best checkpoint."""
    if not checkpoint_dir or not eval_metrics:
        return current_best

    base_dir = Path(checkpoint_dir).expanduser()
    # 例 loop 已完成第 20 次更新并传 step=20，checkpoint.save 写 iter_0000021，
    # 此处必须命中同一目录；元数据 global_step 仍可为 20。目录号与优化器更新数不是同一字段。
    step_id = step + 1
    meta_path = base_dir / f"iter_{step_id:07d}" / "meta.json"

    metadata = _read_checkpoint_metadata(meta_path)
    if not metadata:
        logger.warning(f"Checkpoint meta.json not found at {meta_path}, skipping eval meta update")
        return current_best

    for key in ("eval/avg_loss", "eval/avg_acc", "eval/simulated_acc_len"):
        if key in eval_metrics:
            metadata[key] = eval_metrics[key]
    _write_checkpoint_metadata(meta_path, metadata)

    score = eval_metrics.get("eval/simulated_acc_len")
    # 相等不刷新 best；若本次只有 avg_loss 而没有 simulated_acc_len，也不按 loss 自动选优。
    # best 文件只是指向目录的记录，后面的 max_checkpoints 按新旧清理时没有读取该指针保护它。
    if score is not None and score > current_best:
        current_best = score
        (base_dir / "best_checkpointed_iteration.txt").write_text(str(step_id))
        _write_checkpoint_metadata(base_dir / "best_meta.json", metadata)
        logger.info(f"New best checkpoint: iter_{step_id:07d} (simulated_acc_len={score:.2f})")

    return current_best


# 每分发一批就让训练组取走并缓存，再提交下一小块输入，形成有界的生产消费节奏。
# 若迟迟凑不齐批次则做空闲超时检查；全部处理后清理控制器评估状态并异步落盘缓存。
def generate_eval_cache(
    controller,
    train_group,
    eval_state: EvalSetupState,
) -> None:
    """Drain eval results from inference and cache on each trainer."""
    eval_dataset_size = eval_state.eval_dataset_size
    dp_size = eval_state.dp_size
    dispatch_bs = eval_state.eval_dispatch_bs
    eval_cache_path = eval_state.eval_cache_path

    last_progress_at = time.monotonic()
    logger.info(
        f"Caching eval hidden states from inference engine "
        f"({eval_dataset_size} samples, dispatch_bs={dispatch_bs})..."
    )
    dispatched_samples = 0
    # 初始已经提交一小段评估输入；每消费一个 dispatch 再补一个，形成有限窗口。
    # 这样即使验证集很大，也不需要全部 hidden states 同时留在 Mooncake。
    next_submit_idx = eval_state.initial_eval_submit_count
    eval_progress = tqdm(total=eval_dataset_size, desc="Eval caching", unit="sample")

    while dispatched_samples < eval_dataset_size:
        ok = ray.get(controller.try_dispatch_eval_batch.remote())
        if ok:
            # controller 加载评估集时先截成 DP 数的整倍数，正常启用时 dispatch_bs=DP，
            # 因而每 rank 每次缓存 1 项；USP 的多个 SP rank 各取同一逻辑样本的不同片段。
            # 若绕开 loader 直接传小于 DP 的正数，整数除法会变 0，不能视为受支持的小尾批。
            train_group.cache_eval_samples(dispatch_bs // dp_size)
            dispatched_samples += dispatch_bs
            if next_submit_idx < eval_dataset_size:
                next_end = min(next_submit_idx + dispatch_bs, eval_dataset_size)
                ray.get(controller.submit_eval_chunk.remote(next_submit_idx, next_end))
                next_submit_idx = next_end
            # 进展在远程消费完成后才更新，而非仅仅发出一次推理请求时更新。
            # 例如 600 秒总耗时中每 20 秒完成一批，不触发 300 秒无进展超时。
            last_progress_at = time.monotonic()
            eval_progress.update(dispatch_bs)
        else:
            _check_idle_timeout(dispatched_samples, last_progress_at, eval_dataset_size)
            time.sleep(0.01)

    eval_progress.close()

    ray.get(controller.finalize_eval_dispatch.remote())
    logger.info(f"Eval caching complete ({dispatched_samples} samples)")
    if eval_cache_path:
        train_group.async_save_eval_cache(eval_cache_path)
        logger.info(f"Eval cache save started (async) to {eval_cache_path}")


# 从训练端缓存执行评估，取 rank 0 已汇总的结果写日志；未启用时返回空字典。
def run_eval(step: int, train_group, eval_enabled: bool) -> dict:
    """Run forward-only eval from cache. Assumes eval cache is already populated."""
    if not eval_enabled:
        return {}
    eval_results = train_group.run_eval()
    eval_metrics = eval_results[0] if eval_results else {}
    if eval_metrics:
        eval_metrics["eval/step"] = step
        if wandb.run is not None:
            wandb.log(eval_metrics)
        logger.info(
            f"Step {step} eval: "
            f"loss={eval_metrics.get('eval/avg_loss', 0):.4f}, "
            f"acc={eval_metrics.get('eval/avg_acc', 0):.4f}, "
            f"sim_acc_len={eval_metrics.get('eval/simulated_acc_len', 0):.2f}"
        )
    return eval_metrics


# 结合验证集大小、历史最佳元数据和缓存路径准备评估状态。
# 缓存 key 包含数据源、目标模型路径、最大长度和辅助层列表，不是完整配置/权重内容哈希。
# 所有 rank 都成功加载缓存才算命中，否则先提交有限数量的验证输入等待构建。
def setup_eval(controller, train_group, args, eval_dataset_size: int) -> EvalSetupState:
    """Prepare eval runtime settings and optionally load/submit eval cache input."""
    eval_interval = args.eval_interval
    eval_enabled = eval_dataset_size > 0
    eval_cache_path: str | None = None
    eval_cache_loaded = False
    initial_eval_submit_count = 0

    # 例原验证集 10 条、DP=4，controller 只保留 8 条，这里每次分发 4 条、共两轮。
    # 原验证集仅 3 条时 controller 截为 0，会关闭评估，而不是给部分 rank 送空批凑数。
    eval_dispatch_bs = min(args.dp_size, eval_dataset_size)

    best_eval_score = -float("inf")
    if eval_enabled and args.checkpoint_dir:
        best_meta_path = Path(args.checkpoint_dir).expanduser() / "best_meta.json"
        if best_meta_path.exists():
            existing = _read_checkpoint_metadata(best_meta_path)
            if "eval/simulated_acc_len" in existing:
                best_eval_score = existing["eval/simulated_acc_len"]
                logger.info(f"Resumed best eval score: {best_eval_score:.2f}")

    if eval_enabled:
        cache_dir = os.path.abspath(getattr(args, "cache_dir", "./cache"))
        data_source = (
            getattr(args, "offline_data_path", "")
            if getattr(args, "inference_engine_type", None) == "offline"
            else getattr(args, "eval_data_path", "")
        )
        aux_hidden_states_layers = getattr(args, "aux_hidden_states_layers", None)
        if aux_hidden_states_layers is not None:
            # Hydra exposes YAML sequences as OmegaConf ListConfig objects,
            # which json.dumps cannot serialize directly.
            aux_hidden_states_layers = list(aux_hidden_states_layers)
        # MD5 只编码下面几个配置字符串：原文件在同路径改了内容或 teacher 同路径换了权重，
        # key 不会因此变化；DP/SP 拓扑也不在该 key 中，缓存命中不是完整运行配置兼容性证明。
        cache_key = hashlib.md5(
            f"{data_source}|"
            f"{getattr(args, 'target_model_path', '')}|"
            f"{getattr(args, 'max_seq_length', 0)}|"
            f"{json.dumps(aux_hidden_states_layers, separators=(',', ':'))}".encode()
        ).hexdigest()[:12]
        eval_cache_path = os.path.join(cache_dir, "eval_cache", cache_key)

        loaded = train_group.load_eval_cache(eval_cache_path)
        # 评估是分布式操作：只有所有 rank 都有缓存才算整体命中。
        # 单个 rank 缺缓存不能直接开始评估，否则各 rank 输入步数不同可能卡在聚合。
        # 代码只检查各 rank 的返回计数为正，没有在此断言数量相等或与当前 eval 集完全一致。
        # “所有 rank 都加载成功”和“数据/拓扑与缓存生成时一致”是两个需要分别核对的条件。
        if all(n > 0 for n in loaded):
            eval_cache_loaded = True
            logger.info(
                f"Eval: loaded cached tensors from {eval_cache_path} ({loaded[0]} batches per rank)"
            )
        else:
            inference_batch_size = args.inference_batch_size
            initial_eval_submit_count = min(
                eval_dataset_size,
                max(eval_dispatch_bs * 2, inference_batch_size),
            )
            ray.get(controller.submit_eval_chunk.remote(0, initial_eval_submit_count))
            logger.info(
                f"Eval: {eval_dataset_size} samples, dispatch_bs={eval_dispatch_bs}, "
                f"initial_submit={initial_eval_submit_count}"
            )

    return EvalSetupState(
        eval_interval=eval_interval,
        eval_enabled=eval_enabled,
        eval_cache_loaded=eval_cache_loaded,
        eval_cache_path=eval_cache_path,
        best_eval_score=best_eval_score,
        eval_dispatch_bs=eval_dispatch_bs,
        eval_dataset_size=eval_dataset_size,
        dp_size=args.dp_size,
        initial_eval_submit_count=initial_eval_submit_count,
    )
