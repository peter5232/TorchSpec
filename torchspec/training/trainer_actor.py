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

# Ray Actor 边界：设置 GPU/rank 与 torch.distributed，再把远程调用委托给具体 Trainer。
# Ray 负责进程编排，NCCL/Gloo 负责训练集合通信，两者的职责不同。

import os
from argparse import Namespace
from datetime import timedelta

import torch.distributed as dist

from torchspec import AutoDraftModelConfig
from torchspec.models.draft.dflash import DFlashConfig
from torchspec.models.draft.dflash2 import DFlash2Config
from torchspec.models.draft.dspark import DSparkConfig
from torchspec.ray.ray_actor import RayActor
from torchspec.utils.distributed import init_gloo_group, init_usp_groups
from torchspec.utils.logging import setup_file_logging


# 先判断 DFlash2/DSpark 等更具体的配置子类，再判断 DFlash，剩余走 Eagle Trainer。
def _trainer_class_for_config(draft_model_config):
    if isinstance(draft_model_config, DFlash2Config):
        from torchspec.training.dflash2_trainer import DFlash2Trainer

        return DFlash2Trainer
    if isinstance(draft_model_config, DSparkConfig):
        from torchspec.training.dspark_trainer import DSparkTrainer

        return DSparkTrainer
    if isinstance(draft_model_config, DFlashConfig):
        from torchspec.training.dflash_trainer import DFlashTrainer

        return DFlashTrainer
    from torchspec.training.eagle3_trainer import Eagle3Trainer

    return Eagle3Trainer


class TrainerActor(RayActor):
    # 设置 rendezvous 地址与 rank 环境变量，绑定本进程 GPU 并建立按 rank 分文件的训练日志。
    def __init__(self, world_size: int, rank: int, master_addr: str, master_port: int):
        self._world_size = world_size
        self._rank = rank

        self.setup_master(master_addr, master_port, port_range=(20000, 21000))

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)

        self.setup_gpu()
        setup_file_logging("training", self._rank)

    # 初始化分布式进程组及可选 USP/Gloo 组，解析 draft 配置，再选择并初始化训练器。
    # CPU offload 可使用复合 cpu/cuda backend；role/with_ref 当前不影响此实现。
    def init(self, args: Namespace, role: str, mooncake_config=None, with_ref: bool = False) -> int:
        self.args = args

        backend = getattr(args, "distributed_backend", "nccl")
        if getattr(args, "fsdp_cpu_offload", False) and getattr(args, "fsdp_cpu_backend", None):
            cpu_backend = args.fsdp_cpu_backend
            backend = f"cpu:{cpu_backend},cuda:{backend}"

        # Ray actor 已有独立 Python 进程，但梯度通信仍需建立 torch.distributed rendezvous。
        # MASTER_ADDR/PORT/WORLD_SIZE/RANK 在构造阶段设好，此处据此加入同一训练作业。
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=getattr(args, "distributed_timeout_minutes", 30)),
        )

        if getattr(args, "attention_backend", None) == "usp":
            init_usp_groups(
                sp_ulysses_size=getattr(args, "sp_ulysses_size", 1),
                sp_ring_size=getattr(args, "sp_ring_size", 1),
            )

        # Gloo 用于控制对象/CPU 侧协调；NCCL 处理 GPU collective。
        # 所有 rank 按相同顺序创建这些组，不能只有某个 rank 为了调试临时加一个 collective group。
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        draft_model_config = getattr(args, "draft_model_config_obj", None)
        if draft_model_config is None and getattr(args, "draft_model_config", None):
            draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

        # 配置类型决定损失/网络适配，Actor 只负责生命周期和远程接口。
        # 因此新增模型变体通常应更换 Trainer 构建 hook，而不是在 Actor 中复制前反向循环。
        trainer_class = _trainer_class_for_config(draft_model_config)
        self._trainer = trainer_class(args)

        target_model_path = getattr(args, "target_model_path", None)

        if draft_model_config is not None:
            self._trainer.init_model(
                draft_model_config=draft_model_config,
                target_model_path=target_model_path,
                mooncake_config=mooncake_config,
            )

        return 0

    # 远程训练入口：将步号和 microbatch 数交 Trainer，返回其当前可兑现的指标。
    def train_from_queue(self, step: int, num_batches: int) -> dict:
        return self._trainer.train_from_queue(step, num_batches)

    # 训练末尾远程取出延迟的一份指标，防止最后一步的统计没有机会被下一步兑现。
    def flush_pending_train_metrics(self) -> dict:
        return self._trainer.flush_pending_metrics()

    # 把控制器分配的队列和每 DP rank batch 配置传给 trainer，建立真正张量消费端。
    def set_train_queue(self, queue, mooncake_config=None, per_dp_rank_batch_size: int = 1):
        return self._trainer.set_train_queue(
            queue, mooncake_config=mooncake_config, per_dp_rank_batch_size=per_dp_rank_batch_size
        )

    # 返回已经计入的优化器步数，用于编排端进度查询。
    def get_global_step(self) -> int:
        return self._trainer.global_step

    # 委托训练检查点保存，参数原样转交；持久化 collective 仍须所有训练 rank 协作。
    def save_model(self, step: int, force_sync: bool = False) -> None:
        self._trainer.save_model(step, force_sync)

    # 把完整 draft 导出到指定目录供推理端更新，与完整续训 checkpoint 格式区分。
    def save_draft_model_for_serving(self, output_dir: str) -> None:
        self._trainer.save_draft_model_for_serving(output_dir)

    # 向支持词表裁剪的 draft 设置 d2t/t2d；不支持时显式报错，防止误用于 DFlash。
    def set_vocab_buffers(self, d2t, t2d) -> None:
        if hasattr(self._trainer, "draft_model") and hasattr(
            self._trainer.draft_model, "set_vocab_buffers"
        ):
            self._trainer.draft_model.set_vocab_buffers(d2t, t2d)
        else:
            raise AttributeError(
                "set_vocab_buffers called but draft model does not support vocab pruning. "
                "DFlash training should not use vocab pruning — check train_entry config."
            )

    # 委托建立独立评估数据流与 collator，保留训练队列的消费状态。
    def set_eval_queue(self, queue, mooncake_config=None, per_dp_rank_batch_size: int = 1):
        return self._trainer.set_eval_queue(
            queue, mooncake_config=mooncake_config, per_dp_rank_batch_size=per_dp_rank_batch_size
        )

    # 让训练器读取并缓存指定数量评估元素，返回当前缓存总数。
    def cache_eval_samples(self, count: int) -> int:
        return self._trainer.cache_eval_samples(count)

    # 把评估 CPU 缓存保存请求交给 trainer 的后台 I/O 流程。
    def save_eval_cache(self, cache_dir: str) -> None:
        return self._trainer.save_eval_cache(cache_dir)

    # 让训练器加载此 rank 的持久化评估缓存，返回加载元素数。
    def load_eval_cache(self, cache_dir: str) -> int:
        return self._trainer.load_eval_cache(cache_dir)

    # 触发具体模型训练器的前向评估与分布式指标归约，不执行优化器更新。
    def eval_from_cache(self) -> dict:
        return self._trainer.eval_from_cache()
