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

# 统一学习率曲线：warmup 后接 constant/linear/cosine/inverse-square-root/WSD。
# 这里的 step 是 scheduler 的优化器步计数，不是 token 数或推理请求编号。

"""Unified learning rate schedulers for torchspec training."""

import math
from typing import Literal, Optional

import torch
from torch.optim.lr_scheduler import LRScheduler

from torchspec.utils.logging import logger

DecayStyle = Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"]
WSDDecayStyle = Literal["linear", "cosine", "exponential", "minus_sqrt"]


class LRSchedulerWithWarmup(LRScheduler):
    """Learning rate scheduler with warmup and multiple decay styles.

    Supports: constant, linear, cosine, inverse-square-root, and WSD (Warmup-Stable-Decay).

    Args:
        optimizer: The optimizer to schedule.
        max_lr: Maximum (peak) learning rate after warmup.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps (linear warmup from init_lr to max_lr).
        min_lr: Minimum learning rate after decay. Defaults to 0.0.
        init_lr: Initial learning rate at step 0. Defaults to 0.0.
        decay_style: Decay style after warmup. One of: constant, linear, cosine,
            inverse-square-root, WSD. Defaults to "cosine".
        wsd_decay_steps: For WSD style, number of steps for final decay phase.
        wsd_decay_style: For WSD style, decay curve in final phase.
        last_epoch: The index of last epoch. Defaults to -1.
    """

    # 规范化超参数并校验上下界/总步数；WSD 必须指定最后衰减阶段长度，再交父类建立计数。
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
        init_lr: float = 0.0,
        decay_style: DecayStyle = "cosine",
        wsd_decay_steps: Optional[int] = None,
        wsd_decay_style: Optional[WSDDecayStyle] = None,
        last_epoch: int = -1,
    ) -> None:
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.init_lr = float(init_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.decay_style = decay_style
        self.wsd_decay_steps = wsd_decay_steps
        self.wsd_decay_style = wsd_decay_style

        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr
        assert self.total_steps > 0
        assert self.warmup_steps < self.total_steps

        if self.decay_style == "WSD":
            assert self.wsd_decay_steps is not None, "WSD decay requires wsd_decay_steps"

        super().__init__(optimizer, last_epoch)

        logger.debug(
            f"LR scheduler: decay_style={decay_style}, warmup={warmup_steps}, total={total_steps}"
        )

    # 优先读取参数组局部上下界；warmup 线性升温，后续按已扣 warmup 的比例计算衰减。
    # constant 分支先于超总步数检查，因此该模式始终保持 max_lr；其他模式超过总步数返回 min_lr。
    def _get_lr_for_group(self, param_group: dict) -> float:
        """Compute learning rate for a specific parameter group."""
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)
        step = self.last_epoch

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.init_lr + (max_lr - self.init_lr) * step / self.warmup_steps

        if self.decay_style == "constant":
            return max_lr

        if step > self.total_steps:
            return min_lr

        if self.decay_style == "inverse-square-root":
            warmup_steps = max(self.warmup_steps, 1)
            num_steps = max(step, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        # 衰减比例从 warmup 结束算起，而不是简单 step/total_steps。
        # 例如总 100 步、warmup=10，第 55 步处于衰减段中点 (55-10)/(100-10)=0.5。
        num_steps_ = step - self.warmup_steps
        decay_steps_ = self.total_steps - self.warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        decay_ratio = max(0.0, min(1.0, decay_ratio))

        delta_lr = max_lr - min_lr

        if self.decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.decay_style == "WSD":
            coeff = self._compute_wsd_coeff(step, decay_ratio)
        else:
            raise ValueError(f"Unknown decay style: {self.decay_style}")

        # 系数描述在 [min_lr,max_lr] 区间内的位置，并非直接乘原始 lr。
        # 若 min=1e-5、max=1e-4、coeff=0.5，则结果为 5.5e-5，而不是 5e-5。
        return min_lr + coeff * delta_lr

    # 在 total_steps-wsd_decay_steps 之前保持系数 1，最后一段才按所选曲线从 1 降到 0。
    # 例如 total=100、decay=20，80 步前稳定，80 到 100 步衰减；decay_ratio 参数当前未参与计算。
    def _compute_wsd_coeff(self, step: int, decay_ratio: float) -> float:
        """Compute coefficient for WSD (Warmup-Stable-Decay) schedule."""
        wsd_anneal_start = self.total_steps - self.wsd_decay_steps
        if step <= wsd_anneal_start:
            return 1.0

        wsd_steps = step - wsd_anneal_start
        wsd_ratio = float(wsd_steps) / float(self.wsd_decay_steps)

        if self.wsd_decay_style == "linear":
            return 1.0 - wsd_ratio
        elif self.wsd_decay_style == "cosine":
            return 0.5 * (math.cos(math.pi * wsd_ratio) + 1.0)
        elif self.wsd_decay_style == "exponential":
            return (2.0 * math.pow(0.5, wsd_ratio)) - 1.0
        elif self.wsd_decay_style == "minus_sqrt":
            return 1.0 - math.sqrt(wsd_ratio)
        else:
            raise ValueError(f"Unknown WSD decay style: {self.wsd_decay_style}")

    # 逐参数组计算当前学习率列表，供 PyTorch LRScheduler 写回 optimizer。
    def get_lr(self) -> list[float]:
        return [self._get_lr_for_group(group) for group in self.optimizer.param_groups]


# 从现有参数组取最大初始 lr 作为峰值，构造线性 warmup 加 cosine 曲线的兼容入口。
def CosineAnnealingWarmupLR(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    eta_min: float = 0.0,
    last_epoch: int = -1,
) -> LRSchedulerWithWarmup:
    """Convenience constructor for cosine annealing with linear warmup.

    This is equivalent to LRSchedulerWithWarmup with decay_style="cosine".
    """
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    max_lr = max(base_lrs)

    return LRSchedulerWithWarmup(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=eta_min,
        init_lr=0.0,
        decay_style="cosine",
        last_epoch=last_epoch,
    )


# 显式把曲线配置转交统一调度器，不另维护一套计算公式。
def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    init_lr: float = 0.0,
    decay_style: DecayStyle = "cosine",
    wsd_decay_steps: Optional[int] = None,
    wsd_decay_style: Optional[WSDDecayStyle] = None,
    last_epoch: int = -1,
) -> LRSchedulerWithWarmup:
    """Factory function to create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        max_lr: Maximum (peak) learning rate after warmup.
        total_steps: Total number of training steps.
        warmup_steps: Number of warmup steps.
        min_lr: Minimum learning rate after decay.
        init_lr: Initial learning rate at step 0.
        decay_style: One of: constant, linear, cosine, inverse-square-root, WSD.
        wsd_decay_steps: For WSD style, number of steps for final decay.
        wsd_decay_style: For WSD style, decay curve type.
        last_epoch: The index of last epoch.

    Returns:
        Configured LRSchedulerWithWarmup instance.
    """
    # 工厂不复制参数到 optimizer，也不额外前进一步；曲线初始化交给统一 scheduler 构造器。
    # 所有 WSD 特定参数原样传入，让缺失或未知曲线在同一个实现中验证，避免入口间规则分叉。
    return LRSchedulerWithWarmup(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        init_lr=init_lr,
        decay_style=decay_style,
        wsd_decay_steps=wsd_decay_steps,
        wsd_decay_style=wsd_decay_style,
        last_epoch=last_epoch,
    )


# 由推理样本总量除以每次 DP batch 大小估算训练步数，并回写 args.train_iters/lr_decay_iters。
# warmup 优先使用比例，其次使用绝对步数，再构造同一调度器。
def get_lr_scheduler_from_args(args, optimizer: torch.optim.Optimizer) -> LRSchedulerWithWarmup:
    """Create scheduler from training args namespace (for FSDP actor compatibility).

    Args:
        args: Training arguments with lr scheduling config.
        optimizer: Optimizer bound to the model.

    Returns:
        Configured LRSchedulerWithWarmup instance.
    """
    # 这里依据生产规模估计完整 DP batch 数：生成样本总数除以每次消费样本数。
    # 整数除法丢弃不足一整 batch 的余数，不能把它当作任意数据流都精确成立的终止条件。
    train_iters = (
        args.num_inference
        * args.inference_batch_size
        * args.n_samples_per_prompt
        // (args.per_dp_rank_batch_size * args.dp_size)
    )
    args.train_iters = train_iters

    lr_decay_iters = getattr(args, "lr_decay_iters", None) or train_iters
    args.lr_decay_iters = lr_decay_iters

    # warmup_fraction 的分母是 lr_decay_iters，不一定是原估计 train_iters。
    # 因此用户显式改变衰减总步数，也会改变按比例设置的 warmup 步数。
    if getattr(args, "lr_warmup_fraction", None) is not None:
        warmup_steps = int(args.lr_warmup_fraction * lr_decay_iters)
    else:
        warmup_steps = getattr(args, "lr_warmup_iters", 0)

    wsd_decay_steps = getattr(args, "lr_wsd_decay_iters", None)

    return LRSchedulerWithWarmup(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=lr_decay_iters,
        warmup_steps=warmup_steps,
        min_lr=getattr(args, "min_lr", 0.0),
        init_lr=getattr(args, "lr_warmup_init", 0.0),
        decay_style=getattr(args, "lr_decay_style", "cosine"),
        wsd_decay_steps=wsd_decay_steps,
        wsd_decay_style=getattr(args, "lr_wsd_decay_style", None),
    )
