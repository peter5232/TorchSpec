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

# BF16 模型配 FP32 master 权重的优化器封装：AdamW 更新高精度副本，再回写计算用参数。
# 梯度 buffer 持久复用；checkpoint 必须连同 master 一起保存，单存 BF16 参数会丢失累计精度。

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchspec.training.lr_scheduler import LRSchedulerWithWarmup
from torchspec.utils.logging import print_on_rank0


# 普通 tensor 用 foreach 批量复制；任一端包含 DTensor 时逐对 copy_ 保留分布式张量语义。
def _copy_tensors_(destinations, sources):
    has_dtensor = any(isinstance(tensor, DTensor) for tensor in destinations) or any(
        isinstance(tensor, DTensor) for tensor in sources
    )
    if has_dtensor:
        for destination, source in zip(destinations, sources, strict=True):
            destination.copy_(source)
        return
    torch._foreach_copy_(destinations, sources)


class BF16Optimizer:
    # 只为 requires_grad 参数建立 FP32 master 与复用梯度 buffer，冻结 teacher/embedding 不进入优化器。
    # 要求 fused AdamW 支持设备端 found_inf，随后绑定 warmup/decay 调度器。
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        decay_style="cosine",
        min_lr=0.0,
        wsd_decay_steps=None,
        wsd_decay_style=None,
    ):
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        # 每个可训练参数对应一个独立 FP32 master，detach 避免这次拷贝进入计算图。
        # 前向仍使用原模型参数，AdamW 操作的是 master；梯度需在 step 中显式复制过去。
        # 直观例子：BF16 在 1 附近的相邻可表示值相差约 0.0078125。
        # 单次 0.001 量级更新可能在回写时被舍入，但 master 保留它，多个小更新可逐步累积，
        # 而每步直接修改 BF16 参数可能把这些变化反复舍掉；这与 loss scaling 是两件事。
        self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
        # 持久梯度 buffer 避免每步反复分配 FP32 存储。
        # 没有模型梯度的参数设 mp.grad=None，不把上步 buffer 残留值误当成本步新梯度。
        self.fp32_grads = [torch.zeros_like(mp) for mp in self.fp32_params]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )
        if not getattr(self.optimizer, "_step_supports_amp_scaling", False):
            raise RuntimeError(
                "BF16Optimizer requires fused AdamW with device-side found_inf support"
            )
        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
            decay_style=decay_style,
            min_lr=min_lr,
            wsd_decay_steps=wsd_decay_steps,
            wsd_decay_style=wsd_decay_style,
        )

    # 把模型梯度拷到 FP32 buffer，裁剪后检测非有限范数，并跨 rank MAX 保证一致跳过更新。
    # AdamW 的 found_inf 跳过参数、moment、weight decay 更新；此封装仍推进 scheduler 并清梯度。
    # 最后将 FP32 master 回写模型；返回的是裁剪前范数，不是裁剪后范数。
    def step(self, closure=None):
        """Perform optimizer step with gradient clipping.

        Args:
            closure: Ignored, for compatibility with PyTorch optimizer interface.

        Returns:
            grad_norm: The gradient norm before clipping (for logging).
        """
        with torch.no_grad():
            grad_destinations = []
            grad_sources = []
            for p, mp, g in zip(self.model_params, self.fp32_params, self.fp32_grads):
                if p.grad is not None:
                    grad_destinations.append(g)
                    grad_sources.append(p.grad)
                    mp.grad = g
                else:
                    # None 表示本步完全未参与该参数的更新，AdamW 连其动量/衰减也不推进。
                    # 与数值全零梯度不同：零梯度仍可被历史动量或 weight decay 推动。
                    mp.grad = None
            if grad_destinations:
                _copy_tensors_(grad_destinations, grad_sources)

        # 调用方已完成模型特定归一化；这里将复制到 FP32 buffer 的梯度统一裁剪。
        # 返回的是裁剪前范数；例如 norm=2、阈值=0.5，各梯度缩小约 4 倍，但返回值仍为 2。
        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)

        # Fused AdamW consumes this device scalar through the same path used by
        # GradScaler. A nonzero value skips parameter, moment, weight-decay, and
        # optimizer-step updates without synchronizing the CUDA scalar to Python.
        found_inf = (~torch.isfinite(grad_norm)).to(dtype=torch.float32)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # A sharded rank may be the only one that observes a nonfinite
            # gradient. All training ranks must make the same update decision.
            collective_found_inf = (
                found_inf.to_local() if isinstance(found_inf, DTensor) else found_inf
            )
            dist.all_reduce(collective_found_inf, op=dist.ReduceOp.MAX)
        # 标记留在设备端让 fused AdamW 判断，避免 Python if isfinite(...).item() 的主机同步。
        # 跨 rank MAX 保证一个分片发现 NaN/Inf 时其他分片也不更新，避免模型版本彼此分叉。
        self.optimizer.found_inf = found_inf
        self.optimizer.step()

        # 两份 .grad 属于不同对象：此调用清 master.grad，末尾循环才清 model.grad。
        # fp32_grads 列表仍持有实际 buffer，下一窗口用 copy_ 覆盖；释放 .grad 引用
        # 不等于释放这块持久存储，也不是把模型参数数值清零。
        self.optimizer.zero_grad()
        # 即使 found_inf 让 fused AdamW 跳过更新，这里的调度步仍然推进；读曲线时需注意这一约定。
        self.scheduler.step()
        with torch.no_grad():
            # 回写会按模型参数 dtype 转换；因此模型中的 BF16 值可被视为 master 的计算副本。
            # 清模型 p.grad 释放当前图的梯度引用，下一累积窗口从空梯度开始，不能误跨窗口累积。
            _copy_tensors_(self.model_params, self.fp32_params)
            for p in self.model_params:
                p.grad = None

        return grad_norm

    # 同时清 AdamW master 梯度与模型梯度，set_to_none 决定释放引用还是原地清零。
    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    # 载入底层 AdamW 状态并由 rank 0 记录日志；独立 master 参数由 checkpoint 包装器恢复。
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        print_on_rank0("Successfully loaded optimizer state_dict.")

    # 把当前模型参数复制到 master；用于只恢复模型后防止旧 master 覆盖新权重。
    # 从 BF16 升回 FP32 不会恢复已损失的低位，能用检查点中的 master 时应优先用它。
    def sync_fp32_params_from_model(self):
        """Reinitialize fp32_params from model params. Call after loading model checkpoint."""
        with torch.no_grad():
            _copy_tensors_(self.fp32_params, self.model_params)

    # 返回 AdamW 的 moments/参数组等状态；不包含独立 master 数值和调度器，保存方需分别处理。
    def state_dict(self):
        return self.optimizer.state_dict()

    # 读取第一个 AdamW 参数组当前 lr，供训练指标记录。
    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    # 直接暴露底层优化器 state 字典，保持已有调用方访问 Adam 状态的接口。
    @property
    def state(self):
        return self.optimizer.state

    # 返回底层参数组引用；其中 params 指向 FP32 master，并非模型 BF16 参数列表。
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    # 暴露当前 scheduler 对象，便于 Trainer 单独保存/恢复学习率步数。
    @property
    def lr_scheduler(self):
        return self.scheduler
