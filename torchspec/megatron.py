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

# Megatron-Core 仅提供模型/专家并行组和梯度辅助函数，训练循环与优化器仍由 TorchSpec 管理。
# 专家参数分片与 dense 参数复制拥有不同归约组，范数统计也必须避免重复计数。

"""Megatron-Core process groups and expert-parallel gradient helpers.

TorchSpec keeps its own training loop, loss and optimizer; this module only
sets up Megatron model-parallel state and handles gradients whose ownership
differs between replicated and expert-sharded parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection


# 严格比较当前 Megatron-Core 版本与 expected，不匹配即报错，避免依赖内部接口版本漂移。
def require_megatron_version(expected: str) -> None:
    from megatron.core.package_info import __version__

    if __version__ != expected:
        raise RuntimeError(f"Megatron-Core {expected} is required, found {__version__}")


# 要求 torch.distributed 已初始化且 EP 整除 world，固定 TP=PP=1 建立 Megatron 组和 CUDA RNG。
# 已有组时核对 EP 大小；最终验证 dense DP 大小等于 EP×expert-DP，这是后续梯度除数的前提。
def initialize_model_parallel(
    *, expert_parallel_size: int, seed: int = 1234
) -> "ProcessGroupCollection":
    """Create TP1/PP1 Megatron groups with the given EP size and seed Megatron's CUDA RNG."""

    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if dist.get_world_size() % expert_parallel_size:
        raise ValueError(
            f"world size {dist.get_world_size()} is not divisible by EP {expert_parallel_size}"
        )
    # 初始化是进程级全局状态，重复调用不能再创建一套不同 EP 布局。
    # 已有布局仅核对专家并行大小；种子只在首次建立模型并行时设置，不在每次查询时重置 RNG。
    if parallel_state.model_parallel_is_initialized():
        actual = parallel_state.get_expert_model_parallel_world_size()
        if actual != expert_parallel_size:
            raise RuntimeError(f"Megatron is already initialized with EP={actual}")
    else:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=expert_parallel_size,
            order="tp-cp-ep-dp-pp",
        )
        model_parallel_cuda_manual_seed(seed)

    # 返回的是多个不同用途的通信组集合，不是一条统一的“模型并行组”。
    # dp 覆盖复制参数的数据并行，expt_dp 连接同一专家分片的副本，ep 连接不同专家分片。
    groups = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "cp", "ep", "dp", "expt_tp", "expt_dp", "tp_ep", "tp_cp", "tp_dp_cp"]
    )
    # synchronize_gradients relies on the dense DP group spanning EP x expert-DP.
    sizes = [dist.get_world_size(g) for g in (groups.dp, groups.ep, groups.expt_dp)]
    if sizes[0] != sizes[1] * sizes[2]:
        raise RuntimeError(
            f"unexpected Megatron groups: dp={sizes[0]}, ep={sizes[1]}, expt_dp={sizes[2]}"
        )
    return groups


# 以 Megatron 的 allreduce=False 标记判断专家分片；没有该标记的参数视为复制参数。
def is_expert_parameter(parameter: torch.nn.Parameter) -> bool:
    return getattr(parameter, "allreduce", True) is False


# dense 参数在 dense DP 组 SUM，专家参数仅在 expert-DP 副本组 SUM，二者都除 dense DP 大小。
# EP 内输入 batch 相同且 token 已路由给专家，不能再把专家梯度按另一套分母重复平均。
def synchronize_gradients(module: torch.nn.Module, groups: "ProcessGroupCollection") -> None:
    """Average gradients across data-parallel replicas, following Megatron DDP.

    All ranks of an EP group see the same batch, so a routed expert's gradient
    already sums one copy of every token per EP rank. Replicated gradients are
    summed over dense DP (which spans EP), expert gradients over expert DP, and
    both are divided by the dense DP size.
    """

    dp_size = dist.get_world_size(groups.dp)
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        group = groups.expt_dp if is_expert_parameter(parameter) else groups.dp
        if dist.get_world_size(group) > 1:
            dist.all_reduce(parameter.grad, group=group)
        if dp_size > 1:
            parameter.grad.div_(dp_size)


# 分别统计复制参数和专家参数平方和，仅专家部分跨 EP 求和，使每个专家分片被计一次。
# 返回裁剪前 L2 norm，并用 min(max_norm/(norm+eps),1) 原地缩放所有梯度；调用需至少存在一个梯度。
def clip_grad_norm_(
    module: torch.nn.Module, max_norm: float, groups: "ProcessGroupCollection"
) -> torch.Tensor:
    """Clip by the global norm, counting each expert shard once and replicated params once."""

    grads = [p for p in module.parameters() if p.grad is not None]
    replicated_sq = torch.zeros((), device=grads[0].device)
    expert_sq = torch.zeros((), device=grads[0].device)
    for parameter in grads:
        squared = parameter.grad.detach().float().pow(2).sum()
        (expert_sq if is_expert_parameter(parameter) else replicated_sq).add_(squared)
    if dist.get_world_size(groups.ep) > 1:
        dist.all_reduce(expert_sq, group=groups.ep)
    total_norm = (replicated_sq + expert_sq).sqrt()
    scale = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for parameter in grads:
        parameter.grad.mul_(scale.to(parameter.grad.dtype))
    return total_norm
