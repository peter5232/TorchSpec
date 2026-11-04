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

# 参数分配与分布式包装：非主 rank 用 meta 建模，再由 rank 0 广播；支持 REPLICATE 和 FULL_SHARD。
# USP 的序列分片贡献是同一样本的梯度片段，梯度归约除数必须与普通 DP 区分。

from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from torchspec.utils.logging import logger


# 先将 bucket 除指定 DP 因子，再异步 SUM 归约，返回 future 中的 tensor 给 DDP。
# USP 需要把 SP 的局部贡献相加，而不是再平均一次 SP 数量。
def _allreduce_with_divide_factor_hook(state: dict, bucket: dist.GradBucket):
    tensor = bucket.buffer()
    tensor.div_(state["divide_factor"])
    return (
        dist.all_reduce(tensor, group=state["process_group"], async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


# 上下文内将新参数注册到指定设备；退出时 finally 恢复全局 register_parameter。
# include_buffers=True 已直接使用 device 上下文并提前返回，下方保留的 buffer patch 分支通常不被执行。
@contextmanager
def _init_on_device(device: torch.device, include_buffers: Optional[bool] = False):
    if include_buffers:
        with device:
            yield
        return

    # include_buffers=False 时仅拦截参数注册，普通 buffer 仍按原构造逻辑创建。
    # 例如用于词表映射的整数 buffer 可能需要真实值，不能一概变成只含形状的 meta tensor。
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    # 先按 Module 原语登记参数，再保留参数类型和 requires_grad 将其换成目标设备版本。
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    # 保留 persistent 标志注册 buffer，再搬到目标设备；属于下方保留的补丁实现。
    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    # 创建强制使用目标 device 的构造器闭包，供需要 patch torch 工厂函数的路径使用。
    def patch_tensor_constructor(fn):
        # 转发工厂调用的其他参数，仅覆盖 device；返回原工厂在指定设备创建的 tensor。
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    # 替换的是全局 Module 方法，退出必须无论成功/异常都恢复。
    # 因此它应限于受控模型构造阶段，不宜与其他线程并发创建 Module 混用。
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


# 用 meta 设备包装模型构造，只建立参数形状/类型而不分配实际权重存储。
@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """Context manager that initialises models on the meta device (no memory).

    Args:
        include_buffers: Whether to also put all buffers on the meta device.
    """
    with _init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


# rank 0 搬已有权重，其他 rank to_empty 实体化空存储，再 broadcast_from_rank0 设置全量状态。
# buffer 额外广播；CPU offload 时参数移 CPU，buffer 留在当前 CUDA，满足计算侧常量访问。
def fsdp2_load_full_state_dict(model, full_state, device_mesh, cpu_offload):
    """Load a full state dict into an FSDP2 model, broadcasting from rank 0.

    Args:
        model: FSDP2-wrapped model.
        full_state: State dict (only rank 0 has real weights, others have empty dict).
        device_mesh: Device mesh for FSDP.
        cpu_offload: If not None, enables StateDictOptions cpu_offload.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        model = model.to_empty(device=torch.cuda.current_device())

    is_cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True
    )

    set_model_state_dict(model, full_state, options=options)

    for _name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if is_cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(torch.cuda.current_device())

    return model


# REPLICATE 仅复制模型并归约梯度，USP 可注册除 DP 数而非总 rank 数的通信 hook。
# FULL_SHARD 按给定子模块先分片再包根模型，并分别配置参数计算 dtype 和梯度归约 dtype。
def apply_fsdp2(
    model,
    mesh=None,
    cpu_offload=False,
    args=None,
    modules_to_shard: Optional[List[nn.Module]] = None,
):
    """Apply FSDP v2 or DDP to a model.

    Args:
        model: The model to wrap with FSDP/DDP.
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states to CPU.
        args: Arguments containing precision settings (fp16/bf16) and fsdp_strategy.
        modules_to_shard: Explicit list of sub-modules to individually shard
            before sharding the root model.  When *None* the root model is
            sharded as a single unit.
    """
    from torch.distributed._composable.replicate import replicate
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )

    strategy = getattr(args, "fsdp_strategy", "REPLICATE") if args else "REPLICATE"
    strategy = strategy.upper()

    if strategy == "REPLICATE":
        logger.info("Using REPLICATE strategy (DDP-like, gradient all-reduce only)")
        replicate(model, device_mesh=mesh)
        if args is not None and getattr(args, "attention_backend", None) == "usp":
            sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
            if sp_size > 1:
                process_group = mesh.get_group() if mesh is not None else dist.group.WORLD
                # 例如 world=8、SP=4，实际独立数据副本数为 2。
                # SP 的四份梯度贡献应相加，数据副本间才平均，所以通信 hook 用除数 2 而不是 8。
                divide_factor = dist.get_world_size(process_group) // sp_size
                if divide_factor <= 0:
                    raise ValueError(
                        f"Invalid USP grad divide factor: world_size="
                        f"{dist.get_world_size(process_group)}, sp_size={sp_size}"
                    )
                model.register_comm_hook(
                    {
                        "process_group": process_group,
                        "divide_factor": divide_factor,
                    },
                    _allreduce_with_divide_factor_hook,
                )
                logger.info(
                    "Registered USP replicate grad hook "
                    f"(all_reduce / {divide_factor}, sp_size={sp_size})"
                )
        return model
    elif strategy != "FULL_SHARD":
        raise ValueError(f"Unknown fsdp_strategy: {strategy}. Use 'FULL_SHARD' or 'REPLICATE'")

    logger.info("Using FULL_SHARD strategy (FSDP, sharded parameters)")

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    param_dtype = torch.bfloat16
    reduce_dtype_str = getattr(args, "fsdp_reduce_dtype", "float32") if args else "float32"
    # 参数 BF16 降低计算和存储成本，归约 FP32 则减少跨 rank 累加误差。
    # 这两个 dtype 不必相同；将归约改 BF16 是通信/数值权衡，而非模型前向 dtype 的必然结果。
    reduce_dtype = torch.bfloat16 if reduce_dtype_str == "bfloat16" else torch.float32

    if args is not None and getattr(args, "fp16", False):
        param_dtype = torch.float16

    logger.info(
        f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}"
    )

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # 先包指定子模块，再包根模型，形成可在层边界释放/重取参数的分片单元。
    # 不给子模块列表时整模型是一个单元，逻辑仍可工作，但峰值内存/通信重叠特征会不同。
    for module in modules_to_shard or []:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
