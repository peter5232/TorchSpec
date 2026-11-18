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
    # bucket 将若干参数梯度拼成通信缓冲；不是序列 token 的 batch，也不是一次 optimizer.step。
    # 例如 DP=2、SP=2，四 rank 对同一标量参数的局部贡献为 [1,3,2,6]，
    # hook 得到 (1+3+2+6)/2=6，即先各样本合并 SP，再平均两个数据副本。
    # 如果沿用四 rank 平均则是 3，额外除掉了 SP=2；这里没有再除一次 SP。
    tensor = bucket.buffer()
    tensor.div_(state["divide_factor"])
    # async_op 返回通信工作句柄，get_future/then 把结果接回 DDP 的依赖链。
    # Python hook 返回时不代表通信已结束；DDP 使用该 Future 协调梯度就绪，
    # 不能把它当作“可立刻在另一线程直接读 bucket”的通知。
    return (
        dist.all_reduce(tensor, group=state["process_group"], async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


# 上下文内将新参数注册到指定设备；退出时 finally 恢复全局 register_parameter。
# include_buffers=True 已直接使用 device 上下文并提前返回，下方保留的 buffer patch 分支通常不被执行。
@contextmanager
def _init_on_device(device: torch.device, include_buffers: Optional[bool] = False):
    # include_buffers=True 直接用设备上下文，连构造中的 tensor 工厂默认设备也受影响。
    # False 路径则在 Parameter 注册时转 meta，构造参数前的临时 tensor 仍可能真实分配；
    # 所以 init_empty_weights 的“无权重存储”不等于整个构造过程绝对零内存分配。
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
        # meta tensor 记录形状/dtype/stride，没有可复制的参数数值；后续不能靠 .to(cuda)
        # 从中“找回初始化值”。这一步还保留 requires_grad，让冻结 embedding 不被误解冻。
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

    # Trainer 先在全局 rank 0 的 CPU 上构造/读取真权重，其他 rank 用 meta 构造。
    # rank 0 的 full_state 在分布式包装前捕获；这里的 to_empty 只分配未初始化存储，
    # 后面的 set_model_state_dict 才赋予它训练所需数值，不能在两步之间做 forward。
    if dist.get_rank() == 0:
        model = model.to(device=torch.cuda.current_device(), non_blocking=True)
    else:
        model = model.to_empty(device=torch.cuda.current_device())

    # 注意本函数的开关按“是否 None”判定，直接传 False 也会启用这里的 offload。
    # 当前 trainer 调用点明确传 True 或 None；不要套用 apply_fsdp2 中 bool 开关的直觉。
    # device_mesh 虽在签名里，但函数体未读取它；显式 buffer 广播也没有传 group。
    is_cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True
    )

    set_model_state_dict(model, full_state, options=options)

    # named_buffers 还会枚举不在常规 state_dict 中的 non-persistent buffer，故再统一广播。
    # src=0 是全局 rank 0，使用默认进程组；各 rank 必须枚举相同顺序、兼容形状的 buffer。
    # 这不是沿传入 device_mesh 只广播给某个局部 SP/DP 子组。
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
        # REPLICATE 分支在下面直接 return，不应用 FULL_SHARD 的精度/offload policy，
        # modules_to_shard 在此分支也不参与包装。模型实际 dtype 由此前 trainer 的转换决定。
        replicate(model, device_mesh=mesh)
        if args is not None and getattr(args, "attention_backend", None) == "usp":
            sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
            if sp_size > 1:
                process_group = mesh.get_group() if mesh is not None else dist.group.WORLD
                # 例如 world=8、SP=4，实际独立数据副本数为 2。
                # SP 的四份梯度贡献应相加，数据副本间才平均，所以通信 hook 用除数 2 而不是 8。
                # 分母依赖调用者传入覆盖 DP×SP 的 mesh。Eagle3Trainer 使用 grad_sync_mesh，
                # 若误传仅跨数据副本的 dp_mesh，这个整数除法不会自动识别拓扑错误。
                # 下方只检查除数正数，未校验 group_size 能被 SP 整除或成员确实覆盖所有片段。
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

    # 此分支没有注册上面 USP 的“SUM/DP”hook，也未在本函数修改 FSDP 的梯度除数。
    # 所以不能从 REPLICATE 的推导直接声称任意 USP+FULL_SHARD 组合拥有相同归一化语义。
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
    # 这是按模块建立参数通信单元，不是把网络层分派给不同 GPU 的 pipeline parallel。
    # 各 rank 仍执行模型各层；分片参数在需要计算时聚合，根包装再管理未归入子单元的参数。
    for module in modules_to_shard or []:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
