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

# 内存估算和观测辅助：形状元数据估算传输量，CUDA API 区分设备可用内存与 PyTorch 分配/保留量。

import gc
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from torchspec.utils.logging import logger

DTYPE_SIZES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.bool: 1,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "bool": 1,
}


# 逐字段计算各维乘积×dtype 字节数后求和；缺失或未识别 dtype 按 2 字节处理。
# 它估算张量净载荷，不含 allocator 对齐、通信注册和模型额外开销。
def estimate_tensor_bytes(tensor_shapes: Dict[str, Tuple[int, ...]], tensor_dtypes: Dict) -> int:
    """Calculate bytes from tensor shapes and dtypes."""
    total = 0
    for name, shape in tensor_shapes.items():
        dtype = tensor_dtypes.get(name, torch.bfloat16)
        elem_size = DTYPE_SIZES.get(dtype, 2)
        numel = 1
        for dim in shape:
            numel *= dim
        total += numel * elem_size
    return total


# 等待 CUDA 完成后触发 Python GC 并清 CUDA 缓存；可选清 pinned host 缓存，会引入同步开销。
def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


# 查询当前 CUDA 设备总/空闲内存，以及 PyTorch allocated/reserved，返回统一展示字典。
def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


# 除 1024**3 并保留两位小数，实际使用二进制 GiB 口径，虽展示字段名为 GB。
def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


# 可先清缓存，再按 rank 记录并返回内存字典；必须各 rank 看，因为分片或不同任务占用可能不同。
def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
