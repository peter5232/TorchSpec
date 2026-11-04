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

# 管理可复用的 CPU pinned/GPU 字节缓冲，以及后台 Mooncake 写入的 Future。
# 缓冲池只负责轮转，不自行判断是否可覆写；调用方必须在复用同一指针前等待对应传输结束。

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from torchspec.utils.logging import logger


# 用 uint8 pinned CPU 张量拥有一段稳定字节内存；RDMA 注册由上层 store 完成，而不是构造器自动完成。
class HostBuffer:
    """
    RDMA-registered host buffer using pinned CPU memory.

    Provides a reusable buffer for zero-copy transfers to Mooncake Store.
    """

    # 按字节分配 page-locked 主机张量并记录地址，使 CUDA 可以直接异步复制到目的内存。
    def __init__(self, size: int):
        self.size = size
        # Allocate pinned CPU memory for RDMA compatibility
        self._tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self._ptr = self._tensor.data_ptr()

    # 返回底层分配首地址供原生传输接口使用；它不是可跨机器直接解引用的 Python 数据。
    @property
    def ptr(self) -> int:
        return self._ptr

    # 把连续张量按字节复制到指定偏移，检查容量并返回实际字节数；non_blocking 只入队，不代表数据已可交给网卡读取。
    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """
        Copy tensor data into this host buffer at the given offset.

        Returns the number of bytes copied.
        """
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        # Get a view into our buffer at the right offset
        host_view = self._tensor[offset : offset + nbytes]

        # PyTorch .copy_() handles CUDA→pinned-CPU directly in one DMA.
        # No need for .cpu() which would create an intermediate unpinned copy.
        # The destination is pinned, so CUDA sources can enqueue the DtoH copy
        # on the caller's copy stream.  AsyncPutManager waits on the recorded
        # copy_done event before handing the buffer to Mooncake.
        host_view.copy_(tensor.view(torch.uint8).view(-1), non_blocking=True)

        return nbytes

    # 释放持有的张量并将地址归零；调用者须已确保没有 CUDA 拷贝或后台 put 还在使用它。
    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0

    # 对象析构时调用同一释放逻辑，作为显式生命周期清理之外的兜底。
    def __del__(self):
        self.free()


# 固定数量缓冲形成环形复用，容量限制给生产速度提供背压，而不是无限积累复制任务。
class HostBufferPool:
    """
    Pool of RDMA-registered host buffers for tensor storage.
    """

    # 仅记录单块大小、块数和当前下标，实际 pinned 内存在 initialize 中申请。
    def __init__(self, buffer_size: int = 4 * 1024**3, pool_size: int = 2):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._buffers: List[HostBuffer] = []
        self._current_idx = 0

    # 一次预分配所有主机缓冲，后续样本复用这些分配以避免反复注册内存。
    def initialize(self) -> None:
        """Pre-allocate all buffers."""
        for _ in range(self.pool_size):
            self._buffers.append(HostBuffer(self.buffer_size))
        logger.info(
            "Initialized host buffer pool: %s x %.1fGB",
            self.pool_size,
            self.buffer_size / (1024**3),
        )

    # 按轮转取下一块，池为空则初始化；它不会等待旧写入，等待责任在 AsyncPutManager。
    def get_buffer(self) -> HostBuffer:
        """Get the next buffer (round-robin)."""
        if not self._buffers:
            self.initialize()
        # 两块缓冲 A/B 可先后装入两批数据；第三批再次拿到 A 时，
        # 上层 wait_for_buffer(A.ptr) 会等第一批传输结束，避免用新数据覆盖网卡还在读的旧数据。
        buf = self._buffers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._buffers)
        return buf

    # 逐块释放主机内存并清空列表；上层先 drain 后台任务才能安全调用。
    def shutdown(self) -> None:
        for buf in self._buffers:
            buf.free()
        self._buffers.clear()


# 按拥有缓冲的基址记录 Future，将 CUDA 拷贝完成等待和 RDMA 写入移到后台线程。
class AsyncPutManager:
    """Runs batch_put_from on a background thread so the caller can return immediately.

    Tracks one in-flight ``Future`` per host-buffer pointer.  Before a buffer is
    reused the caller must call :meth:`wait_for_buffer` to block until the
    previous transfer on that buffer completes.

    Multiple worker threads are used so DtoH-copy and event-synchronization can
    overlap, but ``batch_put_from`` calls are serialized with ``_put_lock``
    because ``MooncakeDistributedStore`` is not thread-safe for concurrent puts.
    """

    # 建立线程池、在途表和首个错误槽；put 锁串行化原生 store 调用，多个线程仍可重叠等待 CUDA 事件。
    def __init__(self, store: Any, max_workers: int = 1, replicate_config: Any = None):
        self._store = store
        self._replicate_config = replicate_config
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="async_put")
        self._in_flight: Dict[int, Future] = {}
        self._last_error: Optional[BaseException] = None
        self._put_lock = threading.Lock()

    # 取出并清空缓存的异常后重新抛出，使异步失败在调用线程可见且不会永久重复报告。
    def check_last_error(self) -> None:
        """Re-raise the first async failure that hasn't been surfaced yet."""
        if self._last_error is not None:
            err = self._last_error
            self._last_error = None
            raise err

    # 只检查已完成的 Future，回收成功项并上报失败，不等待正在传输的缓冲。
    def check_errors(self) -> None:
        """Surface completed async failures without waiting for active puts.

        Successful completed futures are retired here as well, which makes
        their host buffers immediately reusable.  Futures that are still
        running remain in ``_in_flight`` and this method returns immediately.
        """
        self.check_last_error()
        for buffer_ptr, future in list(self._in_flight.items()):
            if not future.done():
                continue
            self._in_flight.pop(buffer_ptr, None)
            try:
                future.result()
            except Exception as exc:
                self._last_error = exc
                self.check_last_error()

    # 阻塞等待指定基址的旧任务，完成后才允许写入新样本；这是缓冲轮转时真正的背压点。
    def wait_for_buffer(self, buffer_ptr: int) -> None:
        """Block until the in-flight transfer using *buffer_ptr* finishes.

        Raises the original exception if the transfer failed.
        """
        future = self._in_flight.pop(buffer_ptr, None)
        if future is None:
            return
        try:
            future.result()
        except Exception as exc:
            self._last_error = exc
            raise

    # 提交指针、长度及可选 CUDA 事件到线程池，以拥有整块内存的基址登记任务，而非每个子张量的偏移地址。
    def submit(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
        owner_buffer_ptr: int,
        wait_event: Optional[Any] = None,
        device_index: Optional[int] = None,
    ) -> None:
        """Submit a ``batch_put_from`` to the background thread.

        *owner_buffer_ptr* is the base pointer of the host buffer that owns the
        staged data — used as the key for :meth:`wait_for_buffer`.

        *wait_event* is an optional CUDA event to synchronize on before the
        RDMA transfer (used when DtoH staging runs on a separate stream).

        *device_index* is the CUDA device ordinal that owns *wait_event*.
        Worker threads do not inherit the caller's device context, so the
        worker must call ``torch.cuda.set_device`` before synchronizing.

        GPU tensor lifetime is managed by the caller via
        ``record_stream`` — the CUDA caching allocator keeps the underlying
        memory alive until the copy stream passes the recorded point.
        """
        future = self._executor.submit(
            self._do_put, keys, buffer_ptrs, sizes, wait_event, device_index
        )
        self._in_flight[owner_buffer_ptr] = future

    # 后台线程先选择事件所属 GPU 并等待 DtoH 完成，再在锁内调用 batch_put_from；任一写入失败就尽力清理整批键并报错。
    def _do_put(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
        wait_event: Optional[Any] = None,
        device_index: Optional[int] = None,
    ) -> None:
        if wait_event is not None:
            if device_index is not None:
                torch.cuda.set_device(device_index)
            wait_event.synchronize()
        with self._put_lock:
            if self._replicate_config is not None:
                results = self._store.batch_put_from(
                    keys, buffer_ptrs, sizes, config=self._replicate_config
                )
            else:
                results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
        failures = [(k, r) for k, r in zip(keys, results) if r != 0]
        if failures:
            try:
                self._store.batch_remove(keys, force=True)
            except Exception:
                logger.warning(
                    "Failed to cleanup keys after async batch_put_from failure: %s",
                    keys,
                    exc_info=True,
                )
            detail = ", ".join(f"{k} (code={r})" for k, r in failures)
            raise RuntimeError(f"async batch_put_from failed: {detail}")

    # 等待全部在途任务，把首个异常缓存给后续 check_last_error，同时清空在途表；本方法自身只记录异常。
    def drain(self) -> None:
        """Wait for every in-flight transfer to finish."""
        for ptr, future in list(self._in_flight.items()):
            try:
                future.result()
            except Exception as exc:
                if self._last_error is None:
                    self._last_error = exc
                logger.warning("async put error during drain: %s", exc)
        self._in_flight.clear()

    # 先排空已提交工作再关闭线程池，保证缓冲释放前不再有后台读访问。
    def shutdown(self) -> None:
        """Drain pending work and shut down the thread pool."""
        self.drain()
        self._executor.shutdown(wait=True)


# GPU 发送和接收缓冲的共同拥有者；以 uint8 表示原始字节，具体 dtype 在解释张量时恢复。
class _GPUBuffer:
    """Base class for RDMA-registered GPU buffers (send and receive).

    Provides allocation, pointer access, free, and lazy initialization.
    Subclasses add direction-specific methods (copy_from_tensor, get_slice).
    """

    _label: str = "GPU"  # overridden by subclasses for log messages

    # 保存字节容量和设备但不立即分配显存；首地址为零表示尚未初始化。
    def __init__(self, size: int, device: torch.device = None):
        self.size = size
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        self._tensor: Optional[torch.Tensor] = None
        self._ptr: int = 0
        self._initialized = False

    # 在指定设备申请连续字节张量，初始化标志避免重复申请；注册 RDMA 仍由 store 负责。
    def initialize(self) -> None:
        if self._initialized:
            return
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()
        self._initialized = True
        logger.info(
            "Initialized %s buffer: %.1fMB on %s",
            self._label,
            self.size / (1024**2),
            self.device,
        )

    # 返回 GPU 内存基址，供已经注册该分配的传输引擎使用。
    @property
    def ptr(self) -> int:
        return self._ptr

    # 清空显存张量、地址和初始化标志；释放前必须由上层完成传输同步。
    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0
            self._initialized = False

    # 析构时回收显存引用，不能替代上层对在途 RDMA 的等待。
    def __del__(self):
        self.free()


# 承接远端张量的可复用接收区；切片是共享内存视图，下一次接收可覆盖旧内容。
class GPUReceiveBuffer(_GPUBuffer):
    """RDMA-registered GPU buffer for receiving tensors via GPU Direct RDMA.

    Allocates contiguous GPU memory that can be registered with Mooncake
    for direct RDMA transfers without CPU involvement.
    """

    _label = "GPU receive"

    # 返回 [offset, offset+size) 的字节视图，不复制；上层随后 view(dtype) 和 reshape 解释形状。
    def get_slice(self, offset: int, size: int) -> torch.Tensor:
        """Get a slice of the buffer as a tensor view."""
        if not self._initialized:
            raise RuntimeError("GPU buffer not initialized")
        return self._tensor[offset : offset + size]


# 先用设备内复制把样本放到已注册显存，再供网卡读取，省去隐藏张量的 GPU→CPU 中转。
class GPUSendBuffer(_GPUBuffer):
    """RDMA-registered GPU buffer for sending tensors via GPU Direct RDMA.

    Pre-allocates contiguous GPU memory registered with Mooncake so that
    put operations can stage tensors with a fast GPU-to-GPU copy (~5ms for 1.2GB)
    instead of GPU-to-CPU (~150ms). The NIC reads directly from GPU memory via
    nvidia_peermem (GPU Direct RDMA).
    """

    _label = "GPU send"

    # 检查初始化和容量，按字节把同设备连续源张量拷到发送区；调用返回仅表示 CUDA 操作已提交。
    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """Copy tensor data into this GPU buffer at the given offset.

        The source tensor must be on the same CUDA device. This is a fast
        GPU-to-GPU copy within device memory.

        Returns the number of bytes copied.
        """
        if not self._initialized:
            raise RuntimeError("GPU send buffer not initialized")

        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()

        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        gpu_view = self._tensor[offset : offset + nbytes]
        gpu_view.copy_(tensor.view(torch.uint8).view(-1))

        return nbytes
