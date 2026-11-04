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

# 通用 Mooncake 客户端拥有网络连接、已注册缓冲、复制流和后台写入线程池。
# 子类定义具体样本字段；本层提供初始化、键可见性等待和已注册多片段发布，协调资源生命周期。

import threading
import time
from abc import ABC
from typing import Any, Dict, Optional, Sequence

import torch
from mooncake.store import MooncakeDistributedStore

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.transfer.mooncake.buffers import (
    AsyncPutManager,
    GPUReceiveBuffer,
    GPUSendBuffer,
    HostBufferPool,
)
from torchspec.utils.logging import logger


# 存储基础设施的拥有者，统一记录所有可选 CPU/GPU 路径资源，避免由每个后端各自重复管理。
class MooncakeHiddenStateStore(ABC):
    """
    Base class for Mooncake Store wrapper to store hidden states from target model.

    Uses RDMA-registered host buffers (MooncakeHostMemAllocator) and put_from
    for zero-copy transfers. Optionally uses GPU Direct RDMA for both sending
    and receiving.
    """

    # 只保存配置与空资源，Event 表示 setup 完成，注册表按指针记录已交给传输引擎的内存。
    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._store: Optional[MooncakeDistributedStore] = None
        self._initialized = False
        self._init_event = threading.Event()
        self._registered_buffers: Dict[int, int] = {}
        self._host_buffer_pool: Optional[HostBufferPool] = None
        self._async_put_manager: Optional[AsyncPutManager] = None
        self._gpu_receive_buffer: Optional[GPUReceiveBuffer] = None
        self._gpu_send_buffer: Optional[GPUSendBuffer] = None
        self._gpu_direct_available = False
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._replicate_config: Any = None

    # 连接服务、检查强制删除支持、配置 hard pin，再创建主机池和可选 GPU 缓冲；最后设置完成事件。
    # CPU 回放不会创建 CUDA 复制流；网络协议与是否存在 GPU Direct 缓冲是两个独立条件。
    def setup(self, device: torch.device | int | None = None) -> None:
        """Initialize the Mooncake Store client."""
        if self._initialized:
            return

        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)

        if self.config.protocol == "rdma" and not self.config.device_name:
            logger.warning(
                "RDMA protocol with empty device_name will use auto-discovery, "
                "which may fail on hosts with mixed IB subnets. "
                "Set mooncake.device_name to a specific RDMA device (e.g. 'mlx5_0')."
            )

        self.config.apply_env_defaults()
        self._store = MooncakeDistributedStore()
        logger.info(
            "Connecting to Mooncake master at %s, metadata server at %s",
            self.config.master_server_address,
            self.config.metadata_server,
        )
        # 这里连接的是 Mooncake 服务并设置客户端共享 segment/缓冲参数，还没有为具体样本分配对象键。
        result = self._store.setup(
            local_hostname=self.config.local_hostname,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
        )
        if result is not None and result != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake client (error={result}). "
                f"Check that Mooncake master is running at {self.config.master_server_address} "
                f"and metadata server is available at {self.config.metadata_server}"
            )

        self._verify_force_delete()
        self._build_replicate_config()

        # 发送端预分配有限 pinned 缓冲并逐个注册；池大小同时限制可不等待地轮转的样本批次数。
        pool_size = self.config.async_put_pool_size
        if pool_size > 0:
            self._host_buffer_pool = HostBufferPool(
                buffer_size=self.config.host_buffer_size,
                pool_size=pool_size,
            )
            self._host_buffer_pool.initialize()

            for buf in self._host_buffer_pool._buffers:
                self._register_buffer(buf.ptr, buf.size)

            self._async_put_manager = AsyncPutManager(
                store=self._store, max_workers=pool_size, replicate_config=self._replicate_config
            )
            logger.info("Async put manager created (pool_size=%d)", pool_size)

        # GPU Direct 是可选能力，失败不应把整个客户端误当已损坏；发送和接收各自检查缓冲是否可用。
        if self.config.enable_gpu_direct and torch.cuda.is_available():
            self._setup_gpu_direct(device)

        # Can't create copy stream for "cpu" device (offline replay engine).
        if torch.cuda.is_available() and (device is None or device.type == "cuda"):
            cuda_device = device if device is not None else torch.device("cuda")
            self._copy_stream = torch.cuda.Stream(device=cuda_device)
            logger.info("DtoH copy stream created on %s", cuda_device)

        # 所有资源准备完成后才设置事件，其他线程可据此进入 put/get；提前置位会暴露半初始化状态。
        self._initialized = True
        self._init_event.set()
        logger.info(
            "Mooncake Store client initialized (protocol=%s, device=%s, gpu_direct=%s, pool_size=%d)",
            self.config.protocol,
            self.config.device_name or "(auto-discovery)",
            self._gpu_direct_available,
            pool_size,
        )

    # 分别分配并注册接收/发送显存；接收注册失败则放弃，发送失败仍可保留接收快路径。
    def _setup_gpu_direct(self, device: torch.device = None) -> None:
        """Initialize GPU send/receive buffers and register for GPU Direct RDMA."""
        try:
            self._gpu_receive_buffer = GPUReceiveBuffer(
                size=self.config.gpu_buffer_size,
                device=device,
            )
            self._gpu_receive_buffer.initialize()

            # 先保证接收缓冲可注册；GPU 内存存在并不代表网卡能直接访问，平台和驱动条件仍可能拒绝注册。
            if not self._register_buffer(
                self._gpu_receive_buffer.ptr, self._gpu_receive_buffer.size
            ):
                logger.warning("Failed to register GPU receive buffer with Mooncake")
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None
                return

            # Allocate GPU send buffer (same size as host buffer — put is 1 sample)
            # 发送区按一批 staging 的 host_buffer_size 配置，而接收区使用独立 gpu_buffer_size，容量用途不同。
            self._gpu_send_buffer = GPUSendBuffer(
                size=self.config.host_buffer_size,
                device=device,
            )
            self._gpu_send_buffer.initialize()

            if not self._register_buffer(self._gpu_send_buffer.ptr, self._gpu_send_buffer.size):
                logger.warning("Failed to register GPU send buffer with Mooncake")
                self._gpu_send_buffer.free()
                self._gpu_send_buffer = None
                # receive buffer is still usable, continue

            # 这个标志主要表示接收路径可用，不能据它独自断言发送区存在；发布端还检查 _gpu_send_buffer。
            self._gpu_direct_available = True
            send_desc = (
                f"{self._gpu_send_buffer.size / (1024**2):.1f}MB"
                if self._gpu_send_buffer
                else "N/A"
            )
            logger.info(
                "GPU Direct RDMA enabled: receive=%.1fMB, send=%s on %s",
                self._gpu_receive_buffer.size / (1024**2),
                send_desc,
                device or "cuda",
            )

        except Exception as e:
            logger.warning("Failed to setup GPU Direct RDMA: %s", e)
            self._gpu_direct_available = False
            if self._gpu_receive_buffer is not None:
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None
            if self._gpu_send_buffer is not None:
                self._gpu_send_buffer.free()
                self._gpu_send_buffer = None

    # 先看完成事件，再有限等待可能的后台 setup；超时且仍未初始化时由当前线程执行 setup。
    def _ensure_initialized(self, timeout: float = 30.0) -> None:
        """Block until setup() has completed (thread-safe via Event).

        If setup() was called from a background thread, this will wait up to
        *timeout* seconds for it to finish.  If nobody has called setup() yet,
        falls back to calling it on the current thread.
        """
        if self._init_event.is_set():
            return
        if not self._init_event.wait(timeout=timeout):
            if not self._initialized:
                self.setup()

    # 同指针已登记即复用；否则调用原生注册接口并只在返回成功时记录，失败返回 False 供上层回退。
    def _register_buffer(self, buffer_ptr: int, size: int) -> bool:
        """Register a buffer for RDMA transfers."""
        if buffer_ptr in self._registered_buffers:
            return True

        try:
            if hasattr(self._store, "register_buffer"):
                result = self._store.register_buffer(buffer_ptr, size)
                if result == 0:
                    self._registered_buffers[buffer_ptr] = size
                    logger.debug("Registered buffer at %#x, size=%s", buffer_ptr, size)
                    return True
                logger.warning("register_buffer returned error code: %s", result)
                return False
        except Exception as e:
            logger.warning("Failed to register buffer: %s", e)

        return False

    # 检查当前客户端是否提供多源片段写入接口，决定能否从 vLLM 分页缓存直接发布。
    @property
    def supports_multi_buffer_put(self) -> bool:
        """Whether this Mooncake client can build one object from many buffers."""
        self._ensure_initialized()
        return callable(getattr(self._store, "batch_put_from_multi_buffers", None))

    # 登记调用方拥有的分配；所有权不转移，调用方必须让内存活到注销成功或 store 关闭。
    def register_external_buffer(self, buffer_ptr: int, size: int) -> bool:
        """Register caller-owned memory for zero-copy Mooncake operations.

        The caller retains ownership and must keep the allocation alive until
        the store is closed (or :meth:`unregister_external_buffer` succeeds).
        Sub-ranges of the registered allocation may subsequently be passed to
        ``batch_put_from_multi_buffers``.
        """
        self._ensure_initialized()
        return self._register_buffer(buffer_ptr, size)

    # 注销已登记外部地址，只有原生调用成功才从注册表删除，避免失败时误认为可以安全释放。
    def unregister_external_buffer(self, buffer_ptr: int) -> bool:
        """Unregister caller-owned memory previously registered with Mooncake."""
        self._ensure_initialized()
        if buffer_ptr not in self._registered_buffers:
            return True
        try:
            result = self._store.unregister_buffer(buffer_ptr)
        except Exception as exc:
            logger.warning("Failed to unregister buffer at %#x: %s", buffer_ptr, exc)
            return False
        if result not in (None, 0):
            logger.warning("unregister_buffer returned error code %s for %#x", result, buffer_ptr)
            return False
        self._registered_buffers.pop(buffer_ptr, None)
        return True

    # 校验每个对象的片段指针/长度列表，再同步把多页拼成一个对象；任一对象失败则清理本批并抛错。
    # 指针指向调用方注册过的内存，本方法不复制或保有源 tensor，生命周期由调用者负责。
    def put_from_registered_multi_buffers(
        self,
        keys: Sequence[str],
        all_buffer_ptrs: Sequence[Sequence[int]],
        all_sizes: Sequence[Sequence[int]],
    ) -> None:
        """Publish objects assembled from registered memory fragments.

        This is the scatter-source counterpart of ``batch_put_from``.  It is
        used by the vLLM PP connector to publish one layer directly from its
        paged hidden-state cache without first packing the blocks into a host
        or GPU staging buffer.
        """
        self._ensure_initialized()
        key_list = list(keys)
        ptr_lists = [list(ptrs) for ptrs in all_buffer_ptrs]
        size_lists = [list(sizes) for sizes in all_sizes]
        if not key_list or len(key_list) != len(ptr_lists) or len(key_list) != len(size_lists):
            raise ValueError(
                "Expected equal non-empty keys/pointer-lists/size-lists, got "
                f"{len(key_list)}/{len(ptr_lists)}/{len(size_lists)}"
            )
        # 每对象的源片段必须一一对应且字节数为正；不同对象允许拥有不同数量的页。
        for key, ptrs, sizes in zip(key_list, ptr_lists, size_lists):
            if not ptrs or len(ptrs) != len(sizes):
                raise ValueError(
                    f"Mooncake object {key!r} has {len(ptrs)} pointers and {len(sizes)} sizes"
                )
            if any(size <= 0 for size in sizes):
                raise ValueError(f"Mooncake object {key!r} has a non-positive fragment size")

        # 能力检查在实际调用处再做一次，避免老客户端在深层原生调用里给出不明确失败。
        batch_put = getattr(self._store, "batch_put_from_multi_buffers", None)
        if not callable(batch_put):
            raise RuntimeError(
                "Mooncake batch_put_from_multi_buffers is unavailable; "
                "mooncake-transfer-engine >= 0.3.12.post1 is required for "
                "direct paged hidden-state publication"
            )
        if self._replicate_config is not None:
            results = list(
                batch_put(key_list, ptr_lists, size_lists, config=self._replicate_config)
            )
        else:
            results = list(batch_put(key_list, ptr_lists, size_lists))
        # 原生结果必须覆盖全部键，否则不能把未报告的对象默认当成功。
        if len(results) != len(key_list):
            raise RuntimeError(
                "batch_put_from_multi_buffers returned "
                f"{len(results)} results for {len(key_list)} keys"
            )
        failures = [(key, result) for key, result in zip(key_list, results) if result != 0]
        if not failures:
            return
        try:
            # 部分写入成功也尽力一起回收；读者需要完整样本，不应消费一次失败发布遗留的残片。
            self._store.batch_remove(key_list, force=True)
        except Exception:
            logger.warning(
                "Failed to cleanup keys after batch_put_from_multi_buffers failure: %s",
                key_list,
                exc_info=True,
            )
        detail = ", ".join(f"{key} (code={result})" for key, result in failures)
        raise RuntimeError(f"batch_put_from_multi_buffers failed: {detail}")

    # 借用一块主机缓冲做 4 KiB 写入和删除，提前触发数据路径初始化；没有主机池则跳过。
    def warmup_rdma(self) -> None:
        """Do a small test PUT to warm up the RDMA data path."""
        import uuid

        self._ensure_initialized()
        if self._host_buffer_pool is None:
            logger.debug("Skipping RDMA warmup — no host buffer pool (get-only store)")
            return
        key = f"_warmup_{uuid.uuid4().hex[:8]}"
        buf = self._host_buffer_pool.get_buffer()
        size = 4096
        self._store.batch_put_from([key], [buf.ptr], [size])
        self._store.batch_remove([key], force=True)
        logger.info("RDMA warmup complete")

    # 单键轻量探测只读元数据；异常折成 False，因此它适合探测，不能区分服务失败和确实缺键。
    def exists(self, key: str) -> bool:
        """Check if a key exists in the store (metadata-only, no data download)."""
        try:
            result = self._store.is_exist(key)
            return result == 1
        except Exception:
            return False

    # 优先单次批量存在性 RPC，旧客户端逐键回退；校验结果数量并保留调用异常，供读取端可靠等待。
    def batch_exists(self, keys: Sequence[str]) -> Dict[str, bool]:
        """Return a metadata-only existence census for *keys*.

        Newer Mooncake clients provide a one-RPC ``batch_is_exist`` API.  Keep
        a per-key fallback for older supported clients, while preserving
        errors so readers fail closed rather than mistaking a metadata failure
        for a missing object.
        """
        self._ensure_initialized()
        key_list = list(keys)
        batch_is_exist = getattr(self._store, "batch_is_exist", None)
        if callable(batch_is_exist):
            results = list(batch_is_exist(key_list))
            if len(results) != len(key_list):
                raise RuntimeError(
                    "Mooncake batch_is_exist returned "
                    f"{len(results)} results for {len(key_list)} keys"
                )
        else:
            results = [self._store.is_exist(key) for key in key_list]
        return {key: result == 1 for key, result in zip(key_list, results)}

    # 用单调时钟轮询全部必需键，超时明确列出缺项；timeout<=0 表示持续等待而不是立即失败。
    # 只有 _hs 可见而 _ids 未完成时不能读取半条样本，PP 多层清单也需要类似完整性检查。
    def wait_for_keys(
        self,
        keys: Sequence[str],
        *,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> None:
        """Wait until every key is visible in Mooncake metadata.

        This is deliberately called before either GPUDirect or host-buffer
        byte movement.  A timeout reports the exact missing keys, which turns
        an incomplete pipeline fragment into a fail-closed sample.
        """
        key_list = list(keys)
        if not key_list:
            return
        if timeout is None:
            timeout = self.config.get_retry_max_wait_seconds
        if poll_interval is None:
            poll_interval = self.config.get_retry_wait_seconds
        poll_interval = max(float(poll_interval), 0.001)
        timeout = float(timeout)
        start = time.monotonic()
        deadline = None if timeout <= 0 else start + timeout
        last_missing = key_list

        while True:
            census = self.batch_exists(key_list)
            last_missing = [key for key in key_list if not census[key]]
            if not last_missing:
                return
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                missing = ", ".join(last_missing)
                raise TimeoutError(
                    "Timed out waiting for Mooncake keys after "
                    f"{now - start:.3f}s; missing: {missing}"
                )
            sleep_for = poll_interval
            if deadline is not None:
                sleep_for = min(sleep_for, max(deadline - now, 0.0))
            time.sleep(sleep_for)

    # 不排空写入队列，只把已完成后台任务的异常带回当前调用栈，适合推理调度步中的轻量检查。
    def check_async_errors(self) -> None:
        """Surface completed background put failures without draining puts."""
        self._ensure_initialized()
        if self._async_put_manager is not None:
            self._async_put_manager.check_errors()

    # 先看包版本，再为非标准安装检查接口说明是否包含 force；不支持立即删除时在启动阶段失败。
    def _verify_force_delete(self) -> None:
        """Fail-fast if Mooncake doesn't support batch_remove(force=True).

        Requires mooncake-transfer-engine >= 0.3.10.post1.
        Primary check uses package version metadata; falls back to docstring
        heuristic for non-pip installs.
        """
        batch_remove = getattr(self._store, "batch_remove", None)
        if batch_remove is None:
            raise RuntimeError(
                "Mooncake version too old: batch_remove() not found. "
                "Requires mooncake-transfer-engine >= 0.3.10.post1."
            )
        try:
            from importlib.metadata import version

            from packaging.version import Version

            installed = Version(version("mooncake-transfer-engine"))
            if installed >= Version("0.3.10.post1"):
                return
        except Exception:
            pass
        doc = getattr(batch_remove, "__doc__", "") or ""
        if "force" not in doc:
            raise RuntimeError(
                "Mooncake version too old: batch_remove(force=True) not supported. "
                "Requires mooncake-transfer-engine >= 0.3.10.post1."
            )

    # 仅在请求 hard pin 且当前客户端支持时构造复制配置，否则记录降级原因并保留普通发布。
    def _build_replicate_config(self) -> None:
        """Build ReplicateConfig for batch_put_from if hard_pin is enabled and supported."""
        self._replicate_config = None
        if not self.config.enable_hard_pin:
            return
        try:
            from mooncake.store import ReplicateConfig

            cfg = ReplicateConfig()
            if hasattr(cfg, "with_hard_pin"):
                cfg.with_hard_pin = True
                self._replicate_config = cfg
                logger.info("Hard pin enabled for batch_put_from")
            else:
                logger.warning(
                    "enable_hard_pin=True but ReplicateConfig lacks with_hard_pin attr "
                    "(needs unreleased Mooncake)"
                )
        except ImportError:
            logger.warning("enable_hard_pin=True but ReplicateConfig not importable")

    # 先停止并等待后台写入，再释放 GPU/主机缓冲和复制流，关闭客户端并清除就绪事件。
    def close(self) -> None:
        """Close the Mooncake Store client."""
        if self._async_put_manager is not None:
            self._async_put_manager.shutdown()
            self._async_put_manager = None
        if self._gpu_send_buffer is not None:
            self._gpu_send_buffer.free()
            self._gpu_send_buffer = None
        if self._gpu_receive_buffer is not None:
            self._gpu_receive_buffer.free()
            self._gpu_receive_buffer = None
        if self._host_buffer_pool is not None:
            self._host_buffer_pool.shutdown()
            self._host_buffer_pool = None
        self._copy_stream = None
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False
        self._init_event.clear()
        self._gpu_direct_available = False
