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

# Eagle3 样本的存储协议：一个逻辑键拆成 _hs、_ids 和可选 _lhs/_tgt，shape/dtype 通过元数据传递。
# 这里负责精度统一、字节打包、异步发布、读回和删除；控制面拿到键不自动等于远端所有字节都已发布。

import ctypes
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from torchspec.transfer.mooncake.helpers import _format_bytes
from torchspec.transfer.mooncake.store import MooncakeHiddenStateStore
from torchspec.utils.distributed import get_usp_rank_coords
from torchspec.utils.logging import logger
from torchspec.utils.usp import split_usp_batch

if TYPE_CHECKING:
    from torchspec.models.target.eagle3_target_model import Eagle3TargetOutput

# Static lookup for dtype → element size in bytes (avoids creating a tensor
# on every call to _compute_tensor_size).
_DTYPE_ELEMENT_SIZES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}

# Canonical dtype for hidden-state tensors written to / read from Mooncake.
HIDDEN_STATES_STORAGE_DTYPE = torch.bfloat16


# 在通用 Mooncake 客户端之上定义训练张量字段与后缀，让不同推理后端共享训练侧读取契约。
class EagleMooncakeStore(MooncakeHiddenStateStore):
    """
    Mooncake Store wrapper specialized for Eagle3 hidden states.

    Uses RDMA-registered host buffers and put_from for zero-copy transfers.
    Each Eagle3 output is stored as multiple tensors with key suffixes:
    - {key}_hs: hidden_states
    - {key}_tgt: target
    - {key}_ids: input_ids
    - {key}_lhs: last_hidden_states (if present)

    Deletions use ``batch_remove(force=True)`` for immediate cleanup.
    """

    TENSOR_SUFFIXES = ["_hs", "_tgt", "_ids", "_lhs"]

    # 为已经规范好 dtype 的任意键/张量组提供批量发布入口；检查数量对应，但不做 put 的 BF16 转换。
    def put_raw_tensors(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        """Publish an already-normalized group of tensors in one batch.

        The vLLM completed-request connector uses this to pack fragments from
        multiple requests into each registered host buffer. This avoids one
        serialized ``batch_put_from`` call per layer while retaining the same
        per-layer Mooncake keys consumed by ``MooncakeDataset``.
        """
        self._ensure_initialized()
        if not keys or len(keys) != len(tensors):
            raise ValueError(
                f"Expected equal non-empty keys/tensors, got {len(keys)}/{len(tensors)}"
            )
        self._put_raw_tensors(keys, tensors)

    # 有 GPU send buffer 时同步发布，否则轮转主机缓冲并异步复制/写入；CPU 回放无需 CUDA 事件。
    # 三个生命周期分别受保护：源 GPU 张量由 record_stream 保活、主机缓冲由 Future 保护、远端对象由消费后删除回收。
    def _put_raw_tensors(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        if self._gpu_direct_available and self._gpu_send_buffer is not None:
            buf = self._gpu_send_buffer
            buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
            # copy_from_tensor() enqueues device-to-device copies on the
            # current stream. Mooncake does not participate in CUDA stream
            # ordering, so make those writes visible before the NIC reads the
            # registered send buffer.
            # GPU Direct 省掉 DtoH，但网卡并不了解 CUDA stream 的排队顺序。
            # 此同步确保设备内 staging 已写完；接下来的同步 put 返回后发送缓冲才能再次覆写。
            torch.cuda.current_stream(buf.device).synchronize()
            self._do_sync_batch_put(keys, buffer_ptrs, sizes)
        elif self._host_buffer_pool is None or self._async_put_manager is None:
            raise RuntimeError(
                "put() requires either GPU Direct (enable_gpu_direct=True) or "
                "async host-buffer puts (async_put_pool_size > 0). "
                "Current config has async_put_pool_size=0 and GPU Direct is "
                f"{'enabled but gpu_send_buffer failed to initialize' if self._gpu_direct_available else 'disabled'}. "
                "Set async_put_pool_size >= 1 or enable GPU Direct."
            )
        else:
            # 主机路径借用下一块预注册内存；取到缓冲并不表示它空闲，下一步必须查旧 Future。
            buf = self._host_buffer_pool.get_buffer()
            self._async_put_manager.check_last_error()
            self._async_put_manager.wait_for_buffer(buf.ptr)

            # Offline replay tensors are already on the host, so CUDA event and
            # stream synchronization is both unnecessary and unavailable.
            # 离线回放的源就在 CPU，直接复制到主机缓冲即可提交；此分支不要求 CUDA 可初始化。
            if not any(tensor.is_cuda for tensor in tensors):
                buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
                self._async_put_manager.submit(keys, buffer_ptrs, sizes, buf.ptr)
                return

            # 跨流依赖分成两步：计算流先记事件，复制流等待它，保证读到最终隐状态；
            # 复制流再记录 copy_done，后台线程等待后才让网卡读取主机缓冲。
            # 仅使用 non_blocking=True 不能建立这两条先后关系。
            compute_event = torch.cuda.Event()
            compute_event.record()

            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(compute_event)
                buffer_ptrs, sizes = self._stage_tensors_into_buffer(buf, tensors)
                copy_done = torch.cuda.Event()
                copy_done.record()

            # 即使 Python 局部 tensor 引用随后离开作用域，record_stream 也让 CUDA 分配器
            # 在复制流完成使用之前不复用源存储；它负责显存生命周期，不负责通知网卡拷贝完成。
            for t in tensors:
                if t.is_cuda:
                    t.record_stream(self._copy_stream)

            # 到此主线程只完成“拷贝已排队”，并没有完成“CPU 字节就绪”或“远端键可读”。
            # submit 把后两步交给线程；HFRunner 在整批末尾 flush，vLLM 则允许消费者等待键。
            self._async_put_manager.submit(
                keys,
                buffer_ptrs,
                sizes,
                buf.ptr,
                wait_event=copy_done,
                device_index=self._copy_stream.device.index,
            )

    # 先统一精度，再使用训练相同的 USP 切分逻辑为每个 SP rank 发布带 _uspN 的对象。
    # dummy loss mask 仅服务切片计算，不是用户真实训练 mask；返回元数据仍描述原始完整样本。
    def put_usp_shards(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor],
        *,
        sp_size: int,
        sp_ring_size: int,
        ttt_length: int,
        max_seq_length: Optional[int],
    ) -> Dict[str, Any]:
        self._ensure_initialized()
        logger.debug("put_usp_shards: starting for key=%s, sp_size=%s", key, sp_size)

        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        # USP 切分需要与输入位置对齐的目标信息；优先用最后隐状态，没有时才取显式 target。
        target_hs = last_hidden_states if last_hidden_states is not None else target
        if target_hs is None:
            raise ValueError("USP sharded Mooncake storage requires last_hidden_states or target")

        # split_usp_batch 还负责训练展开所需的位置/重叠规则；不要用 torch.chunk(S, sp_size) 替代。
        # 返回的 attention/loss/position 在这里不上传：该阶段只发布训练端会按相同规则解释的张量。
        if input_ids.dim() == 2:
            loss_mask_len = input_ids.shape[-1]
        else:
            loss_mask_len = input_ids.shape[0]
        dummy_loss_mask = torch.ones((1, loss_mask_len), dtype=torch.long, device=input_ids.device)
        # SP 总度数拆成 Ulysses 与 ring 两个维度，后续位置切片必须与训练进程的 rank 坐标一致。
        sp_ulysses_size = max(1, sp_size // sp_ring_size)

        # 生产端逐个写出所有训练 SP rank 的分片，减少消费者各自下载完整序列再切开的重复流量。
        for sp_rank in range(sp_size):
            _, ring_rank = get_usp_rank_coords(
                sp_rank=sp_rank,
                sp_ulysses_size=sp_ulysses_size,
                sp_ring_size=sp_ring_size,
            )
            (
                shard_input_ids,
                _attention_mask,
                _loss_mask,
                shard_hidden_states,
                shard_target_hs,
                _position_ids,
            ) = split_usp_batch(
                input_ids=input_ids,
                loss_mask=dummy_loss_mask,
                hidden_states=hidden_states,
                target_hidden_states=target_hs,
                ttt_length=ttt_length,
                sp_rank=sp_rank,
                sp_size=sp_size,
                ring_rank=ring_rank,
                sp_ring_size=sp_ring_size,
                max_len=max_seq_length,
            )

            # rank 写进键空间，把同一逻辑样本的不同序列切片区分开；字段后缀仍沿用 Eagle 协议。
            shard_key = f"{key}_usp{sp_rank}"
            keys = [f"{shard_key}_hs", f"{shard_key}_ids"]
            tensors = [shard_hidden_states, shard_input_ids]
            if last_hidden_states is not None:
                keys.append(f"{shard_key}_lhs")
                tensors.append(shard_target_hs)
            else:
                keys.append(f"{shard_key}_tgt")
                tensors.append(shard_target_hs)
            self._put_raw_tensors(keys, tensors)

        # 这里返回完整样本形状，而不是最后一次循环的 shard 形状；分片读者还需要 USP 布局元数据。
        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {
            "hidden_states": hidden_states.dtype,
            "input_ids": input_ids.dtype,
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype

        logger.debug("put_usp_shards: completed key=%s, shapes=%s", key, shapes)
        return {"shapes": shapes, "dtypes": dtypes}

    # 把隐状态及可选 target 转 BF16，保留 token 精度，按字段后缀发布并返回转换后的真实 shape/dtype。
    # 返回意味着已提交发布；需要立即读取的调用方应 flush，不能只凭拿到键判断传输完成。
    def put(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Store Eagle3 output tensors via async batch_put_from.

        DtoH staging runs on ``_copy_stream`` so the caller's compute stream
        is never blocked.  The RDMA transfer runs on a background thread via
        ``AsyncPutManager``.  With *pool_size* host buffers the caller almost
        never waits — ``wait_for_buffer`` only blocks when every buffer is
        still in-flight.

        For GPU Direct send the path is synchronous (no DtoH needed).

        Returns a dict with ``"shapes"`` and ``"dtypes"`` sub-dicts that
        reflect the *actually stored* tensors (post dtype-cast).  Callers
        should forward these to consumers so metadata always matches bytes.
        """
        self._ensure_initialized()
        logger.debug("put: starting for key=%s", key)

        # 传输格式统一 BF16，避免后端权重精度变化时训练侧仍按固定字节宽度读取。
        if hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            hidden_states = hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if (
            last_hidden_states is not None
            and last_hidden_states.dtype != HIDDEN_STATES_STORAGE_DTYPE
        ):
            last_hidden_states = last_hidden_states.to(HIDDEN_STATES_STORAGE_DTYPE)
        if target is not None and target.dtype != HIDDEN_STATES_STORAGE_DTYPE:
            target = target.to(HIDDEN_STATES_STORAGE_DTYPE)

        # 必需字段先建立键/张量配对；可选 target 与最后隐状态只有非 None 才真实写入。
        keys = [f"{key}_hs", f"{key}_ids"]
        tensors = [hidden_states, input_ids]

        if target is not None:
            keys.append(f"{key}_tgt")
            tensors.append(target)

        if last_hidden_states is not None:
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)

        self._put_raw_tensors(keys, tensors)

        # 返回的是精度转换之后的 tensor 描述；不要把调用者转换前的 dtype 重新填回元数据。
        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        dtypes = {
            "hidden_states": hidden_states.dtype,
            "input_ids": input_ids.dtype,
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
            dtypes["target"] = target.dtype
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)
            dtypes["last_hidden_states"] = last_hidden_states.dtype

        logger.debug("put: completed key=%s, shapes=%s", key, shapes)
        return {"shapes": shapes, "dtypes": dtypes}

    # 等待所有异步 put 并前后检查缓存异常；因此失败不会在 drain 的日志中被悄悄吞掉。
    def flush(self) -> None:
        """Block until all in-flight async puts have completed.

        Called before returning mooncake keys to the controller so that
        consumers can GET immediately.  Because ``put()`` stages DtoH on
        ``_copy_stream``, the copies are typically finished by the time
        this is called — the wait is only for the (fast) RDMA transfer.
        """
        self._ensure_initialized()
        if self._async_put_manager is None:
            return
        # 已缓存错误会在 drain 前立刻抛出，所以“调用过 flush”本身不证明全部任务已排空；
        # 只有正常返回才具备本次检查的成功语义，失败后的资源收尾仍由 close/shutdown 负责。
        self._async_put_manager.check_last_error()
        self._async_put_manager.drain()
        self._async_put_manager.check_last_error()

    # 同步发送指针列表，逐键检查返回码；部分失败时尽力删除整批并附带容量参数报错，避免半个样本看似成功。
    def _do_sync_batch_put(
        self,
        keys: List[str],
        buffer_ptrs: List[int],
        sizes: List[int],
    ) -> None:
        """Synchronous batch_put_from with error handling."""
        total_bytes = sum(sizes)
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
                    "Failed to cleanup keys after batch_put_from failure: %s",
                    keys,
                    exc_info=True,
                )
            failure_details = ", ".join(f"{k} (code={r})" for k, r in failures)
            config_details = (
                f"total_bytes={_format_bytes(total_bytes)}, "
                f"global_segment_size={_format_bytes(self.config.global_segment_size)}, "
                f"local_buffer_size={_format_bytes(self.config.local_buffer_size)}, "
                f"host_buffer_size={_format_bytes(self.config.host_buffer_size)}"
            )
            raise RuntimeError(
                f"batch_put_from failed for keys: {failure_details}. "
                f"{config_details}. Consider increasing Mooncake segment/buffer sizes "
                "or reducing batch/sequence length/prefetch depth."
            )

    # 按字节连续排放每个张量并记录起点与长度；dtype/shape 不写在缓冲头里，消费者依赖外部元数据解释。
    @staticmethod
    def _stage_tensors_into_buffer(buf, tensors: List[torch.Tensor]) -> Tuple[List[int], List[int]]:
        """Copy tensors into a buffer and return (pointers, sizes) for batch_put_from."""
        buffer_ptrs = []
        sizes = []
        # 例如 [4, 24] BF16 隐状态占 192 字节，后续 [4] int64 token 从偏移 192 开始占 32 字节。
        # 发送的是地址和长度；若消费者把 BF16 误写成 FP32，形状一样也会要求两倍字节并读取失败。
        offset = 0
        for tensor in tensors:
            nbytes = buf.copy_from_tensor(tensor, offset=offset)
            buffer_ptrs.append(buf.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes
        return buffer_ptrs, sizes

    # 从元数据组装必需键，先等所有键可见，再尝试 GPU Direct，失败则走主机路径，最终还原 Eagle3TargetOutput。
    def get(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ) -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 tensors into GPU memory.

        For RDMA/InfiniBand: Uses GPUDirect RDMA (batch_get_into directly into GPU).
        For TCP: Uses batch_get_buffer to host buffer, then copies to GPU.

        Automatically falls back to host buffer path if GPUDirect fails.

        Returns:
            Eagle3TargetOutput with the retrieved tensors.
        """
        self._ensure_initialized()

        from torchspec.models.target.eagle3_target_model import Eagle3TargetOutput

        # 根据 shapes 决定对象实际包含哪些可选字段，不能假设每种后端都保存了 logits 或最后隐状态。
        keys = [f"{key}_hs", f"{key}_ids"]
        tensor_specs = [
            (
                "hidden_states",
                shapes["hidden_states"],
                dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
            ),
            ("input_ids", shapes["input_ids"], torch.int64),
        ]

        if "target" in shapes:
            keys.append(f"{key}_tgt")
            tensor_specs.append(
                ("target", shapes["target"], dtypes.get("target", HIDDEN_STATES_STORAGE_DTYPE))
            )

        if "last_hidden_states" in shapes:
            keys.append(f"{key}_lhs")
            tensor_specs.append(
                (
                    "last_hidden_states",
                    shapes["last_hidden_states"],
                    dtypes.get("hidden_states", HIDDEN_STATES_STORAGE_DTYPE),
                )
            )

        # Publication is asynchronous on inference workers.  Wait for a
        # complete metadata census before attempting either byte-transfer
        # path; this also gives GPUDirect the retry/fail-closed behavior that
        # previously existed only in the host-buffer fallback.
        self.wait_for_keys(keys)

        # 两条字节传输路径共享同一 keys/specs；GPU Direct 失败可以回退，不改变逻辑张量语义。
        tensor_map = None
        if self._gpu_direct_available and self._gpu_receive_buffer is not None:
            tensor_map = self._get_tensors_gpu_direct(keys, tensor_specs, device)
            if tensor_map is None:
                logger.warning("GPUDirect batch_get_into failed; falling back to host buffer path.")

        if tensor_map is None:
            tensor_map = self._get_tensors_via_host_buffer(keys, tensor_specs, device)
            logger.debug("Using host buffer path (TCP)")
        logger.debug("Retrieved Eagle3 tensors with base key: %s", key)

        # 恢复模型输出容器只组装引用，不额外 clone；尤其 GPU receive 视图的生命周期继续约束消费者。
        return Eagle3TargetOutput(
            hidden_states=tensor_map["hidden_states"],
            target=tensor_map.get("target"),
            input_ids=tensor_map["input_ids"],
            last_hidden_states=tensor_map.get("last_hidden_states"),
            input_ids_cpu=tensor_map.get("input_ids_cpu"),
        )

    # 实际代码把所有字段接收到同一已注册 GPU 缓冲，再返回 dtype/shape 视图；容量不足或传输失败则返回 None。
    # 注意返回值共享接收缓冲，下一次 get 可覆盖它；应按真实实现理解，不能把上方英文描述理解为每字段独立分配。
    def _get_tensors_gpu_direct(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Transfer directly into GPU memory using batch_get_into (GPUDirect RDMA).

        Pre-allocates individual destination tensors and transfers directly into
        their memory, avoiding an extra GPU→GPU copy.  Falls back to the
        registered GPU receive buffer when a tensor is too small to be
        page-aligned (RDMA typically requires registered memory).

        Returns None if transfer fails, allowing caller to fall back to host
        buffer path.
        """
        total_size = sum(
            self._compute_tensor_size(shape, dtype) for _, shape, dtype in tensor_specs
        )

        # 总大小按所有字段合计；接收区不够时返回 None 让上层改走主机路径，而不是截断数据。
        if total_size > self._gpu_receive_buffer.size:
            logger.warning(
                "GPU buffer too small: need %.1fMB, have %.1fMB. Increase gpu_buffer_size in config.",
                total_size / (1024**2),
                self._gpu_receive_buffer.size / (1024**2),
            )
            return None

        # Compute per-tensor sizes and buffer offsets up-front.
        buffer_ptrs: List[int] = []
        sizes: List[int] = []
        offsets: List[int] = []
        offset = 0

        # 把每字段排到同一连续字节区；shape 决定元素数，dtype 决定每元素字节数。
        for _, shape, dtype in tensor_specs:
            size = self._compute_tensor_size(shape, dtype)
            buffer_ptrs.append(self._gpu_receive_buffer.ptr + offset)
            sizes.append(size)
            offsets.append(offset)
            offset += size

        try:
            # 传输完成后检查返回码；负数触发回退，某些客户端以零或实际字节数表示成功。
            results = self._store.batch_get_into(keys, buffer_ptrs, sizes)
            for i, (k, r) in enumerate(zip(keys, results)):
                if r < 0:
                    logger.warning("batch_get_into failed for %s with error code: %s", k, r)
                    return None
                if r != 0 and r != sizes[i]:
                    logger.warning(
                        "batch_get_into for %s: unexpected return %s (expected 0 or %s)",
                        k,
                        r,
                        sizes[i],
                    )
        except Exception as e:
            logger.warning("batch_get_into exception: %s", e)
            return None

        tensor_map = {}
        for i, (name, shape, dtype) in enumerate(tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim
            # 接收区一直留在 setup 选择的 GPU；此方法没有按 device 参数迁移返回值。
            # 调用者应让 store 的设备与消费设备一致；设备字符串不是网络地址或自动搬运指令。
            buf_slice = self._gpu_receive_buffer.get_slice(offsets[i], sizes[i])
            # View into the registered buffer; valid until the next call.
            # 这里没有 clone：所有返回字段都借用接收区。下一个请求到来会写同一分配，
            # 训练侧若需要跨多次 get 持有旧样本，必须在覆盖前把数据转移到独立存储。
            tensor_map[name] = buf_slice.view(dtype)[:numel].reshape(shape)

        # 若要跨下一次 GET 保留这一层，不仅要 clone，还要等 clone 所在 CUDA stream 完成：
        # 下一次网卡写接收区不受 CUDA 排队约束。训练侧 _clone_fragment_before_receive_buffer_reuse
        # 正是为逐层读取做这个同步；仅 record_stream 不能阻止外部网卡覆盖仍在读取的内存。
        logger.debug("GPU Direct RDMA transfer successful for %s tensors", len(keys))
        return tensor_map

    # 将所有维度相乘再乘 dtype 字节数；它计算存储长度，不包含 tensor 对象、分配器或对齐开销。
    @staticmethod
    def _compute_tensor_size(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute the byte size of a tensor with given shape and dtype."""
        numel = 1
        for dim in shape:
            numel *= dim
        return numel * _DTYPE_ELEMENT_SIZES[dtype]

    # 批量取主机缓冲，缺键则按配置重试；逐字段严格核对字节数，再用 frombuffer 解释并搬到目标设备。
    # input_ids 额外 clone 一份 CPU 副本供训练侧读取，避免为了 token 再从 GPU 同步回来。
    def _get_tensors_via_host_buffer(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Transfer via Mooncake's registered host buffer, then copy to device."""
        wait_seconds = max(self.config.get_retry_wait_seconds, 0.05)
        log_interval = max(self.config.get_retry_log_interval_seconds, wait_seconds)
        max_wait = max(self.config.get_retry_max_wait_seconds, 0.0)
        start_time = time.time()
        last_log = 0.0

        # 即便元数据已经可见，字节获取仍可能暂时缺项，因此此处保留独立的批量 get 重试。
        while True:
            # 每轮重取整组，先前返回的部分 buffers 不拼接进新结果；成功路径才统一解释本轮字段。
            # 例如 hs 已到而 ids 缺失时，本轮不会提前返回 hs 给训练循环。
            buffers = self._store.batch_get_buffer(keys)
            missing = [i for i, buf in enumerate(buffers) if buf is None]
            if not missing:
                break

            elapsed = time.time() - start_time
            # 配置正数上限时超时明确列出缺失对象；非正值不设期限，长期等待需回查生产端发布是否失败。
            if max_wait > 0 and elapsed >= max_wait:
                missing_keys = ", ".join(keys[i] for i in missing)
                raise RuntimeError(
                    f"batch_get_buffer returned None for keys: {missing_keys}. "
                    f"Waited {elapsed:.1f}s; aborting."
                )

            now = time.time()
            if last_log == 0.0 or (now - last_log) >= log_interval:
                missing_keys = ", ".join(keys[i] for i in missing)
                logger.warning(
                    "batch_get_buffer missing keys (%s); sleeping %.2fs.",
                    missing_keys,
                    wait_seconds,
                )
                last_log = now
            time.sleep(wait_seconds)

        tensor_map = {}
        for i, ((name, shape, dtype), buf) in enumerate(zip(tensor_specs, buffers)):
            if buf is None:
                raise RuntimeError(
                    f"batch_get_buffer returned None for key '{keys[i]}' (tensor: {name}). "
                    "This may indicate the key doesn't exist or RDMA transfer failed."
                )

            numel = 1
            for dim in shape:
                numel *= dim
            element_size = _DTYPE_ELEMENT_SIZES[dtype]
            expected_size = numel * element_size

            buf_size = buf.size()
            # 长度不一致常见于捕获 token 数或精度协议错配；拒绝 reshape 错误字节比生成静默错误样本更可靠。
            if buf_size != expected_size:
                actual_elements = buf_size // element_size if element_size > 0 else 0
                logger.error(
                    f"Size mismatch for {name} (key={keys[i]}): "
                    f"got {buf_size} bytes ({actual_elements} elements), "
                    f"expected {expected_size} bytes ({numel} elements). "
                    f"Expected shape: {shape}, dtype: {dtype}, element_size: {element_size}"
                )
                raise RuntimeError(
                    f"Size mismatch for {name}: got {buf_size} bytes, expected {expected_size} bytes"
                )

            # ctypes 和 frombuffer 都不拥有原始网络缓冲；它们只是让 PyTorch 按 dtype 看见同一段内存。
            c_array = (ctypes.c_byte * buf_size).from_address(buf.ptr())
            host_tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(shape)

            # frombuffer 只借用原生缓冲；目标为 CUDA 时这里的传输产生设备端存储。
            # 不要把 to(cpu) 想成必然复制：同设备转换可保留共享关系，因此 CPU 调用也要关注原生缓冲生命周期。
            tensor_map[name] = host_tensor.to(device)

            if name == "input_ids":
                tensor_map["input_ids_cpu"] = host_tensor.clone()

        return tensor_map

    # 消费后按实际存在的可选字段强制删除，最多尝试三次，仅重试失败键；-704 已不存在也算成功。
    # 删除是尽力回收，最终失败只记录日志，避免清理失败中断训练读取流程。
    def remove_eagle3_tensors(
        self,
        key: str,
        has_last_hidden_states: bool = False,
        has_target: bool = False,
    ) -> None:
        """Force-delete all tensors associated with an Eagle3 output.

        Uses ``batch_remove(force=True)`` to bypass lease TTL and delete
        immediately after consumption. Retries up to 3 times on failure.
        Never raises — deletion is best-effort to avoid breaking the
        training fetch path.

        Args:
            key: Base key used when storing
            has_last_hidden_states: Whether last_hidden_states was stored
            has_target: Whether target (logits) was stored
        """
        keys = [f"{key}_hs", f"{key}_ids"]
        if has_target:
            keys.append(f"{key}_tgt")
        if has_last_hidden_states:
            keys.append(f"{key}_lhs")

        for attempt in range(1, 4):
            try:
                results = self._store.batch_remove(keys, force=True)
            except Exception:
                if attempt < 3:
                    logger.warning(
                        "batch_remove raised for %s (attempt %d/3)", key, attempt, exc_info=True
                    )
                    time.sleep(0.5)
                else:
                    logger.error(
                        "Force delete abandoned for %s after 3 exceptions", key, exc_info=True
                    )
                continue
            failed = [(k, r) for k, r in zip(keys, results) if r not in (None, 0, -704)]
            if not failed:
                logger.debug("Force-deleted %s (%d keys)", key, len(results))
                return
            if attempt < 3:
                time.sleep(0.5)
                logger.warning(
                    "Retrying force delete for %s: %d keys failed (attempt %d/3)",
                    key,
                    len(failed),
                    attempt,
                )
            else:
                logger.error("Force delete abandoned for %s: %s", key, failed)
                return
            # 首轮 [hs=成功, ids=失败] 后只保留 ids；后续日志的 keys 已不是原始完整样本键集合。
            keys = [k for k, _ in failed]
