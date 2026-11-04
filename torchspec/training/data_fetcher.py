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

# 训练输入适配：Ray 队列只传 TrainSample 元数据，真正的大张量通过 Mooncake key 读取。
# DataLoader 负责 padding/组 batch，后台预取仅隐藏等待；收到 None 是流结束而不是空样本。

"""Queue-based data fetching with mooncake store.
Data flow:
  TrainActor -> MooncakeDataFetcher -> MooncakeDataset -> MooncakeStore -> Collator
                     |                      |                  |               |
                iter(fetcher)          queue.get()      store.get(key)     pad & batch
"""

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ray.util.queue import Queue as RayQueue
from torch.utils.data import DataLoader, IterableDataset

from torchspec.data.utils import deserialize_packed_loss_mask, resolve_loss_mask, unpack_loss_mask
from torchspec.utils.distributed import (
    get_draft_sp_group,
    get_sp_ring_group,
    get_usp_rank_coords,
)
from torchspec.utils.logging import logger


# 复制接收 buffer 的张量视图，并同步 clone 所在 CUDA stream 后才允许下一次 GPUDirect GET。
# 仅 clone 不够：clone 是异步 CUDA 操作，而远端写 buffer 不受 CUDA stream 顺序保护。
def _clone_fragment_before_receive_buffer_reuse(
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Clone Mooncake receive-buffer views before the next GPUDirect GET.

    CUDA clones are asynchronous, while Mooncake writes the registered receive
    buffer outside CUDA stream ordering. Synchronize the clone stream before a
    subsequent GET can reuse and overwrite that buffer.
    """
    cloned = {name: tensor.clone() for name, tensor in tensors.items()}
    cuda_tensor = next((tensor for tensor in cloned.values() if tensor.is_cuda), None)
    if cuda_tensor is not None:
        torch.cuda.current_stream(cuda_tensor.device).synchronize()
    return cloned


@dataclass
class TrainSample:
    mooncake_key: str
    tensor_shapes: Dict[str, Tuple[int, ...]]
    tensor_dtypes: Optional[Dict[str, torch.dtype]] = None
    packed_loss_mask: Optional[str] = None
    last_turn_loss_only: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    data_id: Optional[str] = None


class MooncakeDataset(IterableDataset):
    """IterableDataset that loads from mooncake via queue.

    Each DP rank waits on its queue for TrainSample items sent by the
    centralized controller. Data is loaded from mooncake.
    """

    # 保存队列、存储客户端、mask 策略和本地序列上限，再初始化 SP 坐标。
    # batch_size 影响读取结果是否必须 clone：collator 同时保留多个样本时不能共享可复用接收 buffer。
    def __init__(
        self,
        ray_queue: RayQueue,
        mooncake_store,
        device: torch.device,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        batch_size: int = 1,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
    ):
        self.ray_queue = ray_queue
        self.mooncake_store = mooncake_store
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.assistant_header_ids = assistant_header_ids
        self.end_token_ids = end_token_ids
        self.dynamic_loss_mask = dynamic_loss_mask
        self.last_turn_loss_only = last_turn_loss_only
        self.skip_after_header = skip_after_header
        self._batch_size = batch_size
        self._min_loss_tokens = min_loss_tokens
        self.usp_enabled = usp_enabled
        self.ttt_length = ttt_length
        self.max_seq_length = max_seq_length
        self._init_sp_context()

    # 先建立无 SP 的单 rank 默认上下文；USP 开启且组存在时记录本 rank 在 SP/Ring 组内的位置。
    def _init_sp_context(self) -> None:
        self._sp_group = None
        self._sp_world_size = 1
        self._sp_rank = 0
        self._sp_ring_size = 1
        self._sp_ring_rank = 0
        if not self.usp_enabled:
            return

        sp_group = get_draft_sp_group()
        if sp_group is None:
            return

        self._sp_group = sp_group
        self._sp_world_size = dist.get_world_size(sp_group)
        self._sp_rank = dist.get_rank(sp_group)

        ring_group = get_sp_ring_group()
        if ring_group is not None:
            self._sp_ring_size = dist.get_world_size(ring_group)
            self._sp_ring_rank = dist.get_rank(ring_group)

    # 把序列化 dtype 名转换为 torch.dtype，按 metadata 决定整样本读取还是 PP 分层读取。
    # 多样本 batch 复制接收视图后释放 store key；单样本保留 pinned 视图以便非阻塞 H2D。
    # packed mask 与 last_turn 标志属于控制元数据，需要并入结果给后续 collator 解析。
    def _load_from_mooncake(self, sample: TrainSample) -> Dict[str, Any]:
        """Load tensors from mooncake key into device memory."""
        dtypes_raw = sample.tensor_dtypes or {}

        # Convert string dtypes to torch.dtype objects
        dtypes = {}
        for key, dtype_val in dtypes_raw.items():
            if isinstance(dtype_val, str):
                # Handle "bfloat16" or "torch.bfloat16" format
                dtype_str = dtype_val.replace("torch.", "")
                dtypes[key] = getattr(torch, dtype_str)
            else:
                dtypes[key] = dtype_val

        # DEBUG: Print the shapes we're requesting
        logger.debug(
            f"_load_from_mooncake: key={sample.mooncake_key}, "
            f"requesting shapes={sample.tensor_shapes}"
        )

        metadata = sample.metadata or {}
        # 同一个 TrainSample 协议可描述完整样本，也可通过 manifest 描述 PP 分阶段上传的层片段。
        # 不能只凭 tensor_shapes 决定 GET 一个 key：PP 的实际内容分布在 manifest 列出的多个 key 中。
        pp_manifest = metadata.get("vllm_pp_layer_manifest")
        if pp_manifest is not None:
            result = self._load_vllm_pp_layers(sample, pp_manifest, dtypes)
        else:
            tensors = self.mooncake_store.get(
                key=sample.mooncake_key,
                shapes=sample.tensor_shapes,
                dtypes=dtypes,
                device=self.device,
            )

            tensor_dict = tensors.to_tensor_dict()
            # 例如 batch_size=2，collator 先收下第一个样本再获取第二个。
            # 若两个样本引用同一接收 buffer，第二次 GET 会让第一个样本也变成第二份数据。
            # clone 隔离底层存储，但 CPU clone 不再保证 pinned，可能牺牲 H2D 重叠能力。
            if self._batch_size > 1:
                # Clone to prevent use-after-free: collator holds sample N while
                # fetching N+1, but cleanup frees the Mooncake buffer (Issue 31).
                # Note: clone() converts pinned → unpinned, breaking non_blocking
                # H2D transfers. Only do this when actually needed.
                result = {k: v.clone() for k, v in tensor_dict.items()}
            else:
                # batch_size=1: safe to use pinned views — consumed immediately.
                # Preserves pinned memory for async H2D via non_blocking=True.
                result = dict(tensor_dict)

            # 这里只在拿到可供消费的本地结果后删除 store 数据。
            # store key 的生命周期和本地 tensor 的生命周期不同：删除远端条目不等于训练已经不再使用本地值。
            self._cleanup_mooncake_data(sample)
        if sample.packed_loss_mask is not None:
            result["packed_loss_mask"] = sample.packed_loss_mask
        if sample.last_turn_loss_only is not None:
            result["last_turn_loss_only"] = sample.last_turn_loss_only
        return result

    # 要求 PP manifest 完整，逐层读 [S,H] 并 clone+同步，验证各片 token 一致。
    # 辅助层按最后维拼成 [S,L*H]，末层特征单独保留；成功收集后删除每个片段 key。
    def _load_vllm_pp_layers(
        self,
        sample: TrainSample,
        manifest: list[dict[str, Any]],
        dtypes: dict[str, torch.dtype],
    ) -> Dict[str, Any]:
        metadata = sample.metadata or {}
        if not metadata.get("vllm_pp_complete", False):
            raise RuntimeError(f"Incomplete vLLM PP sample: mooncake_key={sample.mooncake_key}")

        seq_len = sample.tensor_shapes["input_ids"][0]
        hidden_size = sample.tensor_shapes["last_hidden_states"][-1]
        fragment_shapes = {
            "hidden_states": (seq_len, hidden_size),
            "input_ids": (seq_len,),
        }
        fragment_dtypes = {
            "hidden_states": dtypes.get("hidden_states", torch.bfloat16),
            "input_ids": torch.int64,
        }

        hidden_states = []
        last_hidden_states = None
        input_ids = None
        fragment_keys = []
        # manifest 顺序决定辅助层拼接顺序；只检验 token 相同还不足以允许任意重排层。
        # 每个片段临时请求 [S,H]，组装结果才变成 [S,L*H]，不在传输层额外引入 batch 维。
        for entry in manifest:
            key = entry["mooncake_key"]
            tensors = self.mooncake_store.get(
                key=key,
                shapes=fragment_shapes,
                dtypes=fragment_dtypes,
                device=self.device,
            ).to_tensor_dict()

            # 下一层 GET 可能覆盖同一注册接收区；必须在此把本层数据复制到独立存储并等复制完成。
            cloned = _clone_fragment_before_receive_buffer_reuse(tensors)
            layer_hidden = cloned["hidden_states"]
            layer_input_ids = cloned["input_ids"]
            if input_ids is None:
                input_ids = layer_input_ids
            elif not torch.equal(input_ids, layer_input_ids):
                raise RuntimeError(f"vLLM PP input_ids mismatch for fragment {key}")

            # 末层 teacher 特征用于构造监督，不应混进多辅助层输入中。
            # role 区分这两类张量；若 manifest 缺末层或所有辅助层，下面会直接拒绝样本。
            if entry["role"] == "last_hidden_states":
                last_hidden_states = layer_hidden
            else:
                hidden_states.append(layer_hidden)

            fragment_keys.append(key)

        if not hidden_states or last_hidden_states is None or input_ids is None:
            raise RuntimeError(f"Invalid vLLM PP layer manifest for {sample.mooncake_key}")

        # 只有全部片段校验和组装前置条件成功才进入清理阶段。
        # 异常路径不会在此循环删除所有 fragment；读取失败的资源回收需结合外部生命周期处理，
        # 不能从这段实现推断出任意中途失败都自动清理。
        for key in fragment_keys:
            self.mooncake_store.remove_eagle3_tensors(
                key,
                has_last_hidden_states=False,
                has_target=False,
            )

        return {
            "hidden_states": torch.cat(hidden_states, dim=-1),
            "last_hidden_states": last_hidden_states,
            "input_ids": input_ids,
        }

    # 根据 shapes 中是否存在末层特征/target 删除相应 store 条目，释放生产端存储空间。
    def _cleanup_mooncake_data(self, sample: TrainSample) -> None:
        """Remove data from mooncake store to release buffer space."""
        shapes = sample.tensor_shapes or {}
        has_lhs = "last_hidden_states" in shapes
        has_target = "target" in shapes

        self.mooncake_store.remove_eagle3_tensors(
            sample.mooncake_key,
            has_last_hidden_states=has_lhs,
            has_target=has_target,
        )

    # 把动态匹配、显式 mask 与 last-turn 策略交给 resolve_loss_mask，返回解析后的监督区域。
    def _compute_loss_mask(self, data: Dict[str, Any]) -> torch.Tensor | None:
        return resolve_loss_mask(
            data,
            dynamic_loss_mask=self.dynamic_loss_mask,
            assistant_header_ids=self.assistant_header_ids,
            end_token_ids=self.end_token_ids,
            last_turn_loss_only=self.last_turn_loss_only,
            skip_after_header=self.skip_after_header,
        )

    # 空监督样本优先用 input_ids 最后一维推断长度，否则用特征倒数第二维，兜底长度为 1。
    @staticmethod
    def _fallback_mask_len(data: Dict[str, Any]) -> int:
        """Sequence length for a neutralized sample's zero loss mask."""
        ids = data.get("input_ids")
        if isinstance(ids, torch.Tensor) and ids.dim() >= 1:
            return ids.shape[-1]
        for key in ("hidden_states", "last_hidden_states", "target"):
            t = data.get(key)
            if isinstance(t, torch.Tensor) and t.dim() >= 2:
                return t.shape[-2]
        return 1

    # 无有效 mask 或监督 token 少于阈值时原地换成全零 mask，并返回更新后的计数。
    # 不能直接丢弃：一个 DP rank 少一次 forward/backward 会让其余 rank 卡在 FSDP collective。
    def _resolve_and_neutralize_loss_mask(
        self, data: Dict[str, Any], mooncake_key: str, neutralized_count: int
    ) -> int:
        """Zero an empty / sub-min_loss_tokens loss mask in place instead of
        dropping the sample; a per-rank drop desyncs FSDP collectives."""
        mask = self._compute_loss_mask(data)  # None == all-zero
        # 保留全零样本的形状和迭代次数，比过滤样本更重要：collective 要求各 rank 同步参与。
        neutralize = mask is None or (
            self._min_loss_tokens > 0
            and isinstance(mask, torch.Tensor)
            and mask.sum() < self._min_loss_tokens
        )
        if not neutralize:
            return neutralized_count

        if isinstance(mask, torch.Tensor):
            n_tokens = int(mask.sum())
            data["loss_mask"] = torch.zeros_like(mask)
        else:
            # resolve_loss_mask returns None without setting data["loss_mask"]
            n_tokens = 0
            data["loss_mask"] = torch.zeros(self._fallback_mask_len(data), dtype=torch.long)

        neutralized_count += 1
        logger.warning(
            f"Neutralized loss mask ({n_tokens} < min_loss_tokens="
            f"{self._min_loss_tokens}, mooncake_key={mooncake_key}, "
            f"total_neutralized={neutralized_count})"
        )
        return neutralized_count

    # 阻塞读取 TrainSample，超时/异常或 None 结束；普通路径补 batch 维后逐样本 yield。
    # [ S ] token/mask 变 [1,S]，特征 [S,H] 变 [1,S,H]；USP 则交给预分片读取路径。
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over samples synchronously.

        Blocks waiting for each item from the queue and loads from mooncake.
        Empty / sub-min_loss_tokens masks are neutralized, not dropped
        (see _resolve_and_neutralize_loss_mask).
        """
        yield_count = 0
        neutralized_count = 0
        while True:
            if self.usp_enabled:
                data, neutralized = self._usp_get_sharded_item(neutralized_count=neutralized_count)
                neutralized_count += neutralized
                if data is None:
                    break
                yield_count += 1
                yield data
                continue

            logger.debug(f"__iter__: waiting for item from ray_queue (yield_count={yield_count})")
            try:
                # 队列空时阻塞是正常流控：teacher 和训练速度不必相同。
                # 此实现把超时/异常也当作迭代终止，调用方必须结合日志区分数据流正常结束与读取失败。
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"__iter__: Exception waiting for data: {e}, timeout={self.timeout}")
                break

            if item is None:
                logger.debug("__iter__: received None sentinel, stopping iteration")
                break

            logger.debug(f"__iter__: got item, mooncake_key={item.mooncake_key}")
            data = self._load_from_mooncake(item)

            neutralized_count = self._resolve_and_neutralize_loss_mask(
                data, item.mooncake_key, neutralized_count
            )

            # Note: target is computed in the collator from last_hidden_states for sglang mode

            # Add batch dimension if missing (sglang stores without batch dim)
            # 这里只补缺失的 batch 维，不做跨样本 padding，也不把所有二维 tensor 都当 hidden state。
            # input_ids 的二维 [B,S] 本来合法，只有指定特征字段的 [S,H] 才应升到 [1,S,H]。
            for key, tensor in data.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    # Check if tensor is missing batch dimension
                    # 1D tensors (loss_mask, input_ids) should be 2D: (1, seq_len)
                    # 2D tensors (hidden_states, last_hidden_states) should be 3D: (1, seq_len, dim)
                    if tensor.dim() == 1:
                        data[key] = tensor.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
                    elif tensor.dim() == 2 and key in [
                        "hidden_states",
                        "last_hidden_states",
                        "target",
                    ]:
                        data[key] = tensor.unsqueeze(0)  # (seq_len, dim) -> (1, seq_len, dim)

            # Debug: log all tensor shapes after adding batch dim
            if data:
                shapes_str = {
                    k: v.shape if hasattr(v, "shape") else type(v) for k, v in data.items()
                }
                logger.debug(f"final shapes (with batch dim): {shapes_str}")
            yield_count += 1
            logger.debug(f"__iter__: yielding batch {yield_count}, keys={list(data.keys())}")
            yield data

    # 从全局 input_ids shape 取原始长度，并应用 max_seq_length 上限，保持分片几何一致。
    def _usp_global_len(self, sample: TrainSample) -> int:
        global_len = sample.tensor_shapes["input_ids"][-1]
        if self.max_seq_length is not None:
            global_len = min(global_len, self.max_seq_length)
        return global_len

    # 向上取整 global_len/SP，所有 rank 使用相同主块长度，最后一片不足部分由 padding 补齐。
    def _usp_chunk_size(self, global_len: int) -> int:
        return (global_len + self._sp_world_size - 1) // self._sp_world_size

    # USP 必须携带 packed mask；解包后按 global_len 截断/补零，得到所有 SP rank 一致的全局监督信息。
    def _usp_loss_mask(self, sample: TrainSample, global_len: int) -> torch.Tensor:
        if sample.packed_loss_mask is None:
            raise RuntimeError("USP sharded Mooncake reads require packed_loss_mask metadata")
        loss_mask = unpack_loss_mask(deserialize_packed_loss_mask(sample.packed_loss_mask))
        loss_mask = loss_mask[:global_len]
        if loss_mask.shape[0] < global_len:
            loss_mask = F.pad(loss_mask, (0, global_len - loss_mask.shape[0]))
        return loss_mask

    # 本地序列长度为 ceil(S/SP)+ttt_length；额外尾部供多步目标跨片边界对齐。
    # 返回 batch=1 的 token 和特征 shape，按元数据决定是否请求 last_hidden_states/target。
    def _local_usp_shapes(self, sample: TrainSample) -> dict[str, tuple[int, ...]]:
        local_len = self._usp_chunk_size(self._usp_global_len(sample)) + self.ttt_length
        shapes: dict[str, tuple[int, ...]] = {
            "input_ids": (1, local_len),
            "hidden_states": (1, local_len, sample.tensor_shapes["hidden_states"][-1]),
        }
        if "last_hidden_states" in sample.tensor_shapes:
            shapes["last_hidden_states"] = (
                1,
                local_len,
                sample.tensor_shapes["last_hidden_states"][-1],
            )
        if "target" in sample.tensor_shapes:
            shapes["target"] = (1, local_len, sample.tensor_shapes["target"][-1])
        return shapes

    # 按 SP rank 截出本地 mask 并补零，attention_mask 仅标记实际读取的有效 token。
    # RoPE position_ids 覆盖当前 Ring 块中 Ulysses 聚合后的主序列，不含额外 ttt 尾部，长度未必等于本地输入长度。
    def _local_usp_loss_and_position(
        self,
        sample: TrainSample,
        local_len: int,
    ) -> dict[str, torch.Tensor]:
        sp_ulysses_size = max(1, self._sp_world_size // self._sp_ring_size)
        global_len = self._usp_global_len(sample)
        chunk_size = self._usp_chunk_size(global_len)
        start = self._sp_rank * chunk_size
        end = min(start + local_len, global_len)
        # 最后一个 rank 的有效片段可能少于 local_len，甚至起点已超过 global_len。
        # valid_len 截到零可让 attention_mask 全零，shape 仍保持与其他 rank 相同。
        valid_len = max(0, end - start)

        loss_mask = self._usp_loss_mask(sample, global_len)[start:end].unsqueeze(0)
        if loss_mask.shape[-1] < local_len:
            loss_mask = F.pad(loss_mask, (0, local_len - loss_mask.shape[-1]))

        attention_mask = torch.zeros((1, local_len), dtype=torch.long)
        attention_mask[:, :valid_len] = 1

        usp_chunk_size = max(local_len - self.ttt_length, 0)
        # 本地读取含 TTT 尾部，但 attention 的主序列位置先去掉它，再拼上同 Ring 块的 Ulysses 主块。
        # 例如主块 5、U=2，Ring 每块覆盖 10 个位置；Ring rank 1 的位置是 10..19。
        # 这不是把局部 start 直接作为每个 attention token 的位置起点。
        ring_chunk = usp_chunk_size * sp_ulysses_size
        _, ring_rank = get_usp_rank_coords(
            sp_rank=self._sp_rank,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=self._sp_ring_size,
        )
        ring_start = ring_rank * ring_chunk
        position_ids = torch.arange(
            ring_start,
            ring_start + ring_chunk,
            dtype=torch.long,
        ).unsqueeze(0)

        return {
            "loss_mask": loss_mask.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "position_ids": position_ids.to(self.device),
        }

    # 用全局监督 mask 判定不足阈值，保证同一 SP 样本各 rank 决定一致；调用者实际置零而非丢弃。
    def _should_skip_usp_sharded_sample(self, sample: TrainSample) -> bool:
        """Return the SP-consistent skip decision for a pre-sharded USP sample."""
        full_loss_mask = self._usp_loss_mask(sample, self._usp_global_len(sample))
        min_tokens = max(1, self._min_loss_tokens)
        return int(full_loss_mask.sum().item()) < min_tokens

    # 验证 usp_sharded 标记，读取本 rank 的 _usp{rank} key，附加本地 mask/position 后删除该分片。
    # 无监督样本仍返回同形状张量但 loss_mask 全零，维持 DP/FSDP 步数一致。
    def _usp_get_sharded_item(
        self, neutralized_count: int
    ) -> tuple[Dict[str, torch.Tensor] | None, int]:
        neutralized = 0
        while True:
            try:
                item = self.ray_queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                logger.warning(
                    f"_usp_get_sharded_item: Exception waiting for data: {e}, "
                    f"timeout={self.timeout}"
                )
                return None, neutralized
            if item is None:
                return None, neutralized

            # USP 路径要求生产端已按一致几何上传 _usp{rank} 分片。
            # 收到普通未分片样本时直接报错，不能在这个消费者里凭猜测静默切成另一种布局。
            metadata = item.metadata or {}
            if not metadata.get("usp_sharded", False):
                raise RuntimeError(
                    "USP sharded data fetcher received a non-sharded Mooncake sample. "
                    f"mooncake_key={item.mooncake_key}"
                )

            shapes = self._local_usp_shapes(item)
            dtypes_raw = item.tensor_dtypes or {}
            dtypes = {}
            for key, dtype_val in dtypes_raw.items():
                if isinstance(dtype_val, str):
                    dtypes[key] = getattr(torch, dtype_val.replace("torch.", ""))
                else:
                    dtypes[key] = dtype_val

            # 名称是 should_skip，但这里是“监督不合格”的判定，后面采取置零而不跳过。
            # 判断使用全局 packed mask，因此同一 SP 组对该样本一致；数据仍实际读取以保持消费/清理节奏。
            should_skip = self._should_skip_usp_sharded_sample(item)
            shard_key = f"{item.mooncake_key}_usp{self._sp_rank}"
            tensors = self.mooncake_store.get(
                key=shard_key,
                shapes=shapes,
                dtypes=dtypes,
                device=self.device,
            ).to_tensor_dict()
            tensors.update(self._local_usp_loss_and_position(item, shapes["input_ids"][-1]))

            self.mooncake_store.remove_eagle3_tensors(
                shard_key,
                has_last_hidden_states="last_hidden_states" in shapes,
                has_target="target" in shapes,
            )

            # 这条样本即使不贡献 loss，也要 yield 给训练循环。
            # 否则某个 DP 副本比别的副本少一次 backward，不能仅靠 SP 内判定一致来避免 FSDP 失步。
            if should_skip:
                # Neutralize, don't drop: a per-DP-group drop desyncs FSDP.
                neutralized += 1
                tensors["loss_mask"] = torch.zeros_like(tensors["loss_mask"])
                logger.warning(
                    f"Neutralized USP sharded sample (loss mask zeroed): "
                    f"mooncake_key={item.mooncake_key}, sp_rank={self._sp_rank}, "
                    f"total_neutralized={neutralized_count + neutralized}"
                )

            return tensors, neutralized


# 把队列包装为 IterableDataset，再用 collator 组 batch；num_workers=0 避免复制存储/通信上下文。
# prefetch_factor 在该数据路径是兼容字段，真正后台预取由 PrefetchedDataFetcher 负责。
def create_mooncake_dataloader(
    ray_queue: RayQueue,
    mooncake_store,
    collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int = 1,
    prefetch_factor: int = 2,
    timeout: Optional[float] = None,
    assistant_header_ids: Optional[List[int]] = None,
    end_token_ids: Optional[List[int]] = None,
    dynamic_loss_mask: bool = False,
    last_turn_loss_only: bool = False,
    skip_after_header: int = 0,
    min_loss_tokens: int = 0,
    usp_enabled: bool = False,
    ttt_length: int = 1,
    max_seq_length: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader that fetches from mooncake via queue.

    Data flow:
      Controller (dispatches dispatch_batch_size samples) ->
      Ray Queue (per_dp_rank_batch_size samples per rank) ->
      DataLoader (batches per_dp_rank_batch_size samples together with padding) ->
      Training loop (one iteration per step)

    The collator pads sequences within the batch to the same length.

    Args:
        ray_queue: Ray Queue to receive TrainSample from controller.
        mooncake_store: Mooncake store client for loading tensors.
        collator: Collator for padding and batching samples.
        device: Target device for tensors.
        batch_size: Number of samples per batch (= per_dp_rank_batch_size).
        prefetch_factor: Unused, kept for API compatibility.
        timeout: Timeout in seconds for waiting on queue. None means wait forever.
        assistant_header_ids: Token IDs for assistant header (for loss mask skip check).
        end_token_ids: Token IDs for end of turn (for loss mask skip check).
        dynamic_loss_mask: Whether loss mask is computed dynamically from input_ids.
        last_turn_loss_only: Global fallback for last-turn-only loss masking.

    Returns:
        DataLoader instance.
    """
    # 构建 Dataset 时保留所有 mask/USP 参数，后续 __iter__ 才真正执行阻塞 get。
    # 因此创建 DataLoader 本身不代表已拿到任何 teacher 数据，也不启动独立生产过程。
    dataset = MooncakeDataset(
        ray_queue,
        mooncake_store,
        device,
        prefetch_factor,
        timeout,
        assistant_header_ids=assistant_header_ids,
        end_token_ids=end_token_ids,
        dynamic_loss_mask=dynamic_loss_mask,
        last_turn_loss_only=last_turn_loss_only,
        skip_after_header=skip_after_header,
        batch_size=batch_size,
        min_loss_tokens=min_loss_tokens,
        usp_enabled=usp_enabled,
        ttt_length=ttt_length,
        max_seq_length=max_seq_length,
    )

    # IterableDataset 没有随机索引；每次 next 都消费队列并推进远端数据生命周期。
    # num_workers=0 让 store/进程组在当前进程使用，避免 fork/spawn 后复制 CUDA 和通信资源。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
    )


class MooncakeDataFetcher:
    """Queue-based data fetcher for mooncake with DataLoader backend.

    Provides iteration over training samples that are pushed to a Ray queue
    by the AsyncTrainingController and loaded from mooncake.

    Batch size design:
      - micro_batch_size: Samples per GPU per training step (user config)
      - per_dp_rank_batch_size = micro_batch_size * sp_size (derived)
      - dispatch_batch_size = per_dp_rank_batch_size * dp_size (derived)
      - DataLoader batch_size = per_dp_rank_batch_size (all samples batched together)
      - Training loop does ONE iteration per step

    The collator pads sequences within the batch to the max length.
    """

    # 构造并持有 DataLoader，完整转交 mask、USP、序列上限配置；它不自行创建生产任务。
    def __init__(
        self,
        queue: RayQueue,
        mooncake_store,
        collator: Callable[[List[Dict]], Dict[str, torch.Tensor]],
        device: torch.device,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        timeout: Optional[float] = None,
        assistant_header_ids: Optional[List[int]] = None,
        end_token_ids: Optional[List[int]] = None,
        dynamic_loss_mask: bool = False,
        last_turn_loss_only: bool = False,
        skip_after_header: int = 0,
        min_loss_tokens: int = 0,
        usp_enabled: bool = False,
        ttt_length: int = 1,
        max_seq_length: Optional[int] = None,
    ):
        self.batch_size = batch_size
        # Fetcher 是稳定的迭代入口，Dataset 负责单样本读取，DataLoader/collator 负责 batch 组织。
        # 这一层不重新解释 batch_size 或 ttt_length，直接转交以保持控制器与读取器的配置一致。
        self._dataloader = create_mooncake_dataloader(
            ray_queue=queue,
            mooncake_store=mooncake_store,
            collator=collator,
            device=device,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            timeout=timeout,
            assistant_header_ids=assistant_header_ids,
            end_token_ids=end_token_ids,
            dynamic_loss_mask=dynamic_loss_mask,
            last_turn_loss_only=last_turn_loss_only,
            skip_after_header=skip_after_header,
            min_loss_tokens=min_loss_tokens,
            usp_enabled=usp_enabled,
            ttt_length=ttt_length,
            max_seq_length=max_seq_length,
        )

    # 返回底层 DataLoader 迭代器，由它调用 dataset 并按 batch_size 执行 collator。
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self._dataloader)


class PrefetchedDataFetcher:
    """Wraps MooncakeDataFetcher with async pre-fetching.

    A background thread continuously fetches batches from the underlying
    MooncakeDataFetcher (which blocks on Mooncake TCP), staging them in a
    thread-safe queue.  The training loop reads from this queue, overlapping
    data transfer with GPU compute.

    Without prefetch: [data] → [compute] → [data] → [compute]  (sequential)
    With prefetch:    [compute] → [compute] → [compute]         (overlapped)
                      [data]      [data]      [data]

    The background thread starts lazily on the first ``__iter__`` call and
    keeps running across multiple ``itertools.islice`` invocations (one per
    training step).  The training loop simply reads from the shared queue.
    """

    _SENTINEL = object()

    # 建立有界线程队列和错误/线程状态；容量限制预取驻留内存，并对生产线程形成背压。
    def __init__(
        self,
        inner: MooncakeDataFetcher,
        prefetch_depth: int = 2,
        target_device: Optional[torch.device] = None,
    ):
        self.inner = inner
        self.prefetch_depth = prefetch_depth
        self.target_device = target_device
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._error: Optional[BaseException] = None

    # 后台连续读取 inner 并 put；保存异常原始 traceback，finally 投递 sentinel 让消费者退出。
    def _prefetch_loop(self) -> None:
        try:
            for batch in self.inner:
                self._queue.put(batch)
        except Exception as e:
            # Preserve the original traceback so re-raise in __next__
            # points to the actual failure site, not to __next__ itself.
            import sys

            self._error = e.with_traceback(sys.exc_info()[2])
        finally:
            self._queue.put(self._SENTINEL)

    # 只在首次消费时启动 daemon 线程；后续多个训练步共享同一线程和迭代流。
    def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._thread.start()

    # 确保后台线程启动后返回自身，所以多次 islice 不会重建或回放已消费的样本。
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._ensure_started()
        return self

    # 只搬运 tensor 字段，元数据保持原样；non_blocking 允许 pinned CPU 来源与 GPU 工作重叠。
    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch of tensors to the target device (GPU)."""
        if self.target_device is None:
            return batch
        return {
            k: v.to(self.target_device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # 先传播后台错误，再等待队列元素；sentinel 转为 StopIteration，普通 batch 在消费线程搬目标设备。
    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._error is not None:
            raise self._error
        item = self._queue.get()
        if item is self._SENTINEL:
            if self._error is not None:
                raise self._error
            raise StopIteration
        return self._to_device(item)
