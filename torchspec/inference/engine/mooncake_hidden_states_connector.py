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

# vLLM 的隐状态发布桥：调度端记录请求/页号及清单，worker 从分页缓存取出完整请求并写入 Mooncake。
# 两端是不同进程中的不同实例，只通过 connector metadata 传递状态；请求结束不等于缓存页可以立即复用。

"""KV Connector that writes hidden states directly to Mooncake.

vLLM discovers this connector via ``kv_connector_module_path`` in the
``kv_transfer_config`` dict -- no registration in vLLM's factory needed.

Architecture note: vLLM creates separate connector instances for the scheduler
process and each worker process.  Scheduler-side methods (``build_connector_meta``,
``request_finished``) run on one instance; worker-side methods (``save_kv_layer``,
``wait_for_save``) run on another.  They do NOT share state.  Metadata returned
by ``request_finished`` must therefore be pre-computed on the scheduler side.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)

HIDDEN_STATES_DTYPE_STR = "bfloat16"


# 规范请求 ID 以构造可兼容的存储键；调度端生成清单和 worker 写入必须使用同一转换。
def _sanitize_mooncake_key(key: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    if sanitized and sanitized[0].isdigit():
        sanitized = "k" + sanitized
    return sanitized


# 把每个 slot 拆成块号和块内位置，以高级索引收集 [token, 捕获层, H]，截到实际 token 数。
# 这里高级索引会物化张量，得到与原分页缓存独立的存储，而非简单 reshape 视图。
def _extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from an LBNHC hidden-state cache view.

    ``kv_cache`` has shape ``(num_blocks, num_heads, block_size, head_size)``.
    This matches latest vLLM's standardized hidden-state connector layout.
    """
    block_size = kv_cache.shape[2]
    return kv_cache[
        slot_mapping // block_size,
        :,
        slot_mapping % block_size,
    ][:num_tokens]


# 把逻辑顺序的物理块号展开为 slot，并截去末块未用空间。
# 例如块号 [2, 5]、页长 4、长度 6，映射为 [8, 9, 10, 11, 20, 21]。
def _slot_mapping_from_block_ids(
    block_ids: list[int],
    page_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute a complete slot mapping from finished-request block IDs."""
    block_ids_gpu = torch.tensor(block_ids, dtype=torch.int64, device=device)
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (block_ids_gpu.unsqueeze(1) * page_size + offsets).flatten()[:num_tokens]


# 为单个捕获层计算多个物理页内的指针和字节数，让 Mooncake 按逻辑序列顺序拼成 [S, H] 对象。
# 用 stride 而非假定整个四维张量连续；要求同页的 token 和 hidden 维连续，末页只发送有效 token。
def _cache_layer_fragments(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    layer_position: int,
    num_tokens: int,
) -> tuple[list[int], list[int]]:
    """Return registered-cache source slices for one logical layer tensor.

    The required vLLM layout is ``(blocks, layers, block_size, hidden)``.
    Tokens for one layer are contiguous inside each block, so Mooncake's
    multi-buffer put can concatenate those block slices into the exact
    ``(num_tokens, hidden)`` object consumed by training.
    """
    if kv_cache.dim() != 4:
        raise ValueError(f"Expected a 4D LBNHC cache, got shape={tuple(kv_cache.shape)}")
    num_blocks, num_layers, block_size, hidden_size = kv_cache.shape
    if not 0 <= layer_position < num_layers:
        raise ValueError(
            f"Layer position {layer_position} is outside hidden-state cache shape "
            f"{tuple(kv_cache.shape)}"
        )
    if num_tokens <= 0:
        raise ValueError(f"Expected a positive token count, got {num_tokens}")
    if len(block_ids) * block_size < num_tokens:
        raise ValueError(
            f"Hidden-state cache has {len(block_ids) * block_size} slots for {num_tokens} tokens"
        )
    # 相邻 hidden 元素及相邻 token 行必须物理连续；否则单个指针加长度不能正确表示一页的数据。
    if kv_cache.stride(3) != 1 or kv_cache.stride(2) != hidden_size:
        raise ValueError(
            "Hidden-state cache tokens are not contiguous within each block: "
            f"shape={tuple(kv_cache.shape)}, stride={kv_cache.stride()}"
        )

    element_size = kv_cache.element_size()
    base_ptr = kv_cache.data_ptr()
    ptrs: list[int] = []
    sizes: list[int] = []
    remaining = num_tokens
    # 例如页长 4、请求长 6、H=8、BF16，两页对应大小 [64, 32] 字节。
    # 物理页可以不相邻；按 block_ids 顺序发送即可得到逻辑连续的 [6, 8] 层张量。
    for block_id in block_ids:
        if not 0 <= block_id < num_blocks:
            raise ValueError(f"Hidden-state block id {block_id} is outside [0, {num_blocks})")
        if remaining == 0:
            break
        tokens_in_block = min(block_size, remaining)
        # 块号决定物理页起点，layer_position 选择页内捕获层；乘 element_size 后才从元素偏移变成字节地址。
        element_offset = block_id * kv_cache.stride(0) + layer_position * kv_cache.stride(1)
        ptrs.append(base_ptr + element_offset * element_size)
        sizes.append(tokens_in_block * hidden_size * element_size)
        remaining -= tokens_in_block

    if remaining:
        raise ValueError(f"Hidden-state cache is missing {remaining} token slots")
    return ptrs, sizes


# 跨调度/worker 边界传递请求 ID、CPU token 和物理页号；保存页号意味着发布前必须保留页所有权。
@dataclass
class _PendingSave:
    req_id: str
    token_ids: torch.Tensor
    block_ids: list[int]


# 单步调度元数据容器，把上一轮完成请求的待发布列表交给执行 worker。
@dataclass
class MooncakeConnectorMetadata(KVConnectorMetadata):
    pending_saves: list[_PendingSave] = field(default_factory=list)


# 借用 KV connector 生命周期存储隐状态，不加载外部 KV；每个 PP 阶段只发布自己拥有的捕获层。
class MooncakeHiddenStatesConnector(KVConnectorBase_V1, SupportsHMA):
    """KV Connector that stores extracted hidden states directly to Mooncake.

    Must be used with vLLM's ``extract_hidden_states`` speculative method.
    Mooncake connection parameters are read from environment variables
    (exported by TorchSpec's VllmEngine before creating the LLM instance).
    """

    # 返回 False，声明不优先使用跨层块布局；实际布局仍通过缓存组和注册时校验确定。
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    # 解析隐藏缓存组、页长、捕获层及并行拓扑，分别初始化调度端字典与 worker 缓存引用。
    # 最后一捕获槽用于 last_hidden_states，其余槽才是训练辅助层；两者角色不能混淆。
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        # 普通 attention KV 与训练隐状态可以是不同缓存组，页长和页号必须从隐状态组取得。
        self._cache_layer_group_id: int = self._find_cache_layer_group_id(kv_cache_config)
        self._block_size = self._get_cache_block_size(
            vllm_config,
            kv_cache_config,
            self._cache_layer_group_id,
        )
        self.cache_layers: list[str] = []

        assert self._vllm_config.speculative_config is not None, (
            "MooncakeHiddenStatesConnector requires 'extract_hidden_states' speculative method"
        )
        spec_config = self._vllm_config.speculative_config.draft_model_config.hf_config
        self._layer_ids = list(getattr(spec_config, "eagle_aux_hidden_state_layer_ids", []))
        self.num_hidden_states = len(self._layer_ids)
        self._hidden_size = vllm_config.model_config.get_hidden_size()

        # The last aux layer is the model's final layer (appended by
        # VllmEngine for last_hidden_states capture).  Training hidden
        # states use the remaining layers.
        self._num_training_layers = max(self.num_hidden_states - 1, 1)

        # Scheduler-side state: track requests and pre-computed metadata
        # 这些字典属于调度端实例；worker 不能靠访问同名字段观察调度端修改，必须读 metadata。
        self._request_token_ids: dict[str, list[int]] = {}
        self._req_metadata: dict[str, dict[str, Any]] = {}
        self._pending_saves: dict[str, _PendingSave] = {}

        self._mooncake_store = None
        self._mooncake_setup_done = False
        self._tp_rank: int | None = None
        self._pp_rank: int | None = None
        self._pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self._num_target_layers = vllm_config.model_config.get_total_num_hidden_layers()
        # worker 持有真实缓存分配；直接 RDMA 注册只是借用这块内存，不把所有权转移给 store。
        self._kv_cache: torch.Tensor | None = None
        self._cache_gpu_direct_registered = False
        self._check_layer_layout()
        self._check_mooncake_env()

    # 要求捕获层严格递增且末项等于模型层总数，保证最后槽确实是末层之后的 pre-norm 状态。
    def _check_layer_layout(self) -> None:
        if not self._layer_ids:
            raise ValueError(
                "MooncakeHiddenStatesConnector requires a non-empty "
                "eagle_aux_hidden_state_layer_ids on the draft model config"
            )
        if any(a >= b for a, b in zip(self._layer_ids, self._layer_ids[1:])):
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be strictly ascending, "
                f"got {self._layer_ids}"
            )
        if self._layer_ids[-1] != self._num_target_layers:
            raise ValueError(
                "Final aux layer id must be the target model's post-last-layer "
                f"slot: got eagle_aux_hidden_state_layer_ids={self._layer_ids} "
                f"with vLLM num_hidden_layers={self._num_target_layers}. The "
                "HF config VllmEngine read and vLLM's own layer count disagree."
            )

    # 优先查唯一 HiddenStateCacheSpec 组；只有单组可退到零号，否则拒绝猜测，避免把普通 KV 页当隐状态页。
    @staticmethod
    def _find_cache_layer_group_id(kv_cache_config) -> int:
        """Find the isolated ``HiddenStateCacheSpec`` group.

        Latest vLLM places the hidden-state cache in group zero on the
        scheduler side and exposes the concrete group specs on workers.
        """
        if kv_cache_config is None:
            return 0

        from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

        groups = kv_cache_config.kv_cache_groups
        group_ids = [
            gid
            for gid, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
        ]
        if len(group_ids) == 1:
            return group_ids[0]
        if not group_ids and len(groups) == 1:
            return 0
        raise ValueError(
            "Could not uniquely identify the hidden-state KV cache group "
            f"among {len(groups)} groups: {group_ids}"
        )

    # worker 从具体缓存组规格读页长，调度端尚无缓存配置时用引擎缓存页长。
    @staticmethod
    def _get_cache_block_size(vllm_config, kv_cache_config, group_id: int) -> int:
        if kv_cache_config is None:
            return vllm_config.cache_config.block_size
        return kv_cache_config.kv_cache_groups[group_id].kv_cache_spec.block_size

    # 懒读取并缓存 vLLM TP rank，读取失败按零处理；后续据此限制重复写入者。
    def _get_tp_rank(self) -> int:
        if self._tp_rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                self._tp_rank = get_tensor_model_parallel_rank()
            except Exception:
                self._tp_rank = 0
        return self._tp_rank

    # 懒读取当前 PP 阶段号，非分布式可退到零，供捕获层归属计算使用。
    def _get_pp_rank(self) -> int:
        if self._pp_rank is None:
            try:
                from vllm.distributed import get_pp_group

                self._pp_rank = get_pp_group().rank_in_group
            except Exception:
                self._pp_rank = 0
        return self._pp_rank

    # 按本 PP 阶段的层区间筛选捕获列表的位置；捕获槽表示层执行之后，因此使用 start < id <= end。
    # 返回的是捕获列表下标，而不是模型层号；用它索引缓存的层轴。
    def _local_layer_positions(self) -> list[int]:
        if self._pp_size == 1:
            return list(range(self.num_hidden_states))

        from vllm.distributed.utils import get_pp_indices

        start_layer, end_layer = get_pp_indices(
            self._num_target_layers,
            self._get_pp_rank(),
            self._pp_size,
        )
        return [
            index
            for index, layer_id in enumerate(self._layer_ids)
            if (layer_id == 0 and self._get_pp_rank() == 0) or (start_layer < layer_id <= end_layer)
        ]

    # 缺少 master 地址时提前失败，避免推理成功却从未发布训练数据。
    @staticmethod
    def _check_mooncake_env() -> None:
        if not os.environ.get("MOONCAKE_MASTER_SERVER") and not os.environ.get(
            "MOONCAKE_MASTER_HOST"
        ):
            raise RuntimeError(
                "MooncakeHiddenStatesConnector requires MOONCAKE_MASTER_SERVER or "
                "MOONCAKE_MASTER_HOST to be set; without a store no hidden states "
                "are published and training would silently receive nothing."
            )

    # 只让每个 TP 组的 rank 0 创建存储；配置从引擎事先导出的环境读取，并绑定当前 CUDA 设备。
    def _ensure_mooncake_store(self) -> bool:
        if self._mooncake_setup_done:
            return self._mooncake_store is not None

        if self._get_tp_rank() != 0:
            self._mooncake_setup_done = True
            return False

        from torchspec.config.mooncake_config import MooncakeConfig
        from torchspec.transfer.mooncake.eagle_store import EagleMooncakeStore

        config = MooncakeConfig.from_env()
        store = EagleMooncakeStore(config)

        device: torch.device | None = None
        if torch.cuda.is_initialized():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        store.setup(device=device)

        self._mooncake_store = store
        self._mooncake_setup_done = True
        logger.info(
            "MooncakeHiddenStatesConnector: store initialized "
            f"(master={config.master_server_address})"
        )
        return True

    # ==============================
    # Worker-side methods
    # ==============================
    # 此连接器只生产训练隐状态，没有从远端恢复 KV 的任务，因此该生命周期回调为空。
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    # 没有远端层加载需要等待；保留空实现满足 vLLM connector 接口。
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    # 只暴露已经完成的后台写入错误，不等待仍在途的写入；消费者会逐键等待可见性。
    def wait_for_save(self):
        if self._mooncake_store is not None:
            # Do not serialize the engine step behind the RDMA write.  Readers
            # wait for the published keys before moving bytes; here we only
            # surface failures from puts that have already completed.
            self._mooncake_store.check_async_errors()

    # 定位唯一 CacheOnlyAttentionLayer，保存缓存张量并校验页长，随后尝试将其直接注册为 RDMA 源。
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )

        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )
        self._kv_cache = kv_caches[self.cache_layers[0]]
        if self._block_size != self._kv_cache.shape[2]:
            raise ValueError(
                "Hidden-state block-size mismatch: "
                f"derived {self._block_size}, cache view has {self._kv_cache.shape[2]}"
            )
        self._register_hidden_state_cache_for_gpu_direct()

    # 仅在 CUDA BF16、RDMA、支持多缓冲写入且布局合法时注册整块缓存；任何不支持条件保留 staging 回退。
    def _register_hidden_state_cache_for_gpu_direct(self) -> None:
        """Try to make the vLLM hidden-state cache a Mooncake RDMA source.

        Registration is an optional fast path. Unsupported layouts, older
        Mooncake clients, CUDA-VMM registration failures, and non-RDMA configs
        retain the existing registered staging-buffer path.
        """
        assert self._kv_cache is not None
        self._cache_gpu_direct_registered = False
        if not self._kv_cache.is_cuda or self._kv_cache.dtype != torch.bfloat16:
            logger.info(
                "Direct hidden-cache publication requires a CUDA BF16 cache; got %s/%s",
                self._kv_cache.device,
                self._kv_cache.dtype,
            )
            return
        # 每个 TP 组只允许指定发布者建立存储；其他 rank 计算模型但不重复注册和发送相同层。
        if not self._ensure_mooncake_store():
            return
        if not self._mooncake_store.config.enable_gpu_direct:
            return
        if self._mooncake_store.config.protocol.lower() != "rdma":
            logger.warning("GPU Direct hidden-cache publication requires Mooncake RDMA")
            return
        # 直接分页路径需要把不相邻页拼成同一远端对象，普通单指针 put 无法表达这种布局。
        if not self._mooncake_store.supports_multi_buffer_put:
            logger.warning(
                "Mooncake lacks batch_put_from_multi_buffers; using the registered staging buffer"
            )
            return

        try:
            # Validate the layout before paying the RDMA registration cost.
            _cache_layer_fragments(self._kv_cache, [0], 0, 1)
            # 注册成本按整块缓存承担一次，后续请求只提交其使用的子区间；调用方保持缓存分配存活。
            cache_nbytes = self._kv_cache.numel() * self._kv_cache.element_size()
            self._cache_gpu_direct_registered = self._mooncake_store.register_external_buffer(
                self._kv_cache.data_ptr(), cache_nbytes
            )
        except Exception:
            logger.warning(
                "Failed to register the hidden-state cache for direct Mooncake puts; "
                "using the registered staging buffer",
                exc_info=True,
            )
            return

        if getattr(self, "_cache_gpu_direct_registered", False):
            logger.info(
                "Registered %.1f GiB vLLM hidden-state cache for direct paged Mooncake puts",
                cache_nbytes / (1024**3),
            )
        else:
            logger.warning(
                "Mooncake rejected hidden-state cache registration; "
                "using the registered staging buffer"
            )

    # 逐层执行回调不保证持有完整请求，本实现不在这里保存；完整请求由 get_finished 发布。
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # Latest vLLM extracts completed hidden-state caches from get_finished.
        # Per-layer metadata is not guaranteed to contain a complete request.
        pass

    # 按结束请求页号 gather 全序列，再为本阶段每个捕获层输出独立 _hs 和 _ids 对。
    # 同一 token 列表按层重复发布，使每层对象都符合训练侧的单对象读取协议。
    def _pending_save_tensors(
        self,
        pending: _PendingSave,
        local_positions: list[int],
    ) -> list[tuple[str, torch.Tensor]]:
        assert self._kv_cache is not None

        num_tokens = pending.token_ids.shape[0]
        slot_mapping = _slot_mapping_from_block_ids(
            pending.block_ids,
            self._block_size,
            num_tokens,
            device=self._kv_cache.device,
        )
        if slot_mapping.shape[0] < num_tokens:
            raise RuntimeError(
                f"Completed request {pending.req_id} has only "
                f"{slot_mapping.shape[0]} hidden-state slots for {num_tokens} tokens"
            )

        # 先按请求的逻辑 token 顺序 gather，得到独立存储；因此原分页缓存稍后复用不会改写这些源张量。
        hidden_states_3d = _extract_from_kv_cache(
            self._kv_cache,
            slot_mapping,
            num_tokens,
        )
        if hidden_states_3d.dtype != torch.bfloat16:
            hidden_states_3d = hidden_states_3d.to(torch.bfloat16)
        input_ids = pending.token_ids.to(hidden_states_3d.device)
        mooncake_key = _sanitize_mooncake_key(pending.req_id)

        tensors: list[tuple[str, torch.Tensor]] = []
        # 同一 worker 只发送其 PP 阶段拥有的层；position 索引捕获轴，键后缀使用真正模型层 ID。
        for position in local_positions:
            layer_key = f"{mooncake_key}_layer{self._layer_ids[position]}"
            tensors.extend(
                [
                    (f"{layer_key}_hs", hidden_states_3d[:, position, :]),
                    (f"{layer_key}_ids", input_ids),
                ]
            )
        return tensors

    # 优先用已注册缓存直接发送；否则 gather 后把多个请求/层的张量装进固定容量 staging 批次。
    # 单张量超过容量立即失败，不能靠分批绕过它；只有多张量累计超限才提前提交当前批次。
    def _publish_pending_saves(self, pending_saves: list[_PendingSave]) -> None:
        local_positions = self._local_layer_positions()
        if not local_positions:
            return
        if not self._ensure_mooncake_store():
            return
        assert self._kv_cache is not None
        if getattr(self, "_cache_gpu_direct_registered", False):
            self._publish_pending_saves_from_registered_cache(pending_saves, local_positions)
            return
        # 按字节容量合并多个键，降低每层一次 put 的调用开销；保持每个键对应一个完整张量。
        capacity = int(self._mooncake_store.config.host_buffer_size)
        keys: list[str] = []
        tensors: list[torch.Tensor] = []
        used_bytes = 0

        # 将已收集的键和张量交给存储，再清空闭包中的列表及字节计数，开始下一缓冲批次。
        def submit() -> None:
            nonlocal keys, tensors, used_bytes
            if keys:
                self._mooncake_store.put_raw_tensors(keys, tensors)
            keys = []
            tensors = []
            used_bytes = 0

        for pending in pending_saves:
            for key, tensor in self._pending_save_tensors(pending, local_positions):
                tensor_bytes = tensor.numel() * tensor.element_size()
                if tensor_bytes > capacity:
                    raise RuntimeError(
                        f"Mooncake tensor {key} needs {tensor_bytes} bytes, "
                        f"host buffer has {capacity}"
                    )
                # 当前张量本身可容纳，但累加超限时先发送上一组；新张量作为下一组起点，不切开对象。
                if keys and used_bytes + tensor_bytes > capacity:
                    submit()
                keys.append(key)
                tensors.append(tensor)
                used_bytes += tensor_bytes
        submit()

    # 直接发送隐藏缓存页的 scatter 源片段；token ID 较小，仍经普通 staging 路径发布。
    # 发送前同步 CUDA 写缓存事件；隐藏层的同步 put 返回后才允许 vLLM 回收原页。
    def _publish_pending_saves_from_registered_cache(
        self,
        pending_saves: list[_PendingSave],
        local_positions: list[int],
    ) -> None:
        """Publish layer objects straight from registered LBNHC cache pages."""
        assert self._kv_cache is not None
        hidden_keys: list[str] = []
        hidden_ptrs: list[list[int]] = []
        hidden_sizes: list[list[int]] = []
        id_keys: list[str] = []
        id_tensors: list[torch.Tensor] = []

        # 逐请求、逐本地捕获层组装页片段；清单中的层键与调度端 manifest 完全相同。
        for pending in pending_saves:
            num_tokens = pending.token_ids.shape[0]
            input_ids = pending.token_ids.to(self._kv_cache.device)
            mooncake_key = _sanitize_mooncake_key(pending.req_id)
            for position in local_positions:
                layer_key = f"{mooncake_key}_layer{self._layer_ids[position]}"
                ptrs, sizes = _cache_layer_fragments(
                    self._kv_cache,
                    pending.block_ids,
                    position,
                    num_tokens,
                )
                hidden_keys.append(f"{layer_key}_hs")
                # 一个远端对象对应多个源页地址；ptrs 的顺序就是训练端期望的 token 顺序，不按物理地址排序。
                hidden_ptrs.append(ptrs)
                hidden_sizes.append(sizes)
                id_keys.append(f"{layer_key}_ids")
                id_tensors.append(input_ids)

        if hidden_keys:
            # Mooncake's NIC read is outside CUDA stream ordering. Ensure every
            # cache-write kernel submitted before get_finished is visible
            # before RDMA reads the registered pages. The put itself is
            # synchronous, so its return is also the cache-block reuse fence.
            if self._kv_cache.is_cuda:
                cache_ready = torch.cuda.Event()
                cache_ready.record(torch.cuda.current_stream(self._kv_cache.device))
                cache_ready.synchronize()
            self._mooncake_store.put_from_registered_multi_buffers(
                hidden_keys,
                hidden_ptrs,
                hidden_sizes,
            )

        # Token IDs originate on the scheduler CPU and are tiny relative to
        # hidden states. Keep the existing registered staging path for them;
        # when the GPU send buffer is available this is a small HtoD copy, not
        # a hidden-state DtoH copy.
        # 小型 token ID 仍按 staging 容量分批，因此隐藏层发布完成后，ID 键仍可能在异步写入中。
        capacity = int(self._mooncake_store.config.host_buffer_size)
        keys: list[str] = []
        tensors: list[torch.Tensor] = []
        used_bytes = 0
        for key, tensor in zip(id_keys, id_tensors):
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tensor_bytes > capacity:
                raise RuntimeError(
                    f"Mooncake tensor {key} needs {tensor_bytes} bytes, buffer has {capacity}"
                )
            if keys and used_bytes + tensor_bytes > capacity:
                self._mooncake_store.put_raw_tensors(keys, tensors)
                keys = []
                tensors = []
                used_bytes = 0
            keys.append(key)
            tensors.append(tensor)
            used_bytes += tensor_bytes
        if keys:
            self._mooncake_store.put_raw_tensors(keys, tensors)

    # 消费调度元数据中的完成请求并发布；先捕获本地错误，让所有 PP 阶段都过 barrier 后再抛出，防止同伴被挂住。
    # 回报完成 ID 允许调度器释放页；异步 staging 可仍在途，但已 gather 的独立张量由传输流保护。
    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        pending_saves: list[_PendingSave] = []
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, MooncakeConnectorMetadata):
                pending_saves = metadata.pending_saves

        if not pending_saves:
            return None, None

        local_error: BaseException | None = None
        try:
            self._publish_pending_saves(pending_saves)
            if self._mooncake_store is not None:
                # Direct cache puts complete synchronously before block reuse.
                # The fallback path gathers independent tensors and records
                # their DtoH copies before cache release. Surface any completed
                # staging failure without draining active puts; readers wait
                # for every key.
                self._mooncake_store.check_async_errors()
        except BaseException as exc:
            local_error = exc

        # 本阶段 put 失败也必须参加阶段间同步，否则其他阶段可能永远等不到它。
        # barrier 之后再抛本地异常；它协调所有阶段到达发布点，不替代消费者对每个远端键的等待。
        if self._pp_size > 1:
            from vllm.distributed import get_pp_group

            # All stages reach the collective even if a local PUT failed, so a
            # single-stage exception cannot strand its peer before propagation.
            get_pp_group().barrier()

        if local_error is not None:
            raise local_error

        return {pending.req_id for pending in pending_saves}, None

    # ==============================
    # Scheduler-side methods
    # ==============================
    # 不提供外部 KV 前缀命中，返回零匹配且无需异步加载。
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    # 断言外部 token 数为零，守住只写连接器不参与外部 KV 加载的边界。
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert num_external_tokens == 0, "This connector is store-only"

    # 先把待发布任务移交本轮 metadata，再为新请求缓存 token 和可提前返回的形状/逐层清单。
    # 调度端不知道 worker 的 Python 对象状态，清单必须由请求与模型配置独立计算。
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        # 调度端先把上一轮结束的请求打包给 worker，再清空本地待发队列。
        # 不能从 worker 对象读取 _pending_saves：两端 Python 实例根本不共享这些字典。
        meta.pending_saves = list(self._pending_saves.values())
        self._pending_saves.clear()

        # 新请求只在这里登记一次完整 prompt；当前捕获协议不是逐 decode 步追加 token 的流式协议。
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            self._request_token_ids[new_req.req_id] = list(token_ids)

            seq_len = len(token_ids)
            # 对外描述最终拼接后的辅助层宽度，即使实际存储按层分开，训练侧看到的逻辑字段仍为 [S, K×H]。
            training_hidden_size = self._num_training_layers * self._hidden_size
            mooncake_key = _sanitize_mooncake_key(new_req.req_id)
            self._req_metadata[new_req.req_id] = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": {
                    "hidden_states": (seq_len, training_hidden_size),
                    "input_ids": (seq_len,),
                    "last_hidden_states": (seq_len, self._hidden_size),
                },
                "tensor_dtypes": {
                    "hidden_states": HIDDEN_STATES_DTYPE_STR,
                    "input_ids": "int64",
                    "last_hidden_states": HIDDEN_STATES_DTYPE_STR,
                },
                "num_layers": self.num_hidden_states,
                "input_ids_list": token_ids,
            }
            # Keep one object per captured layer for every topology. This maps
            # directly to paged cache fragments and avoids rebuilding the old
            # token-major packed object for TP-only engines. The metadata key is
            # retained for compatibility with existing readers.
            # 每个捕获层独立列出物理键和角色；最终槽从辅助层集合分离，供消费者重建 last_hidden_states。
            self._req_metadata[new_req.req_id]["pp_layer_manifest"] = [
                {
                    "layer_id": layer_id,
                    "mooncake_key": f"{mooncake_key}_layer{layer_id}",
                    "role": (
                        "last_hidden_states"
                        if layer_id == self._num_target_layers
                        else "hidden_states"
                    ),
                }
                for layer_id in self._layer_ids
            ]

        return meta

    # 移出请求缓存并排队保存 token 与页号，返回 True 告诉调度器暂缓释放页，同时返回提前构造的样本元数据。
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        token_ids = self._request_token_ids.pop(
            req_id,
            list(request.prompt_token_ids or []),
        )
        mooncake_meta = self._req_metadata.pop(req_id, None)
        self._pending_saves[req_id] = _PendingSave(
            req_id=req_id,
            token_ids=torch.tensor(token_ids, dtype=torch.long),
            block_ids=list(block_ids),
        )
        # Delay freeing the hidden-state cache blocks until get_finished has
        # completed every local Mooncake PUT.
        return True, mooncake_meta

    # 从多缓存组页表中选出隐状态组，再复用单组结束逻辑；不能直接使用 attention KV 组的块号。
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        cache_group_ids = block_ids[self._cache_layer_group_id] if block_ids else []
        return self.request_finished(request, cache_group_ids)

    # 向 vLLM 声明所需 LBNHC 布局；本实现使用的具体缓存视图为 [块, 捕获层, 页内 token, H]。
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return "LBNHC"
