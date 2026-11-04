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

# TRT-LLM 的 PyTorch 后端用 SaveHiddenStates 捕获训练数据，仓库补丁把结果从文件转存到 Mooncake。
# 本适配器负责配置和键重建；只支持单节点 TP，最后隐状态是 post-norm，不能套用 vLLM 的 pre-norm 假设。

"""
TensorRT-LLM Ray actor engine for distributed deployment.

Wraps TensorRT-LLM's PyTorch-backend ``LLM`` running in ``SaveHiddenStates``
speculative mode.  TRT-LLM natively captures EAGLE3 aux-layer + final post-norm
hidden states into a per-forward buffer; TorchSpec's patch
(``patches/trtllm/.../trtllm.patch``) redirects that buffer to Mooncake instead
of ``.pt`` files when ``TORCHSPEC_TRTLLM_MOONCAKE`` is set.

This engine therefore only has to:
  1. configure ``SaveHiddenStatesDecodingConfig`` with the right capture layers,
  2. flip the env flag and export the Mooncake connection params, and
  3. map each ``RequestOutput.request_id`` back to the Mooncake key the patch
     wrote (using the SAME sanitization as the patch).

Scope: single-node tensor parallelism.  TRT-LLM spawns its own MPI workers, so
multi-node TP needs additional orchestration and is intentionally deferred.
"""

import gc
import os
import re
import socket
import tempfile
from typing import Any

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.transfer.mooncake.eagle_store import HIDDEN_STATES_STORAGE_DTYPE
from torchspec.utils.logging import logger, setup_file_logging
from torchspec.utils.misc import get_default_eagle3_aux_layer_ids

# Keys managed internally by TorchSpec — ignored if present in trtllm_extra_args.
_PROTECTED_ENGINE_KEYS = frozenset(
    {
        "model",
        "backend",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "trust_remote_code",
        "speculative_config",
        "kv_cache_config",
        "disable_overlap_scheduler",
        "enable_chunked_prefill",
    }
)

# Env flag the patched SaveHiddenStatesResourceManager gates on.
_TORCHSPEC_MOONCAKE_ENV = "TORCHSPEC_TRTLLM_MOONCAKE"


# 将请求 ID 的非字母数字字符替换为下划线，并避免数字开头；算法必须和写入补丁一致才能取回同一对象。
def _sanitize_mooncake_key(key: str) -> str:
    """Reconstruct the Mooncake key the patch wrote for a request.

    MUST stay in sync with ``_sanitize_mooncake_key`` in
    ``patches/trtllm/<ver>/trtllm.patch``.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    if sanitized and sanitized[0].isdigit():
        sanitized = "k" + sanitized
    return sanitized


# 把 TRT-LLM 的捕获模式适配成训练数据生产者，返回元数据而不是经 Ray 搬运大张量。
class TrtllmEngine(InferenceEngine, RayActor):
    """Ray actor wrapping TensorRT-LLM's PyTorch ``LLM`` in SaveHiddenStates mode.

    Accepts pre-tokenized input_ids or formatted prompt strings, runs prefill,
    and returns Mooncake keys + tensor metadata (the hidden states themselves are
    streamed to Mooncake by the patched resource manager).
    """

    # 保存放置参数、空模型与捕获状态；实际 GPU 可见性必须在后端导入和 MPI 启动前设置。
    def __init__(
        self,
        args,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int = 1,
        node_rank: int = 0,
        engine_group: int = 0,
    ):
        self.args = args
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine
        self.node_rank = node_rank
        self._engine = None
        self._mooncake_config = None
        self._hidden_size = None
        self.local_gpu_id = None
        self.aux_hidden_state_layer_ids: list[int] = []
        self._store_last_hidden_states = True

        setup_file_logging("inference", self.rank, group=engine_group)

    # 拒绝跨节点和 PP，绑定 GPU，导出存储开关及每引擎键前缀，再创建带隐状态捕获的 LLM。
    # 不同副本的请求计数可能都从零开始，rank 前缀用于避免它们写入同名对象。
    def init(
        self,
        mooncake_config=None,
        dist_init_addr: str | None = None,
        pre_allocated_port: int | None = None,
    ) -> None:
        # TRT-LLM manages cross-worker init internally over MPI; dist_init_addr /
        # pre_allocated_port are accepted for interface parity with the other
        # engines but unused for single-node TP.
        del dist_init_addr, pre_allocated_port

        nnodes = getattr(self.args, "trtllm_nnodes", 1)
        if nnodes > 1:
            raise NotImplementedError(
                "TrtllmEngine currently supports single-node TP only "
                f"(trtllm_nnodes={nnodes}). Multi-node TP is not yet wired up."
            )
        pp_size = getattr(self.args, "trtllm_pp_size", 1)
        assert pp_size == 1, f"trtllm_pp_size must be 1, got {pp_size}"

        # GPU pinning, before any CUDA/tensorrt_llm init (TRT reads CVD at import
        # and picks the device by MPI rank). tp=1: the factory drops the NOSET
        # override so Ray scopes CVD to this actor's single GPU -- don't override
        # it. tp>1: keep all GPUs visible and pin the contiguous block ourselves.
        if self.base_gpu_id is not None:
            if self.num_gpus_per_engine > 1:
                gpu_ids = [str(self.base_gpu_id + i) for i in range(self.num_gpus_per_engine)]
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            self.local_gpu_id = 0
            torch.cuda.set_device(self.local_gpu_id)
            os.environ["LOCAL_RANK"] = "0"
            logger.info(
                f"TrtllmEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )

        # 存储字段开关必须在 MPI worker 创建前通过环境导出；子进程不会自动看到之后的 Python 属性变化。
        self._store_last_hidden_states = getattr(self.args, "store_last_hidden_states", True)
        # Tell the patched resource manager whether to store last_hidden_states.
        # Set before LLM construction so it propagates to the spawned MPI workers.
        os.environ["TORCHSPEC_TRTLLM_STORE_LAST_HIDDEN"] = (
            "1" if self._store_last_hidden_states else "0"
        )
        # Per-engine Mooncake key prefix: each data-parallel engine has its own
        # LLM request_id counter, so without a prefix they collide in the shared
        # store. Set before LLM build so the MPI worker (the patch) inherits it.
        self._key_prefix = f"e{self.rank}_"
        os.environ["TORCHSPEC_TRTLLM_KEY_PREFIX"] = self._key_prefix
        self._mooncake_config = mooncake_config
        self._setup_mooncake_env(mooncake_config)

        # 先解析 H 与捕获层，既用于配置后端，也用于稍后重建与写入张量一致的元数据。
        self._hidden_size = self._get_hidden_size_from_engine()
        self.aux_hidden_state_layer_ids = self._resolve_aux_layer_ids()

        tp_size = self.num_gpus_per_engine
        mem_fraction = getattr(self.args, "trtllm_mem_fraction_static", 0.8)

        logger.info(
            f"TrtllmEngine rank {self.rank}: BEFORE init - "
            f"base_gpu_id={self.base_gpu_id}, tp_size={tp_size}, "
            f"aux_hidden_state_layer_ids={self.aux_hidden_state_layer_ids}, "
            f"hidden_size={self._hidden_size}"
        )

        # 单节点 TP 启动由 TRT-LLM 负责，工厂传入的跨节点初始化参数在这里明确不使用。
        self._init_engine(tp_size, mem_fraction)

        logger.info(
            f"TrtllmEngine rank {self.rank}: initialized from {self.args.target_model_path} "
            f"(tp_size={tp_size}, aux_layers={self.aux_hidden_state_layer_ids}, "
            f"hidden_size={self._hidden_size})"
        )

    # 在 MPI 子进程启动前导出 Mooncake 配置和重定向标志；有配置时先验证两个服务端口可达。
    def _setup_mooncake_env(self, mooncake_config) -> None:
        """Export Mooncake env + the redirect flag so MPI workers inherit them.

        TRT-LLM spawns its workers from this process, so any env set here before
        constructing ``LLM`` is visible to the patched resource manager running
        inside those workers.
        """
        # Always set the flag; the patch only stores when a Mooncake store is
        # actually reachable, so a missing master simply logs and skips.
        os.environ[_TORCHSPEC_MOONCAKE_ENV] = "1"

        if mooncake_config is None:
            logger.warning(
                f"TrtllmEngine rank {self.rank}: no mooncake_config provided; "
                "hidden states will NOT be stored."
            )
            return

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "localhost"
            logger.warning(
                f"TrtllmEngine rank {self.rank}: failed to get local IP, using localhost"
            )

        mooncake_config.local_hostname = local_ip
        mooncake_config.export_env()

        from torchspec.transfer.mooncake.utils import check_mooncake_master_available

        check_mooncake_master_available(
            mooncake_config.master_server_address,
            mooncake_config.metadata_server,
        )

    # 按层执行后的索引筛选合法辅助层，不做 vLLM 的 +1 转换；最后归一化后的状态另用 -1 表示。
    def _resolve_aux_layer_ids(self) -> list[int]:
        """Aux capture layers (post-layer indices, no +1 shift unlike vLLM).

        TRT-LLM's capture hook fires with ``self.layer_idx`` *after* each
        decoder layer runs, so the layer ids map directly (same convention as
        sglang).  The final post-norm state is requested separately via the
        ``-1`` entry added in ``_init_engine``.
        """
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        cfg = getattr(cfg, "text_config", cfg)
        num_layers = cfg.num_hidden_layers

        if self.args.aux_hidden_states_layers is not None:
            aux_ids = list(self.args.aux_hidden_states_layers)
        else:
            aux_ids = get_default_eagle3_aux_layer_ids(self.args.target_model_path)
            if self.rank == 0:
                logger.info(f"Using default aux hidden state layer ids: {aux_ids}")

        aux_ids = [lid for lid in aux_ids if 0 <= lid < num_layers]
        return aux_ids

    # 延迟导入 TRT-LLM，设置辅助层加 -1 捕获槽，并关闭前缀复用、分块 prefill 和调度重叠。
    # 这些约束保证每个请求完整前向且捕获缓冲区在读取前不被下一次执行覆盖。
    def _init_engine(self, tp_size: int, mem_fraction: float | None) -> None:
        """Construct the TRT-LLM PyTorch ``LLM`` in SaveHiddenStates mode."""
        # Imported here (not at module scope) so that importing this module
        # without a CUDA driver -- e.g. on an HF/vLLM/SGLang-only host -- does
        # not trigger tensorrt_llm's libcuda load and break the engine package.
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig, SaveHiddenStatesDecodingConfig

        # eagle3_layers_to_capture: aux layers + the final post-norm state (-1).
        # The resource manager orders -1 last in the capture buffer, which is the
        # split point the patch relies on (aux = [:, :-H], last = [:, -H:]).
        # 辅助层在特征轴前部，-1 的最终 post-norm 表示放最后；补丁据最后 H 列切出 last_hidden_states。
        layers_to_capture = set(self.aux_hidden_state_layer_ids) | {-1}

        # output_directory is a required field on the config even though the
        # Mooncake redirect never writes to disk; point it at a throwaway path.
        spec_config = SaveHiddenStatesDecodingConfig(
            output_directory=os.path.join(tempfile.gettempdir(), "torchspec_trtllm_unused"),
            eagle3_layers_to_capture=layers_to_capture,
        )

        engine_kwargs: dict[str, Any] = {}

        extra_args = getattr(self.args, "trtllm_extra_args", None)
        if extra_args:
            if isinstance(extra_args, DictConfig):
                extra = OmegaConf.to_container(extra_args, resolve=True)
            else:
                extra = dict(extra_args) if not isinstance(extra_args, dict) else extra_args
            blocked = extra.keys() & _PROTECTED_ENGINE_KEYS
            if blocked:
                logger.warning(
                    f"trtllm extra_args contains protected keys that will be ignored: "
                    f"{sorted(blocked)}. These are managed internally by TorchSpec."
                )
                extra = {k: v for k, v in extra.items() if k not in _PROTECTED_ENGINE_KEYS}
            engine_kwargs.update(extra)

        # Block reuse must be OFF for hidden-state capture: when two prompts
        # share a prefix, TRT-LLM reuses the cached KV blocks and only runs a
        # forward pass over the new suffix tokens, so SaveHiddenStates captures
        # hidden states for fewer tokens than the prompt length. The trainer
        # then sees a shape mismatch (engine reports full seq_len, store holds
        # only the recomputed tokens). Disabling reuse forces a full prefill.
        # 共享前缀缓存只保留 attention 的 K/V，并不自动提供每个 token 的所有辅助层输出。
        # 例如第二个请求复用了前三个 token，仅重算后两项就无法构成声明长度为 5 的训练样本。
        kv_cache_kwargs: dict[str, Any] = {
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
        }
        if mem_fraction is not None:
            kv_cache_kwargs["free_gpu_memory_fraction"] = mem_fraction
        engine_kwargs["kv_cache_config"] = KvCacheConfig(**kv_cache_kwargs)

        # 长度和 batch 上限控制引擎预留资源；若 extra_args 已指定相关值，setdefault 会保留它。
        max_seq_length = getattr(self.args, "max_seq_length", None)
        if max_seq_length:
            engine_kwargs.setdefault("max_seq_len", max_seq_length)

        inference_batch_size = getattr(self.args, "inference_batch_size", None)
        if inference_batch_size is not None:
            engine_kwargs.setdefault("max_batch_size", inference_batch_size)

        # Protected, set last so extra_args cannot override:
        #  - overlap scheduler off: the capture buffer is reused every forward;
        #    the overlap pipeline could launch the next forward before
        #    process_and_save reads it.
        #  - chunked prefill off: we need each request's full prefill in one
        #    forward so the patch's per-request token offsets stay contiguous.
        # 捕获区每次前向复用，必须等本次读取完才能让下一次写入；这是生命周期约束而非单纯性能调优。
        engine_kwargs["disable_overlap_scheduler"] = True
        engine_kwargs["enable_chunked_prefill"] = False

        self._engine = LLM(
            model=self.args.target_model_path,
            backend="pytorch",
            tensor_parallel_size=tp_size,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
            speculative_config=spec_config,
            **engine_kwargs,
        )
        logger.info(
            f"TrtllmEngine rank {self.rank}: LLM constructed with "
            f"layers_to_capture={sorted(layers_to_capture)}"
        )

    # 接受 [S] 或单样本 [1, S] 并返回 [S]；拒绝把真正的多样本矩阵误当一条请求。
    def _normalize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            return input_ids.squeeze(0)
        if input_ids.dim() == 1:
            return input_ids
        raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")

    # 预填充输入后按后端 request_id 和引擎前缀重建写入键；返回 prompt 长度、token 列表及原训练 mask。
    # max_tokens=1 服务于后端捕获模式，训练张量的长度仍由 prompt 决定，并非 decode completion 样本。
    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str | None] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Run prefill and return Mooncake keys + tensor metadata.

        Hidden states are stored to Mooncake by the patched resource manager;
        here we reconstruct each key from ``RequestOutput.request_id`` (which
        equals the internal ``py_request_id`` the patch keyed on).
        """
        if self._engine is None:
            raise RuntimeError("TrtllmEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        if multimodal_inputs is not None and any(m for m in multimodal_inputs):
            raise NotImplementedError("TrtllmEngine does not support multimodal inputs yet.")

        # 文本路径交后端分词；token 路径先去单样本 batch 维并按后端字典格式包装。
        use_prompts = formatted_prompts is not None
        if use_prompts:
            batch_size = len(formatted_prompts)
            inputs: list = list(formatted_prompts)
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            if input_ids_list is None:
                raise ValueError("input_ids_ref resolved to None")
            batch_size = len(input_ids_list)
            inputs = [
                {"prompt_token_ids": self._normalize_input_ids(ids).tolist()}
                for ids in input_ids_list
            ]

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = data_id
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match batch size {batch_size}"
            )

        # mask 属于用户样本而不是 TRT 内部请求编号，因此先按 data_id 建映射供结果组装。
        packed_loss_mask_map: dict[str, str | None] = {}
        if packed_loss_mask_list is not None:
            for i, did in enumerate(data_ids):
                if i < len(packed_loss_mask_list):
                    packed_loss_mask_map[did] = packed_loss_mask_list[i]

        # Prefill-only: SaveHiddenStates forces max_new_tokens=1 internally, but
        # we set it here too to avoid allocating decode resources.
        from tensorrt_llm import SamplingParams

        sampling_params = SamplingParams(max_tokens=1)

        outputs = self._engine.generate(inputs, sampling_params, use_tqdm=False)

        results: list[dict[str, Any]] = []
        for i, output in enumerate(outputs):
            did = data_ids[i]
            seq_len = len(output.prompt_token_ids)
            # 后端 patch 使用内部 request_id 写入；不能改用 data_id 猜存储键，两个编号空间并不相同。
            mooncake_key = _sanitize_mooncake_key(f"{self._key_prefix}{output.request_id}")

            result: dict[str, Any] = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": self._get_tensor_shapes(seq_len),
                "tensor_dtypes": self._get_tensor_dtypes(),
                "data_id": did,
                "seq_len": seq_len,
                "input_ids_list": list(output.prompt_token_ids),
            }
            packed_loss_mask = packed_loss_mask_map.get(did)
            if packed_loss_mask is not None:
                result["packed_loss_mask"] = packed_loss_mask
            results.append(result)

        logger.debug(
            f"TrtllmEngine rank {self.rank}: generated {len(results)} mooncake results "
            f"for data_ids={data_ids}"
        )
        return results

    # 以 LLM 对象存在作为轻量就绪信号，未执行实际推理。
    def health_check(self, timeout: float = 5.0) -> bool:
        return self._engine is not None

    # 尝试关闭后端；即使抛异常也清空引用、收集 Python 垃圾并释放未使用的 CUDA 缓存。
    def shutdown(self) -> None:
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception as e:
                logger.warning(f"TrtllmEngine rank {self.rank}: error during shutdown: {e}")
            finally:
                self._engine = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        logger.info(f"TrtllmEngine rank {self.rank}: shutdown complete")

    # 报告位置和模型维度，方便检查启动完成与元数据配置。
    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
            "hidden_size": self._hidden_size,
        }

    # 先读顶层 hidden_size，再读 text_config；两处都没有时明确报错而不猜测维度。
    def _get_hidden_size_from_engine(self) -> int:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            if text_config is not None:
                hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                f"Could not determine hidden_size from model config: {self.args.target_model_path}"
            )
        return hidden_size

    # 对应补丁的按 token 拼接布局：辅助层 [S, K×H]，最后 post-norm 隐状态可选为 [S, H]。
    def _get_tensor_shapes(self, seq_len: int) -> dict:
        """Shapes of the tensors the patch stored to Mooncake (no batch dim).

        The patch writes aux layers concatenated (``num_aux_layers * H``) as
        ``hidden_states`` and the final post-norm state (``H``) as
        ``last_hidden_states`` — matching the sglang engine's layout.
        """
        if self._hidden_size is None:
            raise ValueError(
                f"TrtllmEngine rank {self.rank}: hidden_size not initialized. Call init() first."
            )
        hidden_size = self._hidden_size
        num_aux_layers = len(self.aux_hidden_state_layer_ids)
        concat_hidden_size = num_aux_layers * hidden_size

        shapes = {
            "hidden_states": (seq_len, concat_hidden_size),
            "input_ids": (seq_len,),
        }
        if self._store_last_hidden_states:
            shapes["last_hidden_states"] = (seq_len, hidden_size)
        return shapes

    # 对外声明补丁存储的 BF16 隐状态和 long token，并遵循最后隐状态开关。
    def _get_tensor_dtypes(self) -> dict:
        dtypes = {
            "hidden_states": HIDDEN_STATES_STORAGE_DTYPE,
            "input_ids": torch.long,
        }
        if self._store_last_hidden_states:
            dtypes["last_hidden_states"] = HIDDEN_STATES_STORAGE_DTYPE
        return dtypes
