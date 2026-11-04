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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# 通过 vLLM extract_hidden_states 与自定义 connector 采集完整 prompt 的逐层隐状态。
# Ray 外壳处理多节点部署和请求；真正写入发生在 worker 的 connector，元数据由调度端返回。

"""
VLLM Ray actor engine for distributed deployment.

Uses vLLM's ``extract_hidden_states`` speculative decoding method with a
custom ``MooncakeHiddenStatesConnector`` KV Connector to capture intermediate
hidden states and store them directly to Mooncake via RDMA.

This replaces the previous worker-extension approach that monkey-patched
``model.forward``.  The new approach uses only public vLLM APIs
(``speculative_config`` + ``kv_transfer_config``) and is compatible with
MRV2, CUDA graphs, and ``torch.compile``.
"""

import gc
import os
import socket
import threading
from typing import Any

import ray
import torch
from omegaconf import DictConfig, OmegaConf

from torchspec.inference.engine.base import InferenceEngine
from torchspec.ray.ray_actor import RayActor
from torchspec.utils.logging import logger, setup_file_logging
from torchspec.utils.misc import get_default_eagle3_aux_layer_ids

_PROTECTION_ENGINE_KEYS = frozenset(
    {
        "model",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "gpu_memory_utilization",
        "nnodes",
        "node_rank",
        "master_addr",
        "master_port",
        "speculative_config",
        "kv_transfer_config",
    }
)

# Target architectures whose vLLM model returns
# ``(IntermediateTensors, aux_hidden_states)`` from a non-last pipeline stage.
#
# Our vLLM patch stack enables aux capture on every PP rank, which also removes
# the ``SupportsPP`` check that made stock vLLM reject this configuration at
# config time.  A model without that return contract therefore builds its aux
# list on each stage, discards it, and dies much later inside memory profiling
# with an unpack error that names nothing relevant.  Keep this in sync with
# ``patches/vllm/<image-tag>/vllm_pp_hidden_states.patch``.
_PER_STAGE_AUX_CAPTURE_ARCHITECTURES = frozenset(
    {
        "KimiK3ForConditionalGeneration",
        "KimiLinearForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    }
)


# 封装一个请求入口或一个无前端的多节点 follower；后者同样消耗 GPU 参与 TP/PP 计算。
class VllmEngine(InferenceEngine, RayActor):
    """Ray actor wrapper for vLLM LLM engine with distributed deployment support.

    Uses vLLM's ``extract_hidden_states`` speculative method with a
    ``MooncakeHiddenStatesConnector`` to capture hidden states from selected
    model layers and write them directly to Mooncake.
    """

    # 保存放置参数及引擎、follower executor、监视线程的空引用，为两种节点角色准备生命周期状态。
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
        self._headless_executor = None
        self._headless_monitor_thread = None
        self._mooncake_config = None
        self._hidden_size = None
        self.local_gpu_id = None

        setup_file_logging("inference", self.rank, group=engine_group)

    # 总 GPU 数等于 nnodes×每节点 GPU 数，除以 PP 得到 TP；不整除则拓扑无效。
    # 例如两节点各 8 卡、PP=4，得到 TP=4；这 16 卡仍是一份模型副本。
    @staticmethod
    def _resolve_parallel_sizes(
        nnodes: int,
        num_gpus_per_node: int,
        pp_size: int,
    ) -> tuple[int, int]:
        world_size = nnodes * num_gpus_per_node
        if world_size % pp_size != 0:
            raise ValueError(f"vLLM world_size={world_size} must be divisible by pp_size={pp_size}")
        return world_size // pp_size, pp_size

    # 设置 Mooncake 环境，读取模型层数，把用户层索引转成 vLLM 捕获槽并追加最后 pre-norm 槽，再启动对应节点角色。
    def init(
        self,
        mooncake_config=None,
        dist_init_addr: str | None = None,
        pre_allocated_port: int | None = None,
    ) -> None:
        if self.base_gpu_id is not None:
            self.local_gpu_id = self.setup_gpu(self.base_gpu_id)
            logger.info(
                f"VllmEngine rank {self.rank}: base_gpu_id={self.base_gpu_id}, "
                f"using local GPU {self.local_gpu_id}"
            )

        self._mooncake_config = mooncake_config

        if mooncake_config is not None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "localhost"
                logger.warning(
                    f"VllmEngine rank {self.rank}: failed to get local IP, using localhost"
                )

            mooncake_config.local_hostname = local_ip
            # Export env vars so worker processes (and the connector) can
            # initialize their own Mooncake stores via MooncakeConfig.from_env().
            mooncake_config.export_env()

            from torchspec.transfer.mooncake.utils import (
                check_mooncake_master_available,
            )

            check_mooncake_master_available(
                mooncake_config.master_server_address,
                mooncake_config.metadata_server,
            )

        mem_fraction = getattr(self.args, "vllm_mem_fraction_static", None)
        pp_size = getattr(self.args, "vllm_pp_size", 1)

        if self.args.aux_hidden_states_layers is not None:
            self.aux_hidden_state_layer_ids = list(self.args.aux_hidden_states_layers)
        else:
            self.aux_hidden_state_layer_ids = get_default_eagle3_aux_layer_ids(
                self.args.target_model_path
            )
            if self.rank == 0:
                logger.info(
                    f"Using default aux hidden state layer ids: {self.aux_hidden_state_layer_ids}"
                )

        from transformers import AutoConfig as _AC

        # 保留外层配置用于架构支持校验，文本子配置用于层数和 H；多模态模型这两个层次可能不同。
        _outer_cfg = _AC.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        _cfg = getattr(_outer_cfg, "text_config", _outer_cfg)

        # Layer IDs use post-layer semantics: "capture the residual stream
        # after layer N runs".  vllm's capture hook fires at the INPUT of each
        # listed layer (= output of the previous layer), so we shift by +1 to
        # align with sglang's convention.
        # vllm's `_maybe_add_hidden_state` is called with `layer_idx + 1`
        # *after* each layer runs, so valid capture indices are
        # [0, num_hidden_layers]; we keep ids up to num_hidden_layers
        # (the pre-`norm` slot, see final-layer block below).
        num_layers = _cfg.num_hidden_layers
        # 例如用户选零基第 2 层的输出，vLLM 捕获槽对应 3。
        # 这只是层索引坐标转换，不是把 token 序列向右平移一位。
        self.aux_hidden_state_layer_ids = [
            lid + 1 for lid in self.aux_hidden_state_layer_ids if lid < num_layers
        ]
        if self.rank == 0:
            logger.info(
                f"Shifted aux layer ids +1 for vllm (post-layer → pre-next-layer): "
                f"{self.aux_hidden_state_layer_ids}"
            )

        # Append the model's final layer to capture last_hidden_states
        # (pre-norm) for target logit computation.  Index `num_hidden_layers`
        # is vllm's reserved post-last-layer / pre-`norm` slot, so training
        # can apply the model's final norm itself on top of this.
        # 最后槽用于训练端重建目标分布，保存的是 final norm 之前的残差流。
        # 其他后端可能已经做过 norm，消费者必须根据来源决定，不能重复归一化。
        final_layer_id = num_layers
        if final_layer_id not in self.aux_hidden_state_layer_ids:
            self.aux_hidden_state_layer_ids.append(final_layer_id)
            if self.rank == 0:
                logger.info(
                    f"Appended final layer {final_layer_id} for last_hidden_states: "
                    f"{self.aux_hidden_state_layer_ids}"
                )

        nnodes = getattr(self.args, "vllm_nnodes", 1)
        # 总卡数同时被 TP 与 PP 分摊，不能独立把两者都设为总卡数。
        tp_size, pp_size = self._resolve_parallel_sizes(
            nnodes,
            self.num_gpus_per_engine,
            pp_size,
        )
        self._check_per_stage_capture_support(pp_size, getattr(_outer_cfg, "architectures", None))

        logger.info(
            f"VllmEngine rank {self.rank}: BEFORE init - "
            f"base_gpu_id={self.base_gpu_id}, num_gpus={self.num_gpus_per_engine}, "
            f"tp_size={tp_size}, pp_size={pp_size}, nnodes={nnodes}, node_rank={self.node_rank}, "
            f"aux_hidden_state_layer_ids={self.aux_hidden_state_layer_ids}"
        )

        self._init_engine(
            tp_size, pp_size, nnodes, mem_fraction, dist_init_addr, pre_allocated_port
        )

        # Only node_rank=0 owns the LLM front end.  Follower nodes run vLLM's
        # headless MultiprocExecutor and therefore do not expose an LLM object.
        # follower 无 LLM 前端，不应访问不存在的前端对象；从已加载的配置取得同一 H。
        self._hidden_size = (
            _cfg.hidden_size
            if nnodes > 1 and self.node_rank > 0
            else self._get_hidden_size_from_engine()
        )

        logger.info(
            f"VllmEngine rank {self.rank}: initialized from {self.args.target_model_path} "
            f"(tp_size={tp_size}, aux_layers={self.aux_hidden_state_layer_ids}, hidden_size={self._hidden_size})"
        )

    # 限制子进程可见 GPU，配置捕获 connector、关闭前缀缓存并预留提取临时显存；只有头节点创建 LLM。
    def _init_engine(
        self,
        tp_size: int,
        pp_size: int,
        nnodes: int,
        mem_fraction: float | None,
        dist_init_addr: str | None,
        pre_allocated_port: int | None = None,
    ) -> None:
        """Initialize LLM with extract_hidden_states speculative config."""
        # Pin vLLM's mp workers to the correct physical GPUs.  Without this,
        # all engines on a node see every GPU and their workers collide on
        # devices 0..tp_size-1.
        if self.base_gpu_id is not None:
            gpu_ids = [str(self.base_gpu_id + i) for i in range(self.num_gpus_per_engine)]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            logger.info(
                f"VllmEngine rank {self.rank}: set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
            )

        # extract_hidden_states 决定捕获模型输出，kv_transfer_config 决定由哪个 connector 发布；两者缺一不可。
        engine_kwargs = {
            "model": self.args.target_model_path,
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": pp_size,
            "trust_remote_code": getattr(self.args, "trust_remote_code", True),
            "distributed_executor_backend": "mp",
            "disable_custom_all_reduce": True,
            "speculative_config": {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": list(self.aux_hidden_state_layer_ids)
                    }
                },
            },
            "kv_transfer_config": {
                "kv_connector": "MooncakeHiddenStatesConnector",
                "kv_connector_module_path": (
                    "torchspec.inference.engine.mooncake_hidden_states_connector"
                ),
                "kv_role": "kv_producer",
            },
        }

        extra_args = getattr(self.args, "vllm_extra_args", None)
        if extra_args:
            if isinstance(extra_args, DictConfig):
                extra = OmegaConf.to_container(extra_args, resolve=True)
            else:
                extra = dict(extra_args) if not isinstance(extra_args, dict) else extra_args
            blocked = extra.keys() & _PROTECTION_ENGINE_KEYS
            if blocked:
                logger.warning(
                    f"vllm extra_args contains protected keys that will be ignored: "
                    f"{sorted(blocked)}. These are managed internally by TorchSpec."
                )
                extra = {k: v for k, v in extra.items() if k not in _PROTECTION_ENGINE_KEYS}
            engine_kwargs.update(extra)

        inference_batch_size = getattr(self.args, "inference_batch_size", None)
        if inference_batch_size is not None:
            if "max_num_seqs" not in engine_kwargs:
                engine_kwargs["max_num_seqs"] = inference_batch_size
                logger.info(
                    f"VllmEngine rank {self.rank}: defaulting "
                    f"max_num_seqs={inference_batch_size} from inference_batch_size"
                )

        # 完整训练样本需要每个 token 的辅助层，不能仅计算未命中缓存的后缀。
        engine_kwargs["enable_prefix_caching"] = False

        # Prefill-only: no decode graphs needed and piecewise adds launch
        # overhead not worthwhile for variable-length prefills.
        # Cap capture sizes when the user overrides the mode via extra_args.
        comp_cfg = engine_kwargs.get("compilation_config", {})
        if isinstance(comp_cfg, dict):
            if "cudagraph_mode" not in comp_cfg:
                comp_cfg["cudagraph_mode"] = "NONE"
            elif inference_batch_size is not None and "cudagraph_capture_sizes" not in comp_cfg:
                comp_cfg["cudagraph_capture_sizes"] = [
                    2**i
                    for i in range(inference_batch_size.bit_length())
                    if 2**i <= inference_batch_size
                ]
            engine_kwargs["compilation_config"] = comp_cfg

        if "distributed_timeout_seconds" not in engine_kwargs:
            timeout_min = getattr(self.args, "distributed_timeout_minutes", 10)
            engine_kwargs["distributed_timeout_seconds"] = timeout_min * 60

        max_seq_length = getattr(self.args, "max_seq_length", None)
        if max_seq_length:
            engine_kwargs["max_model_len"] = max_seq_length

        if dist_init_addr:
            host, port_str = dist_init_addr.rsplit(":", 1)
            engine_kwargs["master_addr"] = host
            engine_kwargs["master_port"] = int(port_str)
        elif pre_allocated_port is not None:
            engine_kwargs["master_port"] = pre_allocated_port

        if nnodes > 1:
            engine_kwargs["nnodes"] = nnodes
            engine_kwargs["node_rank"] = self.node_rank

        # 用户显式利用率优先；自动估算为 connector 运行时临时张量留空间，这部分不一定被后端启动 profiler 计入。
        if mem_fraction is not None:
            engine_kwargs["gpu_memory_utilization"] = mem_fraction
        else:
            # vLLM's profiler doesn't account for the connector's runtime
            # allocations (extracting hidden states from KV cache during
            # save_kv_layer).  Auto-compute a utilization that reserves room.
            engine_kwargs["gpu_memory_utilization"] = self._compute_mem_fraction(engine_kwargs)

        # 一个跨节点副本只能有一个调度前端；其他节点进入 follower 协议并等待 head 下发工作。
        if nnodes > 1 and self.node_rank > 0:
            self._init_headless_executor(engine_kwargs)
            logger.info(
                f"VllmEngine rank {self.rank}: initialized headless follower "
                f"(node_rank={self.node_rank}) with layers={self.aux_hidden_state_layer_ids}"
            )
        else:
            from vllm import LLM

            self._engine = LLM(**engine_kwargs)
            logger.info(
                f"VllmEngine rank {self.rank}: initialized extract_hidden_states mode "
                f"with layers={self.aux_hidden_state_layer_ids}"
            )

    # PP 要求每个阶段都返回其拥有的辅助层；只放行已支持这种返回契约的架构，避免训练永远等待缺失层。
    @staticmethod
    def _check_per_stage_capture_support(pp_size: int, architectures: list[str] | None) -> None:
        """Reject pipeline parallelism for targets that cannot capture per stage.

        Hidden-state extraction under PP needs every stage to hand back the aux
        layers it owns.  Only the models listed in
        ``_PER_STAGE_AUX_CAPTURE_ARCHITECTURES`` do that; the rest silently drop
        theirs, so a run would train on fragments that no stage ever wrote.
        """
        if pp_size <= 1:
            return

        supported = sorted(_PER_STAGE_AUX_CAPTURE_ARCHITECTURES)
        if not architectures:
            raise ValueError(
                "vllm_pp_size > 1 requires a target model whose architecture is known "
                "to capture aux hidden states on every pipeline stage, but the target "
                f"config declares no architectures. Supported: {supported}"
            )

        unsupported = [
            arch for arch in architectures if arch not in _PER_STAGE_AUX_CAPTURE_ARCHITECTURES
        ]
        if unsupported:
            raise ValueError(
                f"Target architectures {unsupported} do not implement per-stage aux "
                f"hidden-state capture, so vllm_pp_size={pp_size} would publish "
                "fragments that no pipeline stage writes. Use vllm_pp_size=1, or add "
                "the non-last-rank capture return to the model in "
                "patches/vllm/<image-tag>/vllm_pp_hidden_states.patch and list it in "
                f"_PER_STAGE_AUX_CAPTURE_ARCHITECTURES. Supported: {supported}"
            )

    # follower 仅启动 MultiprocExecutor 和监视线程；若每节点都造 LLM，会重复建立调度端并破坏控制通信。
    def _init_headless_executor(self, engine_kwargs: dict) -> None:
        """Start the vLLM MP follower protocol without creating an LLM front end.

        vLLM multi-node multiprocessing has one scheduler/EngineCore on
        ``node_rank=0``.  Other nodes must create only a headless
        ``MultiprocExecutor``; constructing ``LLM`` on every node creates a
        second EngineCore on each follower and makes it issue control-plane
        collective RPCs without the leader message queue.
        """
        from vllm import EngineArgs
        from vllm.config import CompilationConfig
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        headless_kwargs = dict(engine_kwargs)

        kv_transfer_config = headless_kwargs.get("kv_transfer_config")
        if isinstance(kv_transfer_config, dict):
            headless_kwargs["kv_transfer_config"] = KVTransferConfig(**kv_transfer_config)

        compilation_config = headless_kwargs.get("compilation_config")
        if isinstance(compilation_config, dict):
            headless_kwargs["compilation_config"] = CompilationConfig(**compilation_config)

        engine_args = EngineArgs(**headless_kwargs)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.LLM_CLASS,
            headless=True,
        )
        self._headless_executor = MultiprocExecutor(
            vllm_config,
            monitor_workers=False,
        )
        self._headless_monitor_thread = threading.Thread(
            target=self._headless_executor.start_worker_monitor,
            kwargs={"inline": True},
            name=f"vllm-headless-monitor-rank-{self.rank}",
            daemon=True,
        )
        self._headless_monitor_thread.start()

    _VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
    _HEADLESS_MONITOR_JOIN_TIMEOUT_S = 30.0

    # 用最大序列长度×捕获层数×H×2 字节估算一次提取，再乘二作余量，从默认 0.9 扣除且不低于 0.4。
    # 这是静态估算，不是针对所有并发请求的严格内存上界；显式配置利用率时不会走这里。
    def _compute_mem_fraction(self, engine_kwargs: dict) -> float:
        """Auto-compute gpu_memory_utilization with connector overhead reserved.

        Starts from vLLM's default (0.9) and subtracts the estimated peak
        memory of the MooncakeHiddenStatesConnector's ``save_kv_layer``,
        which creates temporary GPU tensors that vLLM's profiler doesn't
        account for.
        """
        base = self._VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
        max_len = engine_kwargs.get("max_model_len")
        if max_len is None:
            return base

        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(
            self.args.target_model_path,
            trust_remote_code=getattr(self.args, "trust_remote_code", True),
        )
        hf_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        hidden_size = hf_cfg.hidden_size
        num_aux_layers = len(self.aux_hidden_state_layer_ids)

        # Peak per-request: _extract_from_kv_cache materialises
        # (seq_len, num_aux_layers * hidden_size) in bf16.
        connector_bytes = max_len * num_aux_layers * hidden_size * 2
        # 2x safety for PyTorch allocator fragmentation + small extras
        reserved_bytes = int(connector_bytes * 2)

        total_gpu_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        overhead_frac = reserved_bytes / total_gpu_bytes
        adjusted = base - overhead_frac
        adjusted = max(adjusted, 0.4)

        logger.info(
            f"VllmEngine rank {self.rank}: auto gpu_memory_utilization={adjusted:.3f} "
            f"(reserving {reserved_bytes / (1 << 30):.1f} GiB for connector: "
            f"{max_len} tokens × {num_aux_layers} layers × {hidden_size} hidden × bf16 × 2x safety)"
        )
        return adjusted

    # 构建 vLLM 请求并触发捕获，随后把 kv_transfer_params 中的键、逐层清单和原 loss mask 转回训练协议。
    # 仅生成一个 token 以完成后端请求，但本连接器存储的是 prompt 序列；缺失连接器元数据时跳过结果。
    def generate(
        self,
        data_id: str | list[str],
        input_ids_ref: ray.ObjectRef | list[torch.Tensor] | None = None,
        packed_loss_mask_list: list[str | None] | None = None,
        formatted_prompts: list[str] | None = None,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
        multimodal_inputs: list[dict] | None = None,
    ) -> list[dict]:
        """Generate hidden states for training data.

        Hidden states are captured by vLLM's ``extract_hidden_states``
        speculative method and stored to Mooncake by the
        ``MooncakeHiddenStatesConnector``.  Metadata comes back in
        ``output.kv_transfer_params``.
        """
        if self._engine is None:
            raise RuntimeError("VllmEngine not initialized. Call init() first.")

        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Exactly one of input_ids_ref or formatted_prompts must be set")

        use_prompts = formatted_prompts is not None
        input_ids_list: list[torch.Tensor] | None = None

        if use_prompts:
            batch_size = len(formatted_prompts)
        else:
            if isinstance(input_ids_ref, ray.ObjectRef):
                input_ids_list = ray.get(input_ids_ref)
            else:
                input_ids_list = input_ids_ref
            if input_ids_list is None:
                raise ValueError("input_ids_ref resolved to None")
            batch_size = len(input_ids_list)

        # 把文本/token/媒体规范成每请求字典，避免不同输入格式分叉成不同发布协议。
        prompts = self._build_prompts(
            formatted_prompts=formatted_prompts if use_prompts else None,
            input_ids_list=input_ids_list,
            multimodal_inputs=multimodal_inputs,
            batch_size=batch_size,
        )

        if isinstance(data_id, str):
            data_ids = [f"{data_id}_{i}" for i in range(batch_size)]
        elif len(data_id) == batch_size:
            data_ids = data_id
        else:
            raise ValueError(
                f"data_id length {len(data_id)} does not match batch size {batch_size}"
            )

        from vllm import SamplingParams

        # 一个输出 token 触发完整请求结束和 connector 发布；训练层清单仍根据 prompt 长度生成。
        sampling_params = SamplingParams(max_tokens=1, temperature=0)

        # Build packed_loss_mask_map for result assembly
        packed_loss_mask_map: dict[str, str | None] = {}
        if packed_loss_mask_list is not None:
            for i, did in enumerate(data_ids):
                if i < len(packed_loss_mask_list):
                    packed_loss_mask_map[did] = packed_loss_mask_list[i]

        outputs = self._engine.generate(prompts, sampling_params, use_tqdm=False)

        results = []
        for i, output in enumerate(outputs):
            seq_len = len(output.prompt_token_ids)
            did = data_ids[i]

            # 元数据由 connector 调度端预先生成并随 RequestOutput 回传；这里不直接访问 worker 缓存。
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is None:
                logger.error(
                    f"VllmEngine rank {self.rank}: No kv_transfer_params for data_id={did}. "
                    f"The MooncakeHiddenStatesConnector may not have stored this request."
                )
                continue

            mooncake_key = kv_params.get("mooncake_key", did)
            tensor_shapes = kv_params.get("tensor_shapes", {})
            tensor_dtypes = kv_params.get("tensor_dtypes", {})

            result: dict[str, Any] = {
                "mooncake_key": mooncake_key,
                "tensor_shapes": tensor_shapes,
                "tensor_dtypes": tensor_dtypes,
                "data_id": did,
                "seq_len": seq_len,
            }
            # 清单使训练端知道需要读取哪些层对象；即使单 PP 阶段也可沿用同一逐层协议。
            pp_layer_manifest = kv_params.get("pp_layer_manifest")
            if pp_layer_manifest is not None:
                result["metadata"] = {
                    "vllm_pp_complete": True,
                    "vllm_pp_layer_manifest": pp_layer_manifest,
                }

            packed_loss_mask = packed_loss_mask_map.get(did)
            if packed_loss_mask is not None:
                result["packed_loss_mask"] = packed_loss_mask

            # 顺带返回 token 的小型 CPU 列表，供消费者构建 mask 等操作，避免额外 GPU→CPU 同步。
            input_ids_from_kv = kv_params.get("input_ids_list")
            if input_ids_from_kv is not None:
                result["input_ids_list"] = input_ids_from_kv
            else:
                result["input_ids_list"] = list(output.prompt_token_ids)

            results.append(result)

        logger.debug(
            f"VllmEngine rank {self.rank}: generated {len(results)} mooncake results "
            f"for data_ids={data_ids}"
        )
        return results

    # 统一单样本 [1, S] 和 [S]，其他维度明确拒绝，避免列表转换后丢失样本边界。
    def _normalize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 2 and input_ids.shape[0] == 1:
            return input_ids.squeeze(0)
        if input_ids.dim() == 1:
            return input_ids
        raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")

    # 逐项过滤 None；字符串通过后端加载函数解析，已经加载好的媒体对象直接保留。
    @staticmethod
    def _resolve_media(items: list, fetch_fn) -> list:
        """Resolve URL strings to loaded objects via *fetch_fn*, drop None entries."""
        resolved = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, str):
                resolved.append(fetch_fn(item))
            else:
                resolved.append(item)
        return resolved

    # 将 images/videos 转为 vLLM 的 image/video 字段；单个媒体折成对象，多个保留列表，并兼容缺少视频加载工具的环境。
    @staticmethod
    def _to_vllm_multi_modal_data(mm_input: dict | None) -> dict | None:
        """Convert TorchSpec multimodal payload to vLLM ``multi_modal_data``.

        TorchSpec shape:  ``{"images": [...], "videos": [...]}``
        vLLM shape:       ``{"image": <PIL.Image|list>, "video": ...}``

        URL strings are resolved to PIL Images / video objects via
        ``vllm.multimodal.utils.fetch_image`` / ``fetch_video`` so that
        ``LLM.generate()`` receives the data types it expects.
        None entries (from incomplete media blocks) are filtered out.
        """
        if not mm_input:
            return None
        mm_data: dict = {}
        images = mm_input.get("images")
        if images:
            from vllm.multimodal.utils import fetch_image

            loaded = VllmEngine._resolve_media(images, fetch_image)
            if loaded:
                mm_data["image"] = loaded[0] if len(loaded) == 1 else loaded
        videos = mm_input.get("videos")
        if videos:
            try:
                from vllm.multimodal.utils import fetch_video

                loaded = VllmEngine._resolve_media(videos, fetch_video)
            except ImportError:
                loaded = [v for v in videos if v is not None]
            if loaded:
                mm_data["video"] = loaded[0] if len(loaded) == 1 else loaded
        return mm_data or None

    # 按批次位置组装文本或 token 字典并附加对应媒体；先校验媒体列表长度以防样本错配。
    def _build_prompts(
        self,
        formatted_prompts: list[str] | None,
        input_ids_list: list[torch.Tensor] | None,
        multimodal_inputs: list[dict | None] | None,
        batch_size: int,
    ) -> list:
        """Assemble per-request vLLM prompt dicts, attaching multimodal data when present."""
        if multimodal_inputs is not None and len(multimodal_inputs) != batch_size:
            raise ValueError(
                f"multimodal_inputs length {len(multimodal_inputs)} "
                f"does not match batch size {batch_size}"
            )

        prompts: list = []
        for i in range(batch_size):
            if formatted_prompts is not None:
                prompt_dict: dict = {"prompt": formatted_prompts[i]}
            else:
                prompt_dict = {
                    "prompt_token_ids": self._normalize_input_ids(input_ids_list[i]).tolist()
                }

            if multimodal_inputs is not None:
                mm_data = self._to_vllm_multi_modal_data(multimodal_inputs[i])
                if mm_data is not None:
                    prompt_dict["multi_modal_data"] = mm_data

            prompts.append(prompt_dict)
        return prompts

    # 前端 LLM 或 follower executor 任一存在即算就绪；follower 没有 LLM 是正常部署状态。
    def health_check(self, timeout: float = 5.0) -> bool:
        # A follower node owns a headless executor instead of an LLM front end,
        # so it is live without ``_engine``.
        return self._engine is not None or self._headless_executor is not None

    # 先关闭 follower，再按可用版本接口关闭前端；最终释放引用和 CUDA 空闲缓存，错误只记录日志。
    def shutdown(self) -> None:
        self._shutdown_headless_executor()

        if self._engine is not None:
            try:
                llm_engine = getattr(self._engine, "llm_engine", None)
                engine_core = getattr(llm_engine, "engine_core", None)

                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
                elif llm_engine is not None and hasattr(llm_engine, "shutdown"):
                    llm_engine.shutdown()
                elif hasattr(self._engine, "close"):
                    self._engine.close()
            except Exception as e:
                logger.warning(f"VllmEngine rank {self.rank}: Error during engine shutdown: {e}")
            finally:
                self._engine = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"VllmEngine rank {self.rank}: shutdown complete")

    # 关闭 follower 子进程并限时等待监视线程退出，避免无 LLM 的节点在清理时遗漏 GPU 占用。
    def _shutdown_headless_executor(self) -> None:
        """Tear down a follower node's executor and its worker processes.

        Followers never build an ``LLM``, so without this their
        ``MultiprocExecutor``, its monitor thread and every child worker
        process outlive the actor and keep holding GPU memory.
        """
        if self._headless_executor is None:
            return

        try:
            self._headless_executor.shutdown()
        except Exception as e:
            logger.warning(
                f"VllmEngine rank {self.rank}: Error during headless executor shutdown: {e}"
            )
        finally:
            self._headless_executor = None

        monitor_thread = getattr(self, "_headless_monitor_thread", None)
        if monitor_thread is not None:
            monitor_thread.join(timeout=self._HEADLESS_MONITOR_JOIN_TIMEOUT_S)
            if monitor_thread.is_alive():
                logger.warning(
                    f"VllmEngine rank {self.rank}: headless monitor thread did not exit "
                    f"within {self._HEADLESS_MONITOR_JOIN_TIMEOUT_S}s"
                )
            self._headless_monitor_thread = None

    # 返回前端初始化标志及模型维度；注意 initialized 只看 _engine，与兼容 follower 的 health_check 判据不同。
    def get_status(self) -> dict:
        return {
            "rank": self.rank,
            "initialized": self._engine is not None,
            "base_gpu_id": self.base_gpu_id,
            "hidden_size": self._hidden_size,
        }

    # 从目标模型配置的顶层或嵌套文本配置读取 H；缺失时失败，不能构造错误字节长度。
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
