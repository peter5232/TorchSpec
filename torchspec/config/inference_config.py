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

# 中文导读：推理后端的配置边界。TP 是把一个模型的张量计算分到多卡，PP 是把层分成多个阶段；
# 它们与训练侧 DP（多个副本处理不同数据）不是同一个维度。extra_args 保留引擎特有选项，
# 公共字段则用于资源分配、特征采集和调度。字段最终如何传递需跟到 inference/factory.py。

"""Inference configuration classes.

SGLangConfig exposes only the fields that TorchSpec actually references.
Power users can pass arbitrary extra kwargs to sgl.Engine via ``extra_args``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from torchspec.config.mooncake_config import MooncakeConfig


# SGLang 的并行规模、静态显存比例、启动通信与多模态设置；未列出的引擎参数经 extra_args 传递。
@dataclass
class SGLangConfig:
    """Essential SGLang engine configuration.

    Only fields that TorchSpec explicitly uses are listed here.
    Any additional sgl.Engine kwargs can be supplied via ``extra_args``
    and will be forwarded as-is.
    """

    # Parallelism
    tp_size: int = 8
    pp_size: int = 1
    nnodes: int = 1

    # Memory
    mem_fraction_static: float = 0.8

    # Observability (read by TorchSpec's wandb integration)
    enable_metrics: bool = False

    # Multimodal
    enable_multimodal: bool = False

    # Networking (port is auto-selected by SglEngine via get_free_port)
    dist_init_addr: Optional[str] = None
    dist_timeout: int = 60
    init_timeout: int = 300

    # Passthrough: forwarded as-is to sgl.Engine.
    # Use this for any sgl.Engine kwarg that TorchSpec doesn't need to
    # inspect (e.g. quantization, context_length, attention_backend,
    # log_level, ...).
    extra_args: Dict[str, Any] = field(default_factory=dict)


# vLLM 特征采集配置；静态显存比例为 None 时引擎可根据 connector 额外开销推导。
# 不能直接把 TP/PP 当作独立副本数，它们共同运行一份目标模型。
@dataclass
class VllmConfig:
    """Essential vLLM engine configuration.

    Only fields that TorchSpec explicitly uses are listed here.
    Any additional vLLM engine kwargs can be supplied via ``extra_args``
    and will be forwarded as-is.

    Uses vLLM's ``extract_hidden_states`` speculative method with a
    ``MooncakeHiddenStatesConnector`` KV Connector for hidden states
    retrieval (requires vLLM >= 0.18.0).
    """

    # Parallelism
    tp_size: int = 8
    pp_size: int = 1
    nnodes: int = 1

    # Memory (None = auto-compute from connector overhead; see VllmEngine)
    mem_fraction_static: Optional[float] = None

    # Observability
    enable_metrics: bool = False

    # Multimodal
    enable_multimodal: bool = False

    # Networking (port is auto-selected by VllmEngine)
    dist_init_addr: Optional[str] = None
    dist_timeout: int = 60
    init_timeout: int = 300

    # Passthrough: forwarded as-is to vLLM LLM.
    # Use this for any vLLM kwarg that TorchSpec doesn't need to
    # inspect (e.g. quantization, max_model_len, trust_remote_code, ...).
    extra_args: Dict[str, Any] = field(default_factory=dict)


# TensorRT-LLM SaveHiddenStates 路径的配置，目前仅接通单节点 TP。
# mem_fraction_static 对应 KV cache 的可用显存比例，与其他引擎同名参数的分配对象并非完全相同。
@dataclass
class TrtllmConfig:
    """Essential TensorRT-LLM engine configuration.

    Wraps TRT-LLM's PyTorch-backend ``LLM`` in ``SaveHiddenStates`` mode; the
    TorchSpec patch redirects captured hidden states to Mooncake. Only fields
    TorchSpec explicitly uses are listed; any other ``LLM`` kwarg can be passed
    via ``extra_args``.

    Single-node tensor parallelism only (nnodes must be 1); multi-node TP is
    not yet wired up.
    """

    # Parallelism (TP degree is derived from inference_num_gpus_per_engine).
    tp_size: int = 8
    pp_size: int = 1
    nnodes: int = 1

    # KV-cache memory fraction (TRT-LLM's KvCacheConfig.free_gpu_memory_fraction).
    mem_fraction_static: Optional[float] = 0.8

    init_timeout: int = 600

    # Passthrough: forwarded as-is to the TRT-LLM LLM constructor
    # (e.g. attn_backend, max_num_tokens, dtype, ...).
    extra_args: Dict[str, Any] = field(default_factory=dict)


# 当前集成面向单节点 eager prefill 特征采集；TP 的实际资源数由每引擎 GPU 配置派生。
@dataclass
class TokenSpeedConfig:
    """TokenSpeed offline hidden-state inference configuration.

    The first integration targets single-node eager prefill. Tensor parallelism
    is derived from ``inference_num_gpus_per_engine``.
    """

    tp_size: int = 1
    nnodes: int = 1
    mem_fraction_static: float = 0.8
    init_timeout: int = 600
    extra_args: Dict[str, Any] = field(default_factory=dict)


# 回放模式只需磁盘目录和读取引擎数；这些引擎搬运已有特征，不运行目标模型。
@dataclass
class OfflineTrainingConfig:
    """Configuration for training from materialized target outputs."""

    data_path: Optional[str] = None
    num_engines: int = 1


# 跨后端的采集和缓冲配置；辅助层用于草稿输入，最后一层 hidden state 可用于重建 teacher logits。
@dataclass
class InferenceConfig:
    aux_hidden_states_layers: Optional[list] = None
    inference_batch_size: int = 1
    inference_buffer_threshold: int = 32
    inference_engine_type: str = "hf"
    inference_fetch_batch: int = 1
    inference_num_gpus: Optional[int] = None
    inference_num_gpus_per_engine: int = 1
    inference_num_gpus_per_node: int = 8
    inference_node_ips: Optional[list[str]] = None
    inference_node_selectors: Optional[list[dict[str, str]]] = None
    last_hidden_states_prenorm: Optional[bool] = None
    max_sample_pool_size: int = 0
    store_last_hidden_states: bool = True
    offline: OfflineTrainingConfig = field(default_factory=OfflineTrainingConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    trtllm: TrtllmConfig = field(default_factory=TrtllmConfig)
    tokenspeed: TokenSpeedConfig = field(default_factory=TokenSpeedConfig)

    # 明确配置优先；否则仅 vLLM 默认视为 final norm 前的特征。
    # 如果把 pre-norm 特征直接乘 lm_head，会改变 teacher 分布；因此这不是无关紧要的元信息。
    def resolve_last_hidden_states_prenorm(self) -> bool:
        """Whether last_hidden_states from the engine are pre-norm.

        vLLM's extract_hidden_states connector can only capture raw layer
        outputs (pre-norm), while sglang and hf provide post-norm outputs.
        """
        if self.last_hidden_states_prenorm is not None:
            return self.last_hidden_states_prenorm
        return self.inference_engine_type == "vllm"


# 直接调用 Transformers 模型所需的紧凑配置；包含模型、精度、层索引和可选存储连接。
@dataclass
class HFInferenceConfig:
    """Configuration for HFRunner."""

    model_path: str
    max_seq_length: int = 8192
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    aux_hidden_states_layers: Optional[list[int]] = None
    mooncake_config: Optional[MooncakeConfig] = None


# 提前拒绝非正批大小；零批次会导致调度看似运行却永远不产出训练样本。
def _validate_inference_batch_config(config: DictConfig) -> None:
    if config.inference.inference_batch_size <= 0:
        raise ValueError(
            f"inference_batch_size must be > 0 (got {config.inference.inference_batch_size}); "
            f"0 causes inference-pool starvation (silent dispatch spin) and vLLM rejects "
            f"max_num_seqs=0 late after init"
        )
