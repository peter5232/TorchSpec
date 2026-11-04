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

# 配置 architecture 名称到具体 draft 类的路由；服务配置、训练构造和权重加载共享这个入口。
# EAGLE 名称不意味着只支持 Llama，映射同时包括 MLA 和块预测家族。
import json
import os
from typing import Union

from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import LlamaConfig, PretrainedConfig, modeling_utils
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from torchspec.models.draft.deepseek_eagle import Eagle3DeepseekV2ForCausalLM
from torchspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from torchspec.models.draft.dflash2 import DFlash2Config, DFlash2DraftModel
from torchspec.models.draft.dspark import (
    DSparkConfig,
    DSparkDraftModel,
    K3DSparkConfig,
    K3DSparkModel,
)
from torchspec.models.draft.llama3_eagle import LlamaForCausalLMEagle3
from torchspec.utils.logging import logger


class EagleLlamaConfig(LlamaConfig):
    """LlamaConfig whose drafts may size attention by an explicit head_dim."""

    # 跳过 Llama 专有的 hidden_size 整除 head 数校验，保留通用配置校验。
    # 显式 head_dim 可使注意力宽度独立于残差宽度，例如 24*256 不必等于 hidden_size。
    def validate_architecture(self):
        # Skip only LlamaConfig's hidden_size % num_attention_heads check, which explicit
        # head_dim drafts break (Qwen3.8: 24 heads x 256 over hidden_size 5120).
        PretrainedConfig.validate_architecture(self)


class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,
        EagleLlamaConfig: LlamaForCausalLMEagle3,
        DeepseekV3Config: Eagle3DeepseekV2ForCausalLM,
        DFlashConfig: DFlashDraftModel,
        DFlash2Config: DFlash2DraftModel,
        DSparkConfig: DSparkDraftModel,
        K3DSparkConfig: K3DSparkModel,
    }

    # 按配置的精确类型选择模型类，再按需转换浮点 dtype；不会自动加载预训练权重。
    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        _model_cls = cls._model_mapping[type(config)]
        model = _model_cls(config, **config_kwargs)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model

    # 临时过滤 embedding 未初始化提示，让基类完成权重加载；finally 必须恢复全局 logger 方法。
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        original_warn = modeling_utils.logger.warning

        # 只吞掉涉及 embed_tokens.weight 的初始化提示，其余 warning 原样转发。
        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        finally:
            modeling_utils.logger.warning = original_warn

        return model


class AutoDraftModelConfig:
    _config_mapping = {
        "LlamaForCausalLMEagle3": EagleLlamaConfig,
        "LlamaForCausalLMEagle4": EagleLlamaConfig,
        "Eagle3DeepseekV2ForCausalLM": DeepseekV3Config,
        "DFlashDraftModel": DFlashConfig,
        "DFlash2DraftModel": DFlash2Config,
        "Qwen3DSparkModel": DSparkConfig,
        "DSparkDraftModel": DSparkConfig,
        "K3DSparkModel": K3DSparkConfig,
    }

    # 复制字典后强制显式出现的 tie_word_embeddings=False，验证唯一 architecture 并选择配置类。
    # 未提供 draft_vocab_size 时补成 target 词表尺寸，避免把缺省误认为裁剪。
    @classmethod
    def from_dict(cls, config: dict):
        config = dict(config)

        if "tie_word_embeddings" in config:
            logger.info("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        if "draft_vocab_size" not in config or config["draft_vocab_size"] is None:
            config["draft_vocab_size"] = config.get("vocab_size", None)

        return cls._config_mapping[architecture].from_dict(config)

    # 从 JSON 读取配置，再走与字典入口完全相同的架构与词表校验。
    @classmethod
    def from_file(cls, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls.from_dict(config)
