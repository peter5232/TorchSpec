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

# 教师数据生成协议与 Hugging Face 实现。用少量 hook 捕获指定层而不返回全部层激活，节约显存。
# auxiliary hidden 与 final-norm 后 hidden 分开传递：前者训练特征融合，后者用于生成教师词表分布。
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from torchspec.utils.distributed import get_tp_device_mesh, get_tp_group


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    target: Optional[torch.Tensor] = None
    last_hidden_states: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    # Pre-copied CPU tensor for loss mask computation, avoids GPU→CPU round trip
    # when mooncake uses the TCP/host-buffer path (data already passes through CPU).
    input_ids_cpu: Optional[torch.Tensor] = None

    # 只导出非 None 的可选张量，加上必须的 hidden/input_ids，形成传输或存储使用的扁平字段字典。
    # input_ids_cpu 是可复用 CPU 副本，用于 loss mask 扫描避免 GPU→CPU 再搬一次。
    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        tensors = {
            "hidden_states": self.hidden_states,
            "input_ids": self.input_ids,
        }
        if self.loss_mask is not None:
            tensors["loss_mask"] = self.loss_mask
        if self.target is not None:
            tensors["target"] = self.target
        if self.last_hidden_states is not None:
            tensors["last_hidden_states"] = self.last_hidden_states
        if self.input_ids_cpu is not None:
            tensors["input_ids_cpu"] = self.input_ids_cpu
        return tensors


class Eagle3TargetModel(ABC):
    """
    Abstract interface for the target model backend used to generate Eagle3 training data.
    """

    # 初始化待配置的辅助层列表，后续设置后才能知道要捕获哪几路教师特征。
    def __init__(self):
        self.aux_hidden_states_layers = None

    # 抽象后端构造契约：接受模型位置、dtype、device 与缓存参数，具体后端负责加载策略。
    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Eagle3TargetModel":
        """
        Initialize the target model backend from a pretrained model path.
        """

    # 抽象数据生成契约：输入 token/attention/loss mask，输出隐藏态与监督所需元数据。
    @abstractmethod
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Generate the eagle3 data from the target model.
        """

    # 未指定时选 [1,L//2-1,L-4] 三路教师层；显式列表支持 DFlash 等更多路特征。
    # 拒绝空列表，层号表示实际 decoder block 索引，不是 HF hidden_states 中含 embedding 的索引。
    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """Set the layers to capture aux hidden states from the target model.

        Generalized to support any number of layers: Eagle3 uses 3, DFlash uses 5.
        When called with None, defaults to 3 Eagle3-style layers for backward compat.
        """
        if aux_hidden_states_layers is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {self.model.config} does not have num_hidden_layers"
                )
            aux_hidden_states_layers = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        if not self.aux_hidden_states_layers:
            raise ValueError("aux_hidden_states_layers must be a non-empty list")


class HFTargetModel(Eagle3TargetModel):
    # 包装已构造的 HF 模型并继承辅助层配置字段，不重新加载或复制模型参数。
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    # TP>1 时传 auto tp_plan 和指定 device_mesh，单卡则使用 device_map，再交 HF 完成加载。
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "HFTargetModel":
        """
        Initialize the HuggingFace target model backend from a pretrained model path.
        """
        tp_group = get_tp_group()
        tp_size = tp_group.size() if tp_group is not None else 1

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_tp_device_mesh(),
            }
        else:
            device_kwargs = {
                "device_map": device or "auto",
            }

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **device_kwargs,
            **kwargs,
        )
        return cls(target_model)

    # 按常见模型包装层定位 decoder blocks，找不到时显式报错以防 hook 注册在错误模块。
    def _get_transformer_layers(self):
        """
        Helper to find the module list containing the transformer layers.
        Adapts to common architectures (Llama, Qwen, Mistral, OPT, etc.)
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "layers"):
            return self.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        else:
            raise ValueError(
                "Could not locate transformer layers in the model architecture to register hooks."
            )

    # 定位 norm/ln_f 等 final norm 模块，捕获其输出才能直接接教师 LM head。
    def _get_final_norm(self) -> nn.Module:
        """
        Helper to find the final norm layer (post-transformer, pre-lm_head).
        Its output is the last hidden states before projection to vocab logits.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        elif hasattr(self.model, "norm"):
            return self.model.norm
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        else:
            raise ValueError(
                "Could not locate final norm layer in the model architecture to register hook."
            )

    # 在 no_grad 下注册指定 block 和 final norm hook，关闭 cache/全层返回后执行一次 teacher forward。
    # finally 移除 hook，验证捕获数量，再按配置顺序拼成 [B,S,K*H]；last_hidden_states 是 post-norm。
    @torch.no_grad()
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Optimized HF backend:
        Instead of returning all hidden states (memory heavy), we use forward hooks
        to capture only the specific layers required by Eagle3.
        """
        captured_states = {}
        last_hidden = {}
        handles = []

        # 为每个 layer_idx 创建独立闭包，避免循环变量晚绑定导致多层覆盖同一字典项。
        def get_hook(layer_idx):
            # 兼容 block 的 Tensor/tuple 返回，将首个隐藏态保存到对应层号，不拷贝所有层激活。
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured_states[layer_idx] = hidden

            return hook

        # 保存 final norm 的输出供教师 head 使用，避免重复 norm 或误用最后 block 的 pre-norm 状态。
        def norm_hook(module, input, output):
            last_hidden["value"] = output

        layers = self._get_transformer_layers()
        target_indices = self.aux_hidden_states_layers

        for idx in target_indices:
            if 0 <= idx < len(layers):
                handles.append(layers[idx].register_forward_hook(get_hook(idx)))
            else:
                raise ValueError(
                    f"Layer index {idx} out of bounds for model with {len(layers)} layers."
                )

        final_norm = self._get_final_norm()
        handles.append(final_norm.register_forward_hook(norm_hook))

        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                output_router_logits=False,
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        expected = len(target_indices)
        if len(captured_states) != expected:
            raise RuntimeError(
                f"Expected to capture {expected} layers, but captured {len(captured_states)}"
            )
        assert "value" in last_hidden, "Failed to capture last hidden states from final norm layer"

        device = input_ids.device
        hidden_states = torch.cat(
            [captured_states[idx].to(device) for idx in target_indices], dim=-1
        )

        loss_mask = loss_mask[..., None].to(device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            last_hidden_states=last_hidden["value"].to(device),
            loss_mask=loss_mask,
            input_ids=input_ids,
        )
