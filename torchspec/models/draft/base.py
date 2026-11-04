# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# EAGLE draft 的参数层契约、因果掩码、输出头和检查点单张量读取。
# 训练循环只依赖 embedding/project/backbone/logits 等接口；norm 后反馈与 logits 归一化是两项独立约定。
import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.modeling_utils import PreTrainedModel


# 仅 EAGLE4 架构启用滑窗，优先 layer_types，再 use_sliding_window，最后看 window 是否配置。
# 不能把从 Qwen 继承来的 EAGLE3 sliding_window 字段误当成训练实际启用窗口。
def draft_sliding_window(config) -> Optional[int]:
    """Context window W of an EAGLE-4 sliding-window draft, or None.

    At every TTT depth a query sees the W context keys ending at its own position, plus its
    own chain. Enablement follows SGLang's EAGLE-4 loader: ``layer_types`` wins, then
    ``use_sliding_window``, then a bare ``sliding_window``. Other architectures keep full
    attention: Qwen-derived EAGLE3 configs carry a ``sliding_window`` they never used.
    """
    if "LlamaForCausalLMEagle4" not in (getattr(config, "architectures", None) or ()):
        return None
    window = getattr(config, "sliding_window", None)
    layer_types = getattr(config, "layer_types", None)
    enabled = getattr(config, "use_sliding_window", None)
    if layer_types is not None:
        enabled = "sliding_attention" in layer_types
    elif enabled is None:
        enabled = window is not None
    if enabled and not window:
        raise ValueError(f"Sliding window is enabled but sliding_window={window!r}")
    return window if enabled else None


# 生成 [B,1,T,T+past] 的加性 mask：允许为 0，禁止为 dtype 最小值。
# 窗口仅作用当前 T×T 三角区域；拼接的 past 列全部为 0，要按此实现理解历史缓存可见性。
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    window: Optional[int] = None,
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    if window is not None:
        # Keys window or more positions behind the query fall out of the window.
        mask.masked_fill_(
            mask_cond <= (mask_cond - window).view(mask.size(-1), 1), torch.finfo(dtype).min
        )
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# 把 [B,S] 的 1=有效 padding mask 扩为 [B,1,T,S]，再转成 0/极小值的加性屏蔽。
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# 将下三角因果约束与 padding 约束相加；T=1 时无需单独构造当前 token 的三角阵。
def prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class LowRankHead(nn.Module):
    """Factorized output projection of rank ``hidden_size // divisor``.

    ``down.weight`` is ``[rank, hidden]`` and ``up.weight`` is ``[vocab, rank]``; the dense
    product is never materialized.
    """

    # 建立 H→rank→V 两个无偏置线性层，rank=H//divisor；拒绝 divisor 太小或 rank=0。
    def __init__(self, hidden_size: int, vocab_size: int, divisor: int) -> None:
        super().__init__()
        if divisor < 2:
            raise ValueError(f"lm_head_rank_divisor must be at least 2, got {divisor}")
        rank = hidden_size // divisor
        if rank < 1:
            raise ValueError(
                f"lm_head_rank_divisor={divisor} leaves no rank for hidden_size={hidden_size}"
            )
        self.rank = rank
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)

    # 顺序做 down 再 up，得到词表 logits，避免显式形成 [V,H] 的稠密乘积权重。
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states))


# 未配置 rank divisor 就建稠密头；低秩头只支持完整词表，避免与裁剪行映射混用。
def build_lm_head(config, hidden_size: int, vocab_size: int, target_vocab_size: int) -> nn.Module:
    """Dense head, or a ``LowRankHead`` when the config sets ``lm_head_rank_divisor``."""
    divisor = getattr(config, "lm_head_rank_divisor", None)
    if not divisor:
        return nn.Linear(hidden_size, vocab_size, bias=False)
    if vocab_size != target_vocab_size:
        raise ValueError(
            f"lm_head_rank_divisor is not supported with a pruned draft vocabulary "
            f"(draft_vocab_size={vocab_size} < vocab_size={target_vocab_size})."
        )
    return LowRankHead(hidden_size, vocab_size, divisor)


class Eagle3DraftModel(PreTrainedModel, ABC):
    """
    This is the base class for the Eagle3 draft model implementation. The child class needs to implement
    the abstract methods to support training with TTT.
    """

    # Drafts tie nothing and never call post_init(), where Transformers 5.x would set this.
    all_tied_weights_keys = {}

    # 从配置显式层数或 eagle_aux 层列表推导教师特征路数，默认三路，再解析 EAGLE4 滑窗。
    def __init__(self, config):
        super().__init__(config)

        self.num_aux_hidden_states = getattr(config, "num_aux_hidden_states", None)
        if self.num_aux_hidden_states is None:
            eagle_config = getattr(config, "eagle_config", None) or {}
            layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
            self.num_aux_hidden_states = len(layer_ids) if layer_ids else 3
        self.sliding_window = draft_sliding_window(config)

    # 抽象契约：把 target 词表 token 索引映射到 draft 输入 embedding，保持 batch/序列维。
    @abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed the input ids.
        """

    # 抽象契约：将拼接的 K 路教师特征压缩到 draft hidden_size，供第一个展开深度使用。
    @abstractmethod
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """

    # 抽象契约：由 draft 隐藏态生成词表 logits；具体实现负责与训练 fused loss 一致的 norm。
    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits of the draft model.
        """

    # 以当前 draft 的滑窗策略建因果 mask，再合并 padding mask，输出可直接加到 attention score 的张量。
    def prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_length: int,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """
        Prepare the attention mask of the draft model.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                (batch_size, seq_length),
                hidden_states.dtype,
                device=hidden_states.device,
                past_key_values_length=past_key_values_length,
                window=self.sliding_window,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=seq_length
            ).to(hidden_states.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    # 选择下一轮反馈状态：norm_output 优先，否则可用 post_mlp_layernorm，均未启用则反馈原始残差。
    # 输出 logits 始终另做 final norm，不能根据反馈是否归一化推断 loss 是否需要 norm。
    def next_depth_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hidden state the next TTT depth reads; logits always come from norm(hidden_states).

        ``norm_output`` feeds back the normed state, a ``post_mlp_layernorm`` feeds back its
        own norm of it, otherwise the raw layer output is fed back.
        """
        if self.norm_output:
            return self.norm(hidden_states)
        post_norm = getattr(self.midlayer, "post_mlp_layernorm", None)
        return hidden_states if post_norm is None else post_norm(hidden_states)

    # 抽象契约：接收 token embedding、上轮状态、mask/位置和缓存，返回层输出及更新后的 K/V。
    # 缓存承载训练展开链，各实现存储布局不同，调用者不要把它当普通在线 token cache 解读。
    @abstractmethod
    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_keys: Optional[torch.Tensor] = None,
        cache_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        The backbone of the draft model.
        """

    # 检查输出头是否 LowRankHead，用于 fused loss 选择一段还是两段线性投影。
    @property
    def is_lm_head_factorized(self) -> bool:
        return isinstance(self.lm_head, LowRankHead)

    # 返回 final norm 权重、第一段 head 权重与 eps；低秩时第一段是 [rank,H]，不是 [V,H]。
    def get_lm_head_params(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return (norm_weight, projection_weight, norm_eps) for fused loss computation.

        For a factorized head the projection is the down weight; pair it with
        ``get_lm_head_up_weight()``. Override for a different normalization scheme.
        """
        weight = self.lm_head.down.weight if self.is_lm_head_factorized else self.lm_head.weight
        return self.norm.weight, weight, self.norm.variance_epsilon

    # 低秩头返回第二段 [V,rank] 权重，稠密头返回 None，保持 loss 投影与模型一致。
    def get_lm_head_up_weight(self) -> Optional[torch.Tensor]:
        """Second factor ``[vocab, rank]`` of a factorized head, or None when it is dense."""
        return self.lm_head.up.weight if self.is_lm_head_factorized else None

    # 检查存在 t2d 且并非全 True；初始化时的全 True 占位缓冲不代表已经加载了裁剪映射。
    @property
    def has_vocab_pruning(self) -> bool:
        return hasattr(self, "t2d") and not self.t2d.all()

    # 拒绝覆盖已有不同裁剪映射：head 第 j 行和被选 target token 的对应关系随权重一起固定。
    # 只换 t2d/d2t 不换 head 行会静默错训；校验后使用 copy_ 保持注册 buffer 身份。
    @torch.no_grad()
    def set_vocab_buffers(self, d2t: torch.Tensor, t2d: torch.Tensor) -> None:
        """Set the t2d/d2t vocab mapping buffers directly from tensors.

        Refuses to replace a mapping that arrived with loaded weights. ``lm_head`` row j holds the
        j-th target token selected by the mapping, and the target side is pruned with
        ``target_lm_head_weight[t2d]`` in the same order, so swapping in a mapping that selects a
        different token set repoints every row without touching the rows themselves.
        """
        assert hasattr(self, "t2d") and hasattr(self, "d2t"), (
            "t2d and d2t buffers are not found in the draft model"
        )
        if self.has_vocab_pruning and not (
            torch.equal(self.t2d, t2d.to(self.t2d.device))
            and torch.equal(self.d2t, d2t.to(self.d2t.device))
        ):
            raise ValueError(
                "The draft was seeded from weights whose vocabulary mapping selects a different "
                "token set than the one computed for this run. The loaded lm_head rows are "
                "ordered by the old mapping, so replacing the buffers alone would silently "
                "mistrain every row. Either set model.keep_initial_vocab_mapping=true to train "
                "against the mapping the weights were built with, or drop "
                "model.initial_draft_model_path / training.load_path and train the head from "
                "scratch for the new mapping."
            )
        self.t2d.copy_(t2d)
        self.d2t.copy_(d2t)

    # 关闭 token embedding 权重梯度；训练仍可更新后续投影与注意力层。
    def freeze_embedding(self) -> None:
        """
        Freeze the embeddings of the draft model so that they are not updated during training.
        """
        self.embed_tokens.weight.requires_grad = False

    # 遍历输出头全部参数关闭梯度，因此同时覆盖稠密与低秩两段头。
    def freeze_lm_head(self) -> None:
        """
        Freeze the lm_head of the draft model so that it is not updated during training.
        """
        for param in self.lm_head.parameters():
            param.requires_grad = False

    # 按检查点键只读取 embedding 张量，copy_ 到已存在参数，不更换 Parameter 对象。
    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """
        Load the embedding of the draft model.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
            repository ID or a local directory path containing the model files.
        """
        self.embed_tokens.weight.copy_(load_tensor_from_pretrained(model_path, embedding_key))

    # 从教师稠密头初始化 draft，要求完整词表、非低秩且形状相等。
    # 裁剪映射在构造后才建立，当前阶段无法安全选择教师行，因此直接拒绝而不是猜测。
    @torch.no_grad()
    def load_lm_head(self, model_path: str, lm_head_key: str = "lm_head.weight") -> None:
        """
        Seed the draft lm_head from the target model's lm_head.

        Only valid without vocabulary pruning. A pruned head's rows are ordered by ``t2d``, which
        is not populated until after model construction (see ``set_vocab_buffers``), so the target
        rows cannot be selected in draft order at this point.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
            repository ID or a local directory path containing the model files.
        """
        if hasattr(self, "t2d"):
            raise ValueError(
                "load_lm_head does not support a pruned draft vocabulary: the head's rows are "
                "ordered by a t2d mapping that is only computed after model init. Seed the head "
                "from model.initial_draft_model_path or training.load_path instead."
            )
        if self.is_lm_head_factorized:
            raise ValueError(
                "load_lm_head does not support a factorized lm_head (lm_head_rank_divisor); "
                "the target head is dense and has no factors to seed."
            )
        weight = load_tensor_from_pretrained(model_path, lm_head_key)
        if weight.shape != self.lm_head.weight.shape:
            raise ValueError(
                f"Target lm_head shape {tuple(weight.shape)} does not match the draft lm_head "
                f"{tuple(self.lm_head.weight.shape)}. Seeding the draft head requires the draft "
                "and target to share hidden size and vocabulary."
            )
        self.lm_head.weight.copy_(weight)


# 支持 Hub 缓存、本地单文件和 index.json 分片；只定位包含指定 key 的分片。
# 多个索引被视为歧义并报错，safetensors 可直接读单张量，bin 则需载入 state_dict。
def load_tensor_from_pretrained(model_path: str, key: str) -> torch.Tensor:
    """Read a single tensor by key from an HF checkpoint directory or hub repository id."""
    if not os.path.exists(model_path):
        # model_path is a huggingface repository, so locate its local cache first
        return load_tensor_from_pretrained(snapshot_download(repo_id=model_path), key)

    # check if there is file ending with index.json
    glob_path = os.path.join(model_path, "*.index.json")
    index_json_path = glob.glob(glob_path)

    if len(index_json_path) == 0:
        # No index.json found, look for single model file
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                return f.get_tensor(key)

        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            return state_dict[key]

        raise FileNotFoundError(
            f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
        )
    if len(index_json_path) > 1:
        raise FileNotFoundError(f"Multiple index.json files found in {model_path}")
    index_json_path = index_json_path[0]

    with open(index_json_path, "r") as f:
        index_json = json.load(f)
    ckpt_file = index_json["weight_map"][key]

    if ckpt_file.endswith(".safetensors"):
        with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
            return f.get_tensor(key)
    state_dict = torch.load(os.path.join(model_path, ckpt_file))
    return state_dict[key]
