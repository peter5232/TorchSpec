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

# 批优先训练张量与 Megatron 原生 MTP 的序列优先接口之间的适配。
# 冻结 embedding/head 不等于冻结 MTP：MTP 层仍通过隐藏态学习未来 token 的预测。
"""Batch-first PyTorch wrapper around a native Megatron MTP layer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SequenceFirstEmbedding(nn.Module):
    """Adapt an ``nn.Embedding`` to the keyword, sequence-first call Megatron makes."""

    # 保存已有 embedding 模块引用，避免重新分配或复制词表参数。
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    # 查表得 [B,S,H]，转成连续 [S,B,H] 供 Megatron；position_ids 不影响 token embedding。
    def forward(self, *, input_ids: torch.Tensor, position_ids=None) -> torch.Tensor:
        return self.embedding(input_ids).transpose(0, 1).contiguous()


class MegatronMTPModel(nn.Module):
    """One Megatron ``MultiTokenPredictionLayer`` with the target's frozen embedding and head.

    Returns MTP hidden states; the caller computes the loss with
    ``compute_logits`` and its own objective.
    """

    # 持有 MTP 层和 RoPE，冻结共享 embedding/head；loss 的选择留给训练调用方。
    def __init__(
        self,
        *,
        mtp_layer: nn.Module,
        embedding: nn.Embedding,
        lm_head: nn.Linear,
        rotary_embedding: nn.Module,
    ) -> None:
        super().__init__()
        self.mtp_layer = mtp_layer
        self.embedding = SequenceFirstEmbedding(embedding).requires_grad_(False)
        self.lm_head = lm_head.requires_grad_(False)
        self.rotary_embedding = rotary_embedding

    # 将 MTP 输出直接送冻结 LM head，保留对 hidden_states 的梯度，得到 [B,S,V]。
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    # 校验 token 与 hidden 的 [B,S] 对齐，转 [S,B,H] 后交给原生 MTP，再转回。
    # 原生层把 hidden[t] 与 token[t+1] 的 embedding 结合；最后一行卷回 token 必须由上游 loss mask 排除。
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map unpadded ``[B, S]`` tokens and ``[B, S, H]`` target states to ``[B, S, H]``.

        Row ``t`` of the output combines ``last_hidden_states[:, t]`` with the
        embedding of ``input_ids[:, t + 1]``; the last row sees a wrapped token.
        """

        if input_ids.shape != last_hidden_states.shape[:2]:
            raise ValueError("input_ids and last_hidden_states must share [B, S]")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(
                input_ids
            )
        hidden_states, _, _, _ = self.mtp_layer(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=last_hidden_states.transpose(0, 1).contiguous(),
            attention_mask=None,  # Megatron's causal mask
            rotary_pos_emb=self.rotary_embedding(input_ids.shape[1]),
            embedding=self.embedding,
        )
        return hidden_states.transpose(0, 1).contiguous()
