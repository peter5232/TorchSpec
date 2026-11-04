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

# DFlash2 在 DFlash 训练尾部加入候选选择器监督，并为每一层保留 full/sliding attention 策略。
# 卷积与候选打分参数层见 draft/dflash2.py；此处负责 logits 标定及可累加损失。
"""DFlash2 training wrapper."""

import math

import torch
import torch.nn.functional as F

from torchspec.models.dflash import DFlashModel


class DFlash2Model(DFlashModel):
    # 验证 selector 系数和逐层 attention 类型，滑窗层必须有正窗口。
    # 若未显式指定 is_causal，sliding 层因果而 full 层双向；每层选项与 draft 层数一一对应。
    def __init__(self, *args, selector_loss_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        selector_loss_alpha = float(selector_loss_alpha)
        if not math.isfinite(selector_loss_alpha) or selector_loss_alpha <= 0:
            raise ValueError(
                f"dflash2_selector_loss_alpha must be positive, got {selector_loss_alpha}"
            )
        self.selector_loss_alpha = selector_loss_alpha

        config = self.draft_model.config
        layer_types = list(getattr(config, "layer_types", []) or [])
        if layer_types and len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                "DFlash2 layer_types must contain one entry per draft layer, got "
                f"{len(layer_types)} for {config.num_hidden_layers} layers"
            )
        # 未列出 layer_types 时按所有层 full attention 处理；显式列表长度必须与层数一致。
        attention_types = set(layer_types or ["full_attention"])
        if not attention_types <= {"full_attention", "sliding_attention"}:
            raise ValueError(f"Unsupported DFlash2 layer types: {sorted(attention_types)}")
        explicit_causality = getattr(config, "is_causal", None)
        configured_window = getattr(config, "sliding_window", None)
        # 只要有一层使用滑窗，就要求正窗口；不能把 None 解释为随意的服务端默认值。
        if "sliding_attention" in attention_types:
            if configured_window is None:
                raise ValueError(
                    "DFlash2 sliding_attention layers require an explicit positive sliding_window"
                )
            configured_window = int(configured_window)
            if configured_window < 1:
                raise ValueError(f"sliding_window must be positive, got {configured_window}")

        # 统一展开成逐层策略表，后面骨干按 layer_id 取对应 mask；相同策略可共享同一个本次 forward 的 mask。
        normalized_layer_types = layer_types or ["full_attention"] * config.num_hidden_layers
        self.layer_block_mask_options = []
        for layer_type in normalized_layer_types:
            uses_sliding_window = layer_type == "sliding_attention"
            self.layer_block_mask_options.append(
                {
                    "is_causal": (
                        uses_sliding_window
                        if explicit_causality is None
                        else bool(explicit_causality)
                    ),
                    "sliding_window": configured_window if uses_sliding_window else None,
                }
            )

        # Retain the uniform-policy attributes for callers that inspect them.
        # 这两个属性仅保留统一策略查询的兼容性，不代表混合层模型所有层都采用首层策略。
        self.attention_is_causal = self.layer_block_mask_options[0]["is_causal"]
        self.sliding_window = self.layer_block_mask_options[0]["sliding_window"]

    # 返回首层策略，兼容只查询统一策略的调用者；混合层实际使用逐层接口。
    def _block_mask_options(self) -> dict:
        return self.layer_block_mask_options[0]

    # 按 layer_id 取对应策略，避免 full 和 SWA 层共用错误的可见性。
    def _block_mask_options_for_layer(self, layer_id: int) -> dict:
        return self.layer_block_mask_options[layer_id]

    # 基础投影后先乘 output_multiplier，再按 softcap*tanh(logits/softcap) 限幅。
    # 这改变分布温度和极值，训练输出须与导出的推理配置保持一致。
    def _compute_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        logits = super()._compute_logits(draft_hidden, lm_head_weight)
        logits = logits * self.draft_model.config.output_multiplier
        softcap = self.draft_model.config.final_logit_softcapping
        if softcap is not None and softcap > 0:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    # 从第 1 槽起，用真实前驱 token 为候选打分，并强制把正确后继纳入 top-k 候选。
    # 监督权重遇到第一个无效槽就通过 cumprod 截断；返回 selector CE 的加权分子，不在这里单独求平均。
    def _extra_training_loss(
        self,
        draft_hidden: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        batch, num_blocks, block_size = target_ids.shape
        hidden = draft_hidden.reshape(batch, num_blocks, block_size, -1)[..., 1:, :]
        unary_logits = logits.reshape(batch, num_blocks, block_size, -1)[..., 1:, :]
        predecessor_ids = target_ids[..., :-1]
        successor_ids = target_ids[..., 1:]

        scores, candidate_ids = self.draft_model.candidate_selector.score_candidates(
            hidden,
            unary_logits,
            predecessor_ids,
            training_successor_ids=successor_ids,
        )
        matches = candidate_ids == successor_ids.unsqueeze(-1)
        eligible_weights = objective_weights[..., 1:]
        # 例如监督权重 [1,0,0.5] 变成 [1,0,0]：中间断开后不再训练后续链候选选择。
        eligible_weights = eligible_weights * (eligible_weights > 0).cumprod(dim=-1)
        target_indices = matches.to(torch.int64).argmax(dim=-1)
        selector_ce = F.cross_entropy(
            scores.reshape(-1, scores.shape[-1]),
            target_indices.reshape(-1),
            reduction="none",
        ).reshape_as(eligible_weights)
        selector_num = (selector_ce * eligible_weights).sum()
        selector_den = eligible_weights.sum().detach()
        return self.selector_loss_alpha * selector_num, {
            "selector_loss": (selector_num.detach(), selector_den),
        }
