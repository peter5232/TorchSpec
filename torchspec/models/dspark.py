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

# DSpark 复用 DFlash 特征/掩码骨干，但改为每槽预测下一个 token，并组合 Markov CE、L1 蒸馏与置信度 BCE。
# 这里的接受率是两分布重合质量，不是一次线上生成中观察到的接受/拒绝随机事件。
"""DSpark training model: DFlash training wrapper + Markov / L1 / confidence losses.

Reuses :class:`DFlashModel`'s anchor sampling, block-causal FlexAttention mask,
and MASK-token noise construction verbatim, then layers on the DSpark training
objective:

  - Markov-biased draft logits (teacher-forced previous token).
  - Cross-entropy against the ground-truth next tokens (hard labels).
  - L1 distribution distillation: ``|softmax(draft) - softmax(target)|`` where the
    target distribution is the frozen LM head applied to the *target's* final
    hidden state at the aligned position (requires ``last_hidden_states``).
  - Confidence head BCE against the empirical per-token accept rate.

Combined: ``ce_alpha*ce + l1_alpha*l1 + confidence_alpha*confidence``.

Loss formulation adapted from DeepSeek's DeepSpec
(deepspec/modeling/dspark/loss.py, MIT). Additive loss terms are returned to
the trainer so normalization covers every rank and gradient-accumulation step.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from torchspec.models.dflash import DFlashModel


class DSparkModel(DFlashModel):
    """DSpark training wrapper (DFlash backbone + Markov/L1/confidence heads)."""

    # 选用共享 DFlash 的 decay 路径，同时保存 CE/L1/confidence 三项权重；骨干构造保持一致。
    def __init__(
        self,
        draft_model,
        block_size: int = 7,
        num_anchors: int = 512,
        loss_decay_gamma: float = 4.0,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        confidence_head_alpha: float = 1.0,
    ):
        # Reuse DFlash anchor/mask/noise machinery. The "decay" objective drives
        # the per-within-block position weighting shared by all DSpark terms.
        super().__init__(
            draft_model=draft_model,
            block_size=block_size,
            num_anchors=num_anchors,
            loss_objective="decay",
            dpace_alpha=0.5,
            loss_decay_gamma=loss_decay_gamma,
        )
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)

    # L1 或启用的置信度目标都需要教师分布；即便 L1 系数为零，也可能仍需 last_hidden_states。
    @property
    def uses_target_hidden_states(self) -> bool:
        # The confidence head regresses onto an accept rate derived from the
        # verifier distribution, so the L1 weight alone does not cover it.
        return super().uses_target_hidden_states or (
            self.draft_model.confidence_head is not None and self.confidence_head_alpha > 0
        )

    # 所有 K 个槽都预测新 token，因此用 exp(-k/gamma)，k=0 权重为 1。
    # 这与 DFlash 跳过锚点槽后使用 exp(-(k-1)/gamma) 的索引不同。
    def _decay_weights(self, device: torch.device) -> torch.Tensor:
        """exp(-k/gamma) over within-block position k (DeepSpec convention).

        Every slot 0..B-1 is a real prediction here, so slot 0 (the first
        predicted token) gets weight 1.0 and later slots decay.
        """
        k = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            return torch.exp(-k.float() / self.loss_decay_gamma)
        return torch.ones_like(k, dtype=torch.float32)

    # 标签为 anchor+j+1，前驱为 anchor token 或前一个真实标签；这是 teacher forcing。
    # 有效 mask 经 cumprod 只保留连续监督前缀，L1 对齐教师 hidden[标签位置-1]。
    # alpha=1-0.5*L1 作为 detach 的 BCE 软标签，三个分子共享 decay 权重分母交给 trainer 累加。
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """DSpark training forward.

        Returns DFlashModel.forward's metrics plus a dict of detached per-component
        ``(numerator, denominator)`` pairs (ce_loss / l1_loss / confidence_loss) and
        the additive loss terms. ``loss`` is the combined ce+l1+confidence
        objective; the per-position metrics are cross-entropy based (acceptance
        proxy). DSparkTrainer unpacks the extra elements; the rest of DFlash's
        aggregation is reused unchanged.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # ---- Shared DFlash backbone (steps 1-6) → draft hidden states ----
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids, hidden_states_list, loss_mask
        )
        # 把平铺的 draft 序列恢复块结构，后续 Markov、confidence 和逐槽统计都按 [B,N,K,...] 对齐。
        hidden_4d = draft_hidden.view(bsz, n_blocks, self.block_size, -1)

        base_logits = F.linear(draft_hidden, lm_head_weight)
        base_logits_4d = base_logits.view(bsz, n_blocks, self.block_size, -1)
        vocab_size = base_logits_4d.size(-1)

        # ---- Labels + eval mask (DeepSpec convention) ----
        # Slot j predicts the token at anchor+j+1 (the real anchor token seeds
        # slot 0). All block_size slots are supervised — there is no masked
        # anchor slot, unlike DFlash.
        # 例如 anchor=4、K=3，此处预测 x5/x6/x7，三个槽均有效；与 DFlash 的 x4/x5/x6 约定不同。
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, nb, bs]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )  # [B, nb, bs]

        # eval mask = contiguous supervised prefix per block (DeepSpec
        # build_eval_mask): block kept, label in-bounds, target token supervised,
        # then cumprod so a gap truncates the rest of the block.
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_bool = block_keep_mask.unsqueeze(-1) & valid_label_mask & (target_loss_mask > 0.5)
        # 从第一个不可监督位置起整条后续前缀清零；例如 [True,False,True] 变 [True,False,False]。
        eval_bool = eval_bool.to(torch.int32).cumprod(dim=-1).bool()
        eval_mask = eval_bool.float()  # [B, nb, bs]

        decay_weight_mask = eval_mask * self._decay_weights(device)
        local_den = decay_weight_mask.sum()

        # ---- Markov-biased draft logits ----
        # prev token for slot j is the ground-truth token immediately before the
        # one slot j predicts: slot 0's prev is the real anchor token, slot j's
        # is target_ids[j-1]. Matches DeepSpec prev_token_ids.
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)  # [B, nb]
        # 与上例对应前驱是 [x4,x5,x6]，训练取真实 token；不能把它误解成 draft argmax 的在线自回归链。
        prev_token_ids = torch.cat([anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)
        logits_4d = base_logits_4d
        # 基础 head 分布加上真实前驱的低秩 bigram 偏置；Markov 不改 hidden，只改词表 logits。
        if self.draft_model.markov_head is not None:
            logits_4d = self.draft_model.markov_head.apply_block_logits(
                base_logits_4d, token_ids=prev_token_ids
            )

        # ---- Cross entropy (hard labels) ----
        flat_logits = logits_4d.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(
            bsz, n_blocks, self.block_size
        )
        ce_num = (ce_per_token * decay_weight_mask).sum()

        # ---- L1 distribution distillation + accept rate ----
        l1_num = base_logits.new_zeros((), dtype=torch.float32)
        accept_rate = None
        # 即使 L1 权重关闭，启用 confidence 也要构造 target_probs，因为它的软标签依赖接受率。
        if self.uses_target_hidden_states:
            if last_hidden_states is None:
                raise ValueError(
                    "DSpark L1/confidence losses require target last_hidden_states; set "
                    "inference.store_last_hidden_states=true in the run config."
                )
            # target distribution for the token at label_indices = target LM head
            # applied to the target hidden one position earlier (anchor+j).
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, nb, bs]
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            # 教师 gather 的时间点比预测标签早一格；输入最后维 Htarget 与 lm_head_weight 的宽度匹配。
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
            aligned_target_logits = F.linear(aligned_hidden, lm_head_weight).view(
                bsz, n_blocks, self.block_size, vocab_size
            )
            draft_probs = torch.softmax(logits_4d.float(), dim=-1)
            target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)  # [B, nb, bs]
            if self.l1_loss_alpha > 0:
                l1_num = (l1_per_token * decay_weight_mask).sum()
            # 两个完整归一化分布满足 sum(min(p,q))=1-TV=1-L1/2，故此处得到精确分布重合质量。
            accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)

        # ---- Confidence head BCE ----
        conf_num = base_logits.new_zeros((), dtype=torch.float32)
        if self.draft_model.confidence_head is not None and self.confidence_head_alpha > 0:
            # confidence 可同时看 draft 特征和前驱低秩 embedding，使它知道当前 token 转移条件。
            if self.draft_model.confidence_head_with_markov:
                prev_emb = self.draft_model.markov_head.get_prev_embeddings(prev_token_ids).to(
                    hidden_4d.dtype
                )
                conf_features = torch.cat([hidden_4d, prev_emb], dim=-1)
            else:
                conf_features = hidden_4d
            confidence_pred = self.draft_model.confidence_head(conf_features).float()
            # detach 阻止置信度分支通过“改变自己的标签”降低 BCE；分布匹配仍由 CE/L1 项驱动。
            conf_bce = (
                F.binary_cross_entropy_with_logits(
                    confidence_pred, accept_rate.detach(), reduction="none"
                )
                * decay_weight_mask
            )
            conf_num = conf_bce.sum()

        # The trainer pools these additive terms over the entire optimizer window
        # before compensating for FSDP's mean gradient reduction.
        # 三项各自累加后按配置系数合成；local_den 仅由监督与 decay 决定，不因某项系数为零而改定义。
        loss_numerator = (
            self.ce_loss_alpha * ce_num
            + self.l1_loss_alpha * l1_num
            + self.confidence_head_alpha * conf_num
        )
        loss = loss_numerator / local_den.clamp(min=1e-6)

        # Per-component terms for logging only — lets you watch L1 fall while the
        # greedy-CE proxy plateaus. Reported as numerator/denominator rather than a
        # local mean so the trainer can pool them over the same window as the
        # objective; averaging per-micro-batch means would let these curves drift
        # away from the loss actually optimized.
        # 日志分量保留独立分子但共享分母，跨 rank/累积窗口汇总后可与优化的加权组合相互核对。
        component_denominator = local_den.detach()
        loss_components = {
            "ce_loss": (ce_num.detach(), component_denominator),
            "l1_loss": (l1_num.detach(), component_denominator),
            "confidence_loss": (conf_num.detach(), component_denominator),
        }

        # ---- Metrics (cross-entropy based; all block_size slots are productive) ----
        with torch.no_grad():
            flat_binary = eval_mask.reshape(-1)
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (flat_binary > 0.5)
            accuracy = correct.sum().float() / flat_binary.sum().clamp(min=1e-6)

            count_per_position = eval_mask.sum(dim=(0, 1))
            count_pp = count_per_position.clamp(min=1.0)
            loss_per_position = (ce_per_token * eval_mask).sum(dim=(0, 1)) / count_pp
            acc_per_position = (
                correct.view(bsz, n_blocks, self.block_size).float().sum(dim=(0, 1)) / count_pp
            )

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            (loss_numerator, local_den.detach()),
        )
