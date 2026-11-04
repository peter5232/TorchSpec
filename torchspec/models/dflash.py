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

# DFlash 训练包装：教师多层特征作为上下文，每个锚点生成 anchor+MASK 的并行预测块。
# 块间不可互看；标签采用同位置约定，块内第 0 槽是已知 token，损失只统计后面的预测槽。
"""DFlash training model: wraps the DFlash draft model with training-specific logic.

Handles anchor sampling, block-causal mask generation, noise input construction,
and cross-entropy loss with exponential decay weighting.

Matches SpecForge's OnlineDFlashModel (specforge/core/dflash.py).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchspec.models.ops.anchors import sample_anchor_positions
from torchspec.models.ops.flex_attention import compile_friendly_create_block_mask

_VALID_DFLASH_LOSS_OBJECTIVES = {"decay", "dpace"}


# 先用 (1-alpha)*confidence+alpha 平滑置信度，再算前缀连乘的反向累加，整个权重不反传。
# 例如平滑值 [0.8,0.5] 的前缀为 [0.8,0.4]，位置权重为 [1.2,0.4]，强调早期正确性。
def _dpace_position_weights(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute detached D-PACE weights from per-position draft confidences."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {alpha}")

    with torch.no_grad():
        # alpha=1 时每个平滑概率为 1，反向累计给早槽更大权重；alpha=0 时完全依据模型当前置信度。
        smoothed = (1.0 - alpha) * confidences.float() + alpha
        prefix_products = torch.cumprod(smoothed, dim=-1)
        weights = torch.flip(
            torch.cumsum(torch.flip(prefix_products, dims=[-1]), dim=-1),
            dims=[-1],
        )
        return weights.to(dtype=confidences.dtype)


# 返回 FlexAttention 谓词，KV 是 context 后接所有 draft 块，Q 只含 draft 块。
# 上下文严格早于 anchor；块内可双向或因果并可叠加滑窗，块间始终不可见。
def _create_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    is_causal: bool = False,
    sliding_window: int | None = None,
):
    """Create a mask_mod function for DFlash block-causal attention.

    KV: [Context (ctx_len tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos)
      2. Intra-block attention follows is_causal and sliding_window
      3. Different blocks are invisible to each other
      4. Invalid blocks (block_keep_mask=False) see nothing
    """
    num_anchors = anchor_positions.shape[1]

    # 整数除法求块号、取余求块内偏移，再用原始 anchor 坐标做滑窗判断。
    # 例如 context 长 8、块长 3，Q=4 属第二块，只能看 draft KV=11..13。
    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        anchor_pos = anchor_positions[b, q_block_id]

        is_context = kv_idx < ctx_len
        # 必须严格小于 anchor：教师 hidden[anchor] 已包含真实锚点，当前块通过自己的 anchor embedding 引入该 token。
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= ctx_len
        kv_block_id = (kv_idx - ctx_len) // block_size
        # 同块双向可见的未来输入是 MASK，而非未来真实 token embedding，因此不会直接泄露答案。
        mask_draft = is_draft & (q_block_id == kv_block_id)

        q_offset = q_idx % block_size
        kv_offset = (kv_idx - ctx_len) % block_size
        if is_causal:
            mask_draft = mask_draft & (kv_offset <= q_offset)
        if sliding_window is not None:
            query_position = anchor_pos + q_offset
            mask_context = mask_context & (query_position - kv_idx < sliding_window)
            mask_draft = mask_draft & (
                abs(query_position - (anchor_pos + kv_offset)) < sliding_window
            )

        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_draft) & is_valid_block

    dflash_mask_mod.__name__ = f"dflash_mask_A{num_anchors}_B{block_size}_C{ctx_len}"
    return dflash_mask_mod


class DFlashModel(nn.Module):
    """DFlash training wrapper.

    Wraps the DFlash draft model with training-specific logic:
      - Random anchor sampling with block_keep_mask
      - Block-causal attention mask via FlexAttention
      - Noise input construction (anchor + MASK)
      - Cross-entropy loss with configurable position weighting
      - Per-position loss_mask application
    """

    # 校验 decay/dpace 目标和 alpha 范围，保存锚点形状及 CE/L1 系数；参数层由 draft_model 提供。
    def __init__(
        self,
        draft_model,
        block_size: int = 16,
        num_anchors: int = 512,
        loss_objective: str = "decay",
        dpace_alpha: float = 0.5,
        loss_decay_gamma: float = 7.0,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
    ):
        super().__init__()
        loss_objective = loss_objective.lower()
        if loss_objective not in _VALID_DFLASH_LOSS_OBJECTIVES:
            valid = ", ".join(sorted(_VALID_DFLASH_LOSS_OBJECTIVES))
            raise ValueError(
                f"Unknown DFlash loss objective {loss_objective!r}; expected one of {valid}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dflash_dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.block_size = block_size
        self.num_anchors = num_anchors
        self.loss_objective = loss_objective
        self.dpace_alpha = dpace_alpha
        self.loss_decay_gamma = loss_decay_gamma
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)

    # 仅 L1 蒸馏启用时才要求教师末层状态；纯硬标签 CE 可省掉这份传输与存储。
    @property
    def uses_target_hidden_states(self) -> bool:
        return self.l1_loss_alpha > 0

    # 使用共享采样器生成固定 num_anchors 个槽，块长决定候选尾界，空余槽由 keep_mask 屏蔽。
    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return sample_anchor_positions(
            seq_len, loss_mask, self.num_anchors, self.block_size, device
        )

    # 上下文位置为 0..S-1；每个 draft 块的位置为 anchor+[0..K-1]。
    # 展平后的内存顺序不是 RoPE 位置，例如 anchors=[2,7]、K=3 对应 [2,3,4,7,8,9]。
    def _create_position_ids(
        self, anchor_positions: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create position IDs for context and draft tokens."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        draft_position_ids = anchor_positions.unsqueeze(-1) + offsets
        draft_position_ids = draft_position_ids.view(bsz, -1)

        return context_position_ids, draft_position_ids

    # 构建 [B,N*K] 的 MASK token，只把有效块起点替换成真实 anchor token，再查 embedding。
    # clamp 仅保证 gather 不越界；块是否合法仍由 keep_mask 决定，不能把钳制后的尾 token 当监督。
    def _create_noise_embed(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create noise embeddings: anchor token at block starts, MASK elsewhere.

        Matches SpecForge's OnlineDFlashModel._create_noise_embed().
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.draft_model.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.draft_model.mask_token_id, dtype=torch.long, device=device),
        )

        return self.draft_model.embed_tokens(noise_ids)

    # 基类使用空选项，即块内非因果、无滑窗；变体通过这个扩展点改变策略。
    def _block_mask_options(self) -> dict:
        return {}

    # 基类每层复用同一策略；保留 layer_id 接口让 DFlash2 能按层配置 full/SWA。
    def _block_mask_options_for_layer(self, layer_id: int) -> dict:
        """Return the attention-mask options for one draft layer.

        DFlash historically used one mask for every layer.  Keeping this
        method separate from ``_block_mask_options`` lets model variants use
        per-layer attention schedules without changing the public training
        wrapper API.
        """
        del layer_id
        return self._block_mask_options()

    # 优先使用 draft 自带输出头，否则用外部教师权重做 F.linear，输出 [B,N*K,V]。
    def _compute_logits(
        self,
        draft_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.draft_model, "lm_head"):
            return self.draft_model.lm_head(draft_hidden)
        return F.linear(draft_hidden, lm_head_weight)

    # 基类没有附加目标，返回同设备标量零和空指标；DFlash2 在此接 selector 的分子。
    def _extra_training_loss(
        self,
        draft_hidden: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        return logits.new_zeros(()), {}

    # 串联教师特征融合、锚点采样、噪声 embedding、位置和逐层 BlockMask，再执行 draft。
    # 相同 mask 策略在一次调用内共享；CUDA 才构建 mask，非 CUDA 的无 mask fallback 不是同等块隔离语义。
    def _draft_backbone(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor],
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Shared DFlash backbone (context features → anchor sampling → noise
        embedding → position ids → block-causal mask → draft model forward).

        Both ``DFlashModel.forward`` and ``DSparkModel.forward`` build the draft
        hidden states this exact way; only the label/loss tail differs. Keeping
        the attention/mask/anchor wiring here gives it a single source of truth.

        Returns:
            draft_hidden: [B, n_blocks*block_size, D] pre-loss draft hidden states
            anchor_positions: [B, n_blocks] sampled anchor positions
            block_keep_mask: [B, n_blocks] bool validity of each anchor slot
            n_blocks: number of anchor slots (== num_anchors)
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Extract context features from target hidden states
        # 阶段一：K 路教师 [B,S,Htarget] 融合成 [B,S,Hdraft]，同一份 context 供所有 draft decoder 层读取。
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)

        # 2. Sample anchor positions with validity mask
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.shape[1]

        # 3. Create noise embeddings (anchor token + MASK tokens)
        # 阶段二：N 个块一次排成 N*K 长的 draft 序列，真实 anchor token 提供已知前驱，其余输入是 MASK。
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        # 4. Create position IDs
        context_position_ids, draft_position_ids = self._create_position_ids(
            anchor_positions, seq_len
        )

        # 5. Create block-causal attention mask
        # 物理长度用于 mask 维度；RoPE 位置来自 anchor，不能把物理下标 q_idx 当成文本绝对位置。
        draft_len = n_blocks * self.block_size
        kv_len = seq_len + draft_len

        # CPU 路径没有构建块掩码，draft 的 fallback 无因果/块间隔离；CPU 前向不能作为此训练掩码的正确性证据。
        block_mask = None
        if device.type == "cuda":
            # A mixed full/SWA schedule needs a distinct BlockMask for each
            # attention policy.  Cache equal policies so the common uniform
            # schedule still creates exactly one mask.
            mask_cache = {}
            block_masks = []
            for layer_id in range(self.draft_model.num_layers):
                options = self._block_mask_options_for_layer(layer_id)
                # cache 在本次 forward 内复用策略；锚点每步变化，不能跨 batch 仅凭策略复用含旧锚点的 mask。
                cache_key = tuple(sorted(options.items()))
                if cache_key not in mask_cache:
                    mask_mod = _create_dflash_mask_mod(
                        anchor_positions=anchor_positions,
                        block_keep_mask=block_keep_mask,
                        ctx_len=seq_len,
                        block_size=self.block_size,
                        **options,
                    )
                    mask_cache[cache_key] = compile_friendly_create_block_mask(
                        mask_mod=mask_mod,
                        B=bsz,
                        H=None,
                        Q_LEN=draft_len,
                        KV_LEN=kv_len,
                        device=device,
                    )
                block_masks.append(mask_cache[cache_key])
            block_mask = block_masks[0] if len(mask_cache) == 1 else block_masks

        # 6. Draft model forward — pass embeddings directly
        # 阶段三：骨干返回 [B,N*K,H] post-final-norm 特征，并把锚点/keep_mask 交 loss 计算标签。
        # 此阶段不读取 ground-truth 未来标签 embedding，未来标签仅在下面的损失中 gather。
        draft_hidden = self.draft_model(
            draft_input_ids=None,
            context_feature=context_feature,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            block_mask=block_mask,
            noise_embedding=noise_embedding,
        )

        return draft_hidden, anchor_positions, block_keep_mask, n_blocks

    # 槽 k 的标签是 input_ids[anchor+k]，排除 k=0，边界/监督掩码与目标权重分别处理。
    # CE/L1 先逐 token 求值，再返回 numerator 和 detach 的 denominator，供 trainer 跨 rank/累积窗口归一化。
    # 准确率及逐位置 CE 使用未加 decay 的监督 mask，不能直接把它们当作加权优化目标。
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
        """
        Full DFlash training forward pass.

        Returns:
            loss: scalar training loss (objective-weighted)
            accuracy: scalar accuracy (binary mask, no decay)
            loss_per_position: [block_size] mean loss at each within-block position
                (index 0 is the anchor slot and always 0; indices 1..B-1 are the
                predicted tokens at 1..B-1 steps past the anchor)
            acc_per_position: [block_size] mean accuracy at each within-block position
            count_per_position: [block_size] valid label count at each within-block
                position before loss decay is applied
            loss_components: dict of extra per-component ``(numerator, denominator)``
                pairs for logging, pooled by the trainer just like ``loss_terms``
                (empty for the base DFlash objective; populated by subclasses).
            loss_terms: additive objective numerator and denominator. The trainer
                pools these over the full accumulation window and data-parallel group.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1-6. Shared backbone → draft hidden states + anchor bookkeeping.
        draft_hidden, anchor_positions, block_keep_mask, n_blocks = self._draft_backbone(
            input_ids, hidden_states_list, loss_mask
        )

        # 7. Compute logits via frozen LM head
        logits = self._compute_logits(draft_hidden, lm_head_weight)

        # 8. Compute labels and weight mask (SpecForge pattern)
        # Labels: same-position prediction (position k predicts token at anchor+k)
        # 例如 anchor=4、K=3，槽0/1/2标签是 x4/x5/x6；槽0已输入 x4，因此权重必须为零。
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, n_blocks, block_size]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        # 将原序列按每块的原始标签坐标取值，expand 只广播，不必复制 N 份 token 序列。
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )  # [B, n_blocks, block_size]

        # Weight mask: block validity × bounds × exclude anchor (pos 0) × loss_mask
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        # Gather original loss_mask at label positions
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
            2,
            safe_label_indices,
        )
        # 块有效、序列内、非 anchor、原始监督四项相乘；每项含义独立，去掉任何一项都会引入错误监督。
        weight_mask = weight_mask * original_loss_mask_gathered

        # Capture binary mask BEFORE applying objective weights. Accuracy measures
        # "did we predict correctly?" uniformly across positions, while weighting
        # only shapes gradient contribution. SpecForge uses no decay at all;
        # our objective weighting is an addition to the training signal, not the metric.
        # 保留未经过目标加权的评估范围；下面乘 decay/dpace 生成新 tensor，不原地修改这份 mask。
        binary_eval_mask = weight_mask.view(-1)

        # 9. Per-token loss: ce_loss_alpha*CE + l1_loss_alpha*L1.
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = target_ids.view(-1)
        # 把 B*N*K 合并为样本维，保留逐 token CE 供不同位置权重使用，不能先 reduction=mean。
        ce_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        loss_per_token = self.ce_loss_alpha * ce_per_token
        if self.uses_target_hidden_states:
            if last_hidden_states is None:
                raise ValueError(
                    "DFlash L1 distillation (l1_loss_alpha > 0) requires target "
                    "last_hidden_states; set inference.store_last_hidden_states=true in the "
                    "run config."
                )
            # 自回归教师 hidden[t] 的 head 预测 token[t+1]；蒸馏 token[x] 必须取 hidden[x-1]，不能取同位置 hidden。
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, n_blocks, block_size]
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden = torch.gather(last_hidden_states, 1, gather_idx)
            target_logits = F.linear(aligned_hidden, lm_head_weight).view(-1, vocab_size)
            # L1 在概率空间而不是 logits 空间比较；float32 softmax 减少大词表下小概率累计误差。
            target_probs = torch.softmax(target_logits.float(), dim=-1)
            draft_probs = torch.softmax(flat_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            loss_per_token = loss_per_token + self.l1_loss_alpha * l1_per_token

        loss_per_token_by_position = ce_per_token.view(bsz, n_blocks, self.block_size)

        # 优化权重从合法监督集合开始，再乘 decay 或 D-PACE；逐位置准确率不乘这些偏好权重。
        objective_weights = weight_mask
        if (
            self.loss_objective == "decay"
            and self.loss_decay_gamma is not None
            and self.loss_decay_gamma > 0
        ):
            # Loss decay: exp(-(k-1)/γ) so k=1 (1st prediction) gets weight 1.0
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            objective_weights = weight_mask * decay_weights
        elif self.loss_objective == "dpace":
            dpace_weights = torch.ones_like(weight_mask)
            if self.block_size > 1:
                with torch.no_grad():
                    # hard CE=-log q(correct)，因此 exp(-CE) 正好是正确 token 的概率，用它估计连续预测成功的权重。
                    target_confidences = torch.exp(-loss_per_token_by_position[..., 1:].float())
                    dpace_pred_weights = _dpace_position_weights(
                        target_confidences,
                        self.dpace_alpha,
                    ).to(dtype=weight_mask.dtype)
                dpace_weights[..., 1:] = dpace_pred_weights
            objective_weights = weight_mask * dpace_weights

        flat_weights = objective_weights.view(-1)
        loss_numerator = (loss_per_token * flat_weights).sum()
        # 分母是权重和而非有效 token 个数；累积窗口内先加分子/分母再相除，与平均 micro-batch loss 不等价。
        loss_denominator = flat_weights.sum()
        # 附加项必须返回可累加分子；DFlash2 selector 共用外层优化分母，不能先求 selector 局部均值再相加。
        extra_numerator, loss_components = self._extra_training_loss(
            draft_hidden=draft_hidden,
            logits=logits,
            target_ids=target_ids,
            objective_weights=objective_weights,
        )
        loss_numerator = loss_numerator + extra_numerator
        loss = loss_numerator / loss_denominator.clamp(min=1e-6)

        # 10. Accuracy (using binary mask without decay)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum().clamp(min=1e-6)
            accuracy = correct.sum().float() / actual_token_count

            # Per-position-within-block metrics (index 0 = anchor, masked out;
            # indices 1..block_size-1 correspond to 1..B-1 tokens past the anchor).
            # Matches Eagle3's per-TTT-position breakdown semantically.
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size)
            # 按块内槽位聚合 B 与 N，得到长度 K 的曲线；k=0 被排除，因此其计数和指标是零。
            count_per_position = binary_weights.sum(dim=(0, 1))
            count_per_pos = count_per_position.clamp(min=1.0)

            loss_per_position = (loss_per_token_by_position * binary_weights).sum(
                dim=(0, 1)
            ) / count_per_pos
            acc_per_position = (correct.view(bsz, n_blocks, self.block_size).float()).sum(
                dim=(0, 1)
            ) / count_per_pos

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
            (loss_numerator, loss_denominator.detach()),
        )
