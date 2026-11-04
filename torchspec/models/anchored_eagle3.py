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

# EAGLE3 的锚点训练：单层结构允许全序列只投影上下文 K/V，昂贵的查询、MLP 和 loss 只在 N 个锚点运行。
# 这是训练位置抽样，既非在线验收算法，也不是缩短原序列的 RoPE 坐标。
"""Anchored Eagle3: run the TTT unroll at N sampled positions instead of all of them.

The draft is a single layer, so its depth-0 keys and values are a pointwise projection of
the layer input and can be built for the whole sequence without any attention. Only the
queries are gathered, which is what makes every depth cost O(N) rather than O(S).
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from torchspec.models.draft.deepseek_eagle import _apply_rotary_pos_emb_interleaved
from torchspec.models.draft.llama3_eagle import (
    LlamaMutiRotaryEmbedding,
    apply_rotary_pos_emb,
    fuse_layer_input,
)
from torchspec.models.eagle3 import Eagle3Model, LazyTarget, PrecomputedTarget
from torchspec.models.ops.anchors import sample_anchor_positions


# 构造锚点可见性闭包：上下文可见到自身位置，后续每个深度块只可见自己的链槽。
# KV 布局为 S 个上下文位置，之后每个深度追加 N 个锚点，不能让两个锚点串链。
def _anchored_allows(
    anchor_positions: torch.Tensor,
    keep_mask: torch.Tensor,
    ctx_len: int,
    num_anchors: int,
):
    """Whether anchor ``q_idx`` may attend to ``kv_idx``.

    KV is ``[context (ctx_len) | chain depth 1 | chain depth 2 | ...]`` with ``num_anchors``
    slots per chain block. An anchor sees the context up to and including its own position,
    plus its own slot in every chain block written so far. Anchors never see each other.
    """

    # 由 kv_idx 相对 S 的余数识别链槽，再与 keep_mask 相交，屏蔽占位锚点。
    def allows(b, q_idx, kv_idx):
        anchor = anchor_positions[b, q_idx]
        context = (kv_idx < ctx_len) & (kv_idx <= anchor)
        chain = (kv_idx >= ctx_len) & (((kv_idx - ctx_len) % num_anchors) == q_idx)
        return (context | chain) & keep_mask[b, q_idx]

    return allows


# 广播构造 [B,1,N,KV] 布尔 SDPA 掩码；窗口只收窄上下文，不删除自己的展开链。
# 例如 S=8、N=2，锚点槽 1 可见追加位置 9、11、13，不能看槽 0 的 8、10、12。
def anchored_bool_mask(
    anchor_positions: torch.Tensor,
    keep_mask: torch.Tensor,
    ctx_len: int,
    num_anchors: int,
    kv_len: int,
    window: Optional[int] = None,
) -> torch.Tensor:
    """The predicate as a [B, 1, N, KV] boolean mask for scaled_dot_product_attention.

    Two broadcasts. Anchored attention is small enough that a BlockMask costs more to build
    than the density it removes: at S=4096 with 4 depths the build alone is 2.1ms against
    1.2ms for this whole path.
    """
    kv = torch.arange(kv_len, device=anchor_positions.device)
    slot = torch.arange(num_anchors, device=anchor_positions.device).unsqueeze(-1)
    context = (kv < ctx_len) & (kv <= anchor_positions.unsqueeze(-1))
    if window is not None:
        context &= kv > anchor_positions.unsqueeze(-1) - window
    chain = (kv >= ctx_len) & (((kv - ctx_len) % num_anchors) == slot)
    return ((context | chain) & keep_mask.unsqueeze(-1)).unsqueeze(1)


# 沿序列维收集锚点：二维 token/mask 返回 [B,N]，三维隐藏态返回 [B,N,D]。
def _gather(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather along the sequence dim: [B, S(, D)] by [B, N] -> [B, N(, D)]."""
    if source.dim() == 2:
        return torch.gather(source, 1, index)
    return torch.gather(source, 1, index.unsqueeze(-1).expand(-1, -1, source.shape[-1]))


# 将线性投影最后一维拆为 heads×head_dim，再转置成注意力需要的 [B,H,L,D]。
def _heads(projected: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    bsz, length, _ = projected.shape
    return projected.view(bsz, length, num_heads, head_dim).transpose(1, 2)


# 只执行 MLA 的 KV 压缩/展开，跳过昂贵而无用的 Q 投影。
# 返回每头 k_nope、value，以及跨头共享且尚未旋转的单头 k_rope。
def _mla_kv(attn, layer_input: torch.Tensor):
    """MLA's K/V halves alone: (k_nope, k_rope_raw, value), skipping the query projection."""
    bsz, length, _ = layer_input.shape
    compressed, k_rope = torch.split(
        attn.kv_a_proj_with_mqa(layer_input), [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
    )
    kv = attn.kv_b_proj(attn.kv_a_layernorm(compressed))
    kv = kv.view(bsz, length, attn.num_heads, attn.qk_nope_head_dim + attn.v_head_dim)
    k_nope, value = torch.split(kv, [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1)
    return k_nope.transpose(1, 2), value.transpose(1, 2), k_rope.unsqueeze(1)


# 把目标序列收集到锚点行；预计算分布同时收集 coverage，延迟目标复用同一 LM head。
# 后续 loss 用 idx=0 读取这些已对齐行，避免再次叠加 depth 偏移。
def _gather_target(
    target: Union[PrecomputedTarget, LazyTarget], positions: torch.Tensor
) -> Union[PrecomputedTarget, LazyTarget]:
    """Restrict a target to the anchor positions for one depth.

    Gathering N rows is cheaper than the dense path's [:, idx : idx + seq_len] slice, so the
    loss reads the same either way with idx=0 and seq_length=num_anchors.
    """
    if isinstance(target, PrecomputedTarget):
        # coverage_padded travels with the gathered rows: the LK losses read the true
        # target mass as target_p_padded * coverage_padded, and dropping it here would
        # silently fall back to all-ones and overstate alpha under vocab pruning.
        coverage = target.coverage_padded
        return PrecomputedTarget(
            target_p_padded=_gather(target.target_p_padded, positions),
            coverage_padded=None if coverage is None else _gather(coverage, positions),
        )
    return LazyTarget(
        hidden_states_padded=_gather(target.hidden_states_padded, positions),
        lm_head_weight=target.lm_head_weight,
    )


class AnchoredEagle3Model(Eagle3Model):
    """Eagle3 whose TTT depths are evaluated at sampled anchors rather than every position."""

    # 复用 EAGLE3 loss 配置，记录锚点预算和间距；检测 MLA 投影结构。
    # 明确拒绝 MRoPE：当前收集查询的旋转只有一维位置，不能悄悄丢掉视觉坐标。
    def __init__(
        self,
        draft_model,
        *args,
        num_anchors: int = 512,
        anchor_max_gap: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(draft_model, *args, **kwargs)
        self.is_mla = hasattr(draft_model.midlayer.self_attn, "kv_a_proj_with_mqa")
        if isinstance(draft_model.midlayer.self_attn.rotary_emb, LlamaMutiRotaryEmbedding):
            raise ValueError(
                "Anchored Eagle3 does not support MRoPE drafts (rope_scaling.rope_type="
                "'mrope'). The dense path rotates through apply_multimodal_rotary_pos_emb "
                "with the configured mrope_section, while the gathered-query path applies "
                "standard rotary and would ignore the multidimensional position ids."
            )
        self.num_anchors = num_anchors
        self.anchor_max_gap = anchor_max_gap

    # 统一返回分头后的 Q/K/V：普通注意力直接投影，MLA 仅旋转 rope 子维再与 nope 拼接。
    # query=False 为全上下文准备 KV，返回的占位 Q 不参与 attention；MLA 的 V 维度可不同于 Q/K。
    def _qkv(self, attn, layer_input, positions, rope, query: bool = True):
        """(query, key, value) in heads for one set of positions, rotated at ``positions``.

        Both attention shapes are pointwise in the layer input, which is what lets the
        context K/V be built for the whole sequence without running attention. MLA reaches
        its keys through the compressed kv_a/kv_b pair rather than k_proj/v_proj, rotates
        only the rope-side dims, and carries its own softmax scale.
        """
        if self.is_mla:
            if query:
                q, k_nope, k_rope, value = attn._project_qkv(layer_input)
                q_nope, q_rope = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
                q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
                    q_rope, k_rope, rope[0], rope[1], positions
                )
                q = torch.cat([q_nope, q_rope], dim=-1)
            else:
                # _project_qkv would also run q_a_proj/q_b_proj, which is most of the
                # projection cost and is thrown away for a context that needs only K/V.
                k_nope, value, k_rope = _mla_kv(attn, layer_input)
                _, k_rope = _apply_rotary_pos_emb_interleaved(
                    k_rope, k_rope, rope[0], rope[1], positions
                )
                q = k_rope
            k_rope = k_rope.expand(-1, attn.num_heads, -1, -1)
            return q, torch.cat([k_nope, k_rope], dim=-1), value

        key = _heads(attn.k_proj(layer_input), attn.num_key_value_heads, attn.head_dim)
        value = _heads(attn.v_proj(layer_input), attn.num_key_value_heads, attn.head_dim)
        q = _heads(attn.q_proj(layer_input), attn.num_heads, attn.head_dim) if query else key
        q, key = apply_rotary_pos_emb(q, key, rope[0], rope[1], positions)
        return q, key, value

    # 先对完整上下文做逐位置 KV 投影，再对 [B,N,D] 锚点状态展开 length 轮。
    # depth=0 的键已在上下文，不能重复追加；以后每轮只追加 N 个链键。
    # 损失和反馈仍走基类契约，以 effective_mask 排除裁剪词表无法表达的教师首选 token。
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Union[PrecomputedTarget, LazyTarget],
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        draft = self.draft_model
        layer = draft.midlayer
        attn = layer.self_attn
        bsz, seq_len = input_ids.shape
        device, num_anchors = input_ids.device, self.num_anchors
        norm_weight, lm_head_weight, norm_eps = draft.get_lm_head_params()
        lm_head_up = draft.get_lm_head_up_weight()

        # compute_target_p_padded marks positions whose full-vocab argmax falls outside the
        # pruned draft vocabulary; the dense path masks on that instead of loss_mask, and
        # training on those positions would fit a renormalized target the draft cannot emit.
        effective_mask = loss_mask
        if isinstance(target, PrecomputedTarget) and target.position_mask is not None:
            effective_mask = target.position_mask

        # 抽样发生在有效监督行上，固定 N 使算子尺寸稳定；N 大于可用位置时多余槽仍占形状但不计损失。
        anchors, keep_mask = sample_anchor_positions(
            seq_len, effective_mask, num_anchors, self.length, device, self.anchor_max_gap
        )

        input_ids = input_ids.clamp(min=0, max=draft.target_vocab_size - 1)
        projected = draft.project_hidden_states(hidden_states)
        embeddings = draft.embed_input_ids(input_ids).to(projected.dtype)

        # One rotary table covering every position an anchor can reach. Building it per
        # call from the query length would size it to num_anchors, not the sequence.
        # 位置表需要覆盖原始序列 S 加展开深度，而不是查询个数 N；锚点少不代表坐标也小。
        cos, sin = attn.rotary_emb(projected, seq_len=seq_len + self.length)
        rope = (cos.to(device), sin.to(device))

        # Context K/V. One layer means these are pointwise in the layer input, so the whole
        # sequence is a couple of linears -- no attention, no MLP, no cross-position term.
        context_positions = torch.arange(seq_len, device=device).expand(bsz, -1)
        # 只有单层 draft 才可这样预构建：K/V 是融合输入的逐位置投影，不依赖该层 attention 输出。
        # 如果直接推广到多层，上层上下文 K/V 就需要先运行下层 attention，不能只做 linears。
        context, _ = fuse_layer_input(layer, embeddings, projected)
        _, key_cache, value_cache = self._qkv(attn, context, context_positions, rope, query=False)

        current = _gather(projected, anchors)

        plosses, vlosses, acces, acc_counts, alphas = [], [], [], [], []
        # 采样器选定锚点后，整条链一直使用同一个槽；位置随 depth 前进，链身份不变。
        for depth in range(self.length):
            positions = (anchors + depth).clamp(max=seq_len - 1)
            fused, residual = fuse_layer_input(layer, _gather(embeddings, positions), current)

            query, key, value = self._qkv(attn, fused, positions, rope)

            # Depth 0's fused input is context[:, anchors], so its keys are already in the
            # context block; appending them would double-count each anchor's own key.
            # 第0轮已有的 context[:,anchor] 与当前融合输入完全相同，重复追加会重复计算该键的 softmax 质量。
            if depth:
                key_cache = torch.cat([key_cache, key], dim=2)
                value_cache = torch.cat([value_cache, value], dim=2)

            # 一次 SDPA 计算 N 个查询，mask 的最后维仍包含完整上下文 S 和已追加的 N*depth 链键。
            attn_out = F.scaled_dot_product_attention(
                query,
                key_cache,
                value_cache,
                attn_mask=anchored_bool_mask(
                    anchors,
                    keep_mask,
                    seq_len,
                    num_anchors,
                    key_cache.shape[2],
                    draft.sliding_window,
                ),
                enable_gqa=not self.is_mla,
                scale=attn.softmax_scale if self.is_mla else None,
            )
            attn_out = attn_out.transpose(1, 2).reshape(bsz, num_anchors, -1)

            # attention 之后的残差与 MLP 都只在 N 个锚点行运行，这是减少计算的主要来源。
            hidden = residual + attn.o_proj(attn_out)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

            # 位置 clamp 只防越界，keep_mask 决定槽是否真实存在；effective_mask 还排除不可表达的裁剪目标。
            step_mask = _gather(effective_mask, positions) * keep_mask
            step_target = _gather_target(target, positions)
            loss_sum, correct, count, alpha_sum = self._calculate_loss(
                hidden_states=hidden,
                target=step_target,
                mask=step_mask,
                idx=0,
                seq_length=num_anchors,
                norm_weight=norm_weight,
                lm_head_weight=lm_head_weight,
                norm_eps=norm_eps,
                lm_head_up=lm_head_up,
            )
            denominator = count.clamp_min(1.0)
            plosses.append(loss_sum / denominator)
            vlosses.append((loss_sum / denominator).detach())
            acces.append((correct / denominator).detach())
            acc_counts.append(count.detach().float())
            alphas.append((alpha_sum / denominator).detach())

            # 下一轮反馈预测状态，保留计算图；不是 detach 后把每个深度当成独立监督任务。
            current = draft.next_depth_hidden(hidden)

        return plosses, vlosses, acces, acc_counts, alphas
