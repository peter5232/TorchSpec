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

# 固定预算锚点采样器，供 EAGLE3 anchored 与 DFlash 共用。
# 采样发生在监督位置序列上，固定槽数稳定编译形状，keep_mask 负责标出占位槽。
"""Anchor sampling shared by the anchored draft training paths."""

from typing import Optional, Tuple

import torch

from torchspec.utils.logging import logger


# 仅保留自身和下一位置都监督的锚点，尾界同时考虑 block_size 与至少一个后继 token。
# 返回候选总数、布尔有效性、每样本有效数和以 S+1 替代无效位置的索引，便于排序。
def _candidates(seq_len: int, loss_mask: torch.Tensor, block_size: int, device: torch.device):
    """Positions usable as anchors, as (num_candidates, valid, valid_counts, masked_indices).

    An anchor is only usable if its own position and the position right after it are both
    supervised: the block's first prediction target is ``anchor + 1``, so an isolated
    supervised token yields no gradient.
    """
    bsz = loss_mask.shape[0]
    # 对 S=8,K=3，最大候选是位置 5，候选索引为 0..5；另一个上界 S-1 排除无后继的尾 token。
    # 这里选的是块起点，不是在序列中任取 K 个位置；块内各深度目标由模型另行构造。
    max_anchor = max(seq_len - block_size, 0)
    num_candidates = min(max_anchor + 1, seq_len - 1)

    # 只检查 a 和 a+1 都监督，不保证后面整块都监督。例 mask=[0,1,1,0,1,1,1,0]，
    # S=8,K=3 时候选为 a=1,4,5；a=1 后面会遇到断点，后续深度仍需模型自己的目标 mask。
    valid = (loss_mask[:, :num_candidates] > 0.5) & (loss_mask[:, 1 : num_candidates + 1] > 0.5)
    indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
    return num_candidates, valid, valid.sum(dim=1), torch.where(valid, indices, seq_len + 1)


# 序列太短时返回 [B,N] 全零锚点和全 False mask；仍保留固定形状，后续损失自然为零。
def _empty(bsz: int, num_anchors: int, seq_len: int, block_size: int, device: torch.device):
    logger.warning(
        f"Sequence too short for anchor sampling (seq_len={seq_len}, "
        f"block_size={block_size}). Returning dummy anchors so loss is zero."
    )
    return (
        torch.zeros(bsz, num_anchors, dtype=torch.long, device=device),
        torch.zeros(bsz, num_anchors, dtype=torch.bool, device=device),
    )


# 把有效位置排序，用随机相位与受 max_gap 限制的步长在有效位置环上抽样。
# 不足 N 个有效位置时只保留前 valid_count 槽；相位覆盖全环以避免只偏爱序列中间。
# 返回槽顺序可能因取模回绕，并非总是数值递增；gap 数的是监督位置，不是原始 token 距离。
def sample_anchor_positions(
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    block_size: int,
    device: torch.device,
    max_gap: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample anchor positions per sample; returns (anchors, keep_mask).

    Always returns exactly ``num_anchors`` anchor slots so that
    ``Q_LEN = num_anchors * block_size`` is constant across steps, preventing
    FlexAttention recompilation from shape changes. Samples with fewer valid positions
    leave the excess slots masked off.

    ``max_gap`` is how many supervised positions may sit *between* neighbouring anchors,
    so ``max_gap=0`` puts them side by side (5, 6, 7, ...). It matters because anchor ``a``
    supervises token ``a + d`` at unroll depth ``d``: a token is seen at depth ``d`` only if
    the anchor ``d`` positions behind it was also picked. Anchors with a gap of ``g`` leave
    each token ``block_size / (g + 1)`` of its depths, so once ``g`` reaches ``block_size``
    every token is pinned to a single depth.

        max_gap=0     neighbours adjacent; every token in the covered span is supervised
                      at all ``block_size`` depths, as in the dense unroll
        max_gap=k     every (k+1)-th supervised position, over a (k+1)x wider span
        max_gap=None  defaults to ``block_size``: one depth per token

    Gaps count supervised positions, not raw tokens, so slots are never spent on
    unsupervised regions and anchors stay token-adjacent inside a supervised span.

    Args:
        seq_len: sequence length
        loss_mask: [B, seq_len] — 1 for valid positions, 0 for padding
        device: torch device
        max_gap: supervised positions permitted between neighbours; None means block_size

    Returns:
        anchors: [B, num_anchors] — sampled anchor positions (sorted)
        keep_mask: [B, num_anchors] — True for valid sampled anchors
    """
    bsz = loss_mask.shape[0]
    # 当前实现把 S<=K 一律视为短序列，连 S==K 也走 dummy 分支；这是实际判断的边界。
    # dummy 的位置 0 只是合法 gather 地址，keep=False 才表达“不应产生监督”，二者必须一起传。
    if max(seq_len - block_size, 0) == 0:
        return _empty(bsz, num_anchors, seq_len, block_size, device)

    if max_gap is None:
        max_gap = block_size

    num_candidates, _, valid_counts, masked_indices = _candidates(
        seq_len, loss_mask, block_size, device
    )
    # Supervised positions in order; sentinels sort to the back and are masked off below.
    # 无效候选替换成 S+1，所以排序后自动靠末尾；后续 rank 只在有效计数范围内取值。
    ordered = masked_indices.sort(dim=1).values

    # Spread the budget over the whole sample when it reaches, otherwise pack as tightly
    # as max_gap allows. The window wraps and its phase is drawn over every candidate, so
    # each one is picked with probability num_anchors / valid_count exactly as uniform
    # sampling would. Drawing the phase over the leftover slack instead would pick the
    # middle of a sample far more often than its ends.
    # 一条样本有效位置若是 [1,4,5]，以下 rank 的 0/1/2 指这个列表的下标，而非原 token ID。
    # valid_count=0 时用 1 只为避免模零；最终 keep_mask 全 False，不凭空添加有效锚点。
    counts = valid_counts.unsqueeze(1).clamp(min=1)
    # 预算足够时步长为1；预算较小时尽量铺开，但不能超过 max_gap+1，从而控制每个 token 的深度覆盖。
    stride = (valid_counts // num_anchors).clamp(min=1, max=max_gap + 1).unsqueeze(1)
    # 随机相位从整个有效环取值，保证序列首尾候选都可被抽中，避免只在中间小窗口采样的偏置。
    offset = (torch.rand(bsz, 1, device=device) * counts).long()

    slot = torch.arange(num_anchors, device=device).unsqueeze(0)
    # num_anchors * stride <= valid_count by construction, so the wrap never collides.
    # 手算：有 6 个候选、N=3、stride=2、offset=5，则 rank=[5,1,3]。
    # 所以原 docstring 的 sorted 不适用于回绕后的槽顺序；原 token 位置要再通过 ordered.gather 获得。
    # 若候选原位置为 [1,2,6,7,8,9]，上述采样得到 [9,2,7]；rank 间隔不等于 token 间隔。
    rank = (offset + slot * stride) % counts
    # 有效数<N 时 rank 会取模重复，但这些重复槽被 keep_mask 禁用；固定形状与真实监督数量由两张张量分工。
    keep_mask = slot < valid_counts.unsqueeze(1)
    anchors = ordered.gather(1, rank.clamp(max=num_candidates - 1))
    return torch.where(keep_mask, anchors, 0), keep_mask
