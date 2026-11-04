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

# EAGLE3 的编译损失内核：先筛有效行，再做 RMSNorm/head 与分布比较，返回求和量供外层归一化。
# forward_kl 实现是教师软标签交叉熵（比 KL 多教师熵常数）；LK 分支另外统计分布重合质量 alpha。
from typing import Optional

import torch
import torch.nn.functional as F


# float32 中算 logsumexp(z)-sum(p*z)，即 -sum(p log q)，不显式存 log_softmax 张量。
def _forward_kl_from_logits(logits: torch.Tensor, target_p: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.logsumexp(logits_f32, dim=-1) - (target_p * logits_f32).sum(-1)


# 先转 float32，用 exp(z-logsumexp(z)) 求概率，稳定处理大 logits 并避免低精度归一化误差。
def _softmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_f32 = logits.float()
    return torch.exp(logits_f32 - torch.logsumexp(logits_f32, dim=-1, keepdim=True))


# dense 头一次线性得到 V 维；low-rank 头先到 rank 再经 up 到 V，不显式构建稠密权重。
def _project(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_up: Optional[torch.Tensor],
) -> torch.Tensor:
    """Draft logits. ``lm_head_up`` is None for a dense head, the up projection otherwise."""
    logits = F.linear(hidden, lm_head_weight)
    if lm_head_up is not None:
        logits = F.linear(logits, lm_head_up)
    return logits


# 对隐藏态最后维 float32 求均方，回原 dtype 后乘 gamma，再执行输出投影；与 draft final norm 对齐。
def _rmsnorm_logits(
    hs_flat: torch.Tensor,
    norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_up: Optional[torch.Tensor],
    norm_eps: float,
) -> torch.Tensor:
    hs_f32 = hs_flat.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs_flat.dtype) * norm_weight
    return _project(norm_hs, lm_head_weight, lm_head_up)


# 编译区域内 gather valid_idx，只对 N 个有效 [H] 行归一化/投影，返回 soft CE 总和、argmax 正确数和 N。
@torch.compile(dynamic=None)
def compiled_sum_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)
    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == tp.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


# 融合有效行 gather、RMSNorm、LM head 和 soft CE，避免在编译边界外额外物化 [N,V] 副本。
# correct 比较教师分布 argmax 而非数据 token；count 是有效行数，不是原始 B*T。
@torch.compile(dynamic=None)
def compiled_forward_kl_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
):
    """torch.compile'd index_select + RMSNorm + lm_head + Forward KL loss.

    Takes full (B*T, ...) flat tensors and performs index_select inside the
    compiled graph so the compiler can fuse the gather with subsequent ops.

    Args:
        prenorm_hidden_states_flat: (B*T, H) — flattened draft hidden states
        target_p_flat: (B*T, V_out) — flattened target probs (detached)
        valid_idx: (N,) int64 — indices of non-masked positions
        norm_weight: (H,)
        lm_head_weight: (V_out, H) — draft lm_head weight
        norm_eps: float
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp = target_p_flat.index_select(0, valid_idx)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)  # (N, V_out)

    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == tp.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


# 恢复真实子词表质量 p=tp_tilde*coverage，再算 alpha=sum(min(p,q)) 与 -log(alpha)。
# 例如子词表覆盖 0.7，即使 q 完全匹配重归一化教师，alpha 也不应被虚增到 1。
@torch.compile(dynamic=None)
def compiled_lk_alpha_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
    coverage_flat,
):
    """LK^alpha loss: -log(acceptance_rate), summed over valid positions.

    Directly optimizes the (log) acceptance rate alpha_i = sum_x min(p_i(x), q_i(x))
    from "LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"
    (arXiv:2602.23881). Returns (loss_sum, correct_sum, count, alpha_sum) — the same
    sum/count convention as ``compiled_forward_kl_loss`` plus a trailing alpha sum.

    ``target_p_flat`` is the target distribution renormalized over the (possibly
    pruned) draft vocab; ``coverage_flat`` rescales it back to the true,
    un-renormalized target probabilities before computing alpha, per §4.4 of the
    paper (out-of-draft-vocab mass must contribute 0, not be redistributed).
    ``coverage_flat`` is 1 when there is no vocab pruning, leaving ``tp`` unchanged.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp_tilde = target_p_flat.index_select(0, valid_idx)
    coverage = coverage_flat.index_select(0, valid_idx)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)

    tp = tp_tilde * coverage.unsqueeze(-1)
    alpha = torch.min(tp, q).sum(-1)
    token_loss = -torch.log(alpha.clamp(min=1e-8))
    correct = (logits.argmax(-1) == tp_tilde.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


# 每次调用用平均 alpha 的 detach 值生成 lambda=exp(-eta*alpha)，混合 KL 和 TV。
# KL 用归一化子词表 tp_tilde，alpha/TV 用 coverage 还原的 p；USP 可传全组平均值避免分片各自调度。
@torch.compile(dynamic=None)
def compiled_lk_lambda_loss(
    prenorm_hidden_states_flat,
    target_p_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    norm_eps,
    coverage_flat,
    external_mean_alpha,
    eta,
):
    """LK^lambda loss: lambda*KL(p||q) + (1-lambda)*TV(p,q), lambda = exp(-eta*sg[alpha]).

    Returns (loss_sum, correct_sum, count, alpha_sum), matching
    ``compiled_lk_alpha_loss``'s convention. ``lambda`` is computed once per call
    (i.e. once per EAGLE-3 depth) from the mean acceptance rate across the batch
    and sequence, matching the paper's schedule — not per token. See
    ``compiled_lk_alpha_loss`` for ``coverage_flat``'s role in vocab pruning: the
    KL term still uses the renormalized ``tp_tilde`` (raw ``p`` would make KL
    diverge for out-of-draft-vocab mass), but alpha/TV use the true ``tp``.

    ``external_mean_alpha``, if not None, overrides the locally-computed mean
    alpha used to form lambda. This is required under sequence parallelism
    (USP): a single call only sees this rank's shard of the depth's sequence,
    but the schedule is defined over the *full* batch and sequence for the
    depth (arXiv:2602.23881), so the caller must all-reduce alpha across the
    SP group before forming lambda and pass the result in here. Pass None
    when there is no sequence sharding, in which case the local mean already
    covers the whole depth.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp_tilde = target_p_flat.index_select(0, valid_idx)
    coverage = coverage_flat.index_select(0, valid_idx)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)
    log_q = F.log_softmax(logits.float(), dim=-1)

    tp = tp_tilde * coverage.unsqueeze(-1)
    alpha = torch.min(tp, q).sum(-1)
    mean_alpha = external_mean_alpha if external_mean_alpha is not None else alpha.mean()
    lam = torch.exp(-eta * mean_alpha.detach())

    kl = F.kl_div(log_q, tp_tilde, reduction="none").sum(-1)
    # 裁剪路径这里仅计算子词表上的绝对差；完整词表 TV 还含子词表外固定质量的一半。
    # 这部分对 draft 参数是常数，当前实现不加它，因此不能把该 tv 标量直接当完整分布 TV。
    tv = 0.5 * (tp - q).abs().sum(-1)
    token_loss = lam * kl + (1.0 - lam) * tv

    correct = (logits.argmax(-1) == tp_tilde.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


# LazyTarget 路径只对有效教师行做 head+softmax，再计算 draft soft CE，返回可跨 SP 相加的 sum/count。
@torch.compile(dynamic=None)
def compiled_sum_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
):
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)
    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count


# 由有效教师 hidden 延迟构造完整词表概率，直接与 q 求重合质量，完整词表不需要 coverage 修正。
@torch.compile(dynamic=None)
def compiled_lk_alpha_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
):
    """LK^alpha loss from hidden states (LazyTarget path). See ``compiled_lk_alpha_loss``."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)

    alpha = torch.min(tp, q).sum(-1)
    token_loss = -torch.log(alpha.clamp(min=1e-8))
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


# 完整词表 LazyTarget 的 KL/TV 混合；lambda 是整个深度的标量，不是每 token 自适应权重。
@torch.compile(dynamic=None)
def compiled_lk_lambda_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
    external_mean_alpha,
    eta,
):
    """LK^lambda loss from hidden states (LazyTarget path). See
    ``compiled_lk_lambda_loss`` for ``external_mean_alpha``'s role under USP."""
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)
    log_q = F.log_softmax(logits.float(), dim=-1)

    alpha = torch.min(tp, q).sum(-1)
    # lambda is computed once per call (i.e. once per EAGLE-3 depth) from the
    # mean acceptance rate across batch and sequence, matching the paper's
    # schedule — not per token (and, under USP, from external_mean_alpha —
    # the SP-group-reduced mean — rather than this rank's local shard alone).
    mean_alpha = external_mean_alpha if external_mean_alpha is not None else alpha.mean()
    lam = torch.exp(-eta * mean_alpha.detach())

    kl = F.kl_div(log_q, tp, reduction="none").sum(-1)
    tv = 0.5 * (tp - q).abs().sum(-1)
    token_loss = lam * kl + (1.0 - lam) * tv

    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count, alpha.sum()


# 先 gather 教师/草稿隐藏态再生成两边 logits，避免完整 [B*T,V] 常驻。
# 教师 hidden 的冻结由调用链保证，本函数本身不 detach 任意输入；只在冻结教师数据上调用。
@torch.compile(dynamic=None)
def compiled_forward_kl_loss_from_hs(
    prenorm_hidden_states_flat,
    target_hidden_states_flat,
    valid_idx,
    norm_weight,
    lm_head_weight,
    lm_head_up,
    target_lm_head_weight,
    norm_eps,
):
    """torch.compile'd index_select + target softmax + RMSNorm + lm_head + Forward KL loss.

    Like compiled_forward_kl_loss but takes full (B*T, ...) flat tensors and
    performs index_select inside the compiled graph.  This lets the compiler
    fuse the gather with subsequent ops, avoiding a separate (N, V_full) copy
    outside the compiled region.

    Used for the non-pruning (LazyTarget) path where V_full is large.
    """
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # Target probs (detached weights → no grad flows through target)
    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    # RMSNorm
    hs_f32 = hs.float()
    variance = hs_f32.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(variance + norm_eps)
    norm_hs = (hs_f32 * rstd).to(hs.dtype) * norm_weight

    logits = _project(norm_hs, lm_head_weight, lm_head_up)

    token_loss = _forward_kl_from_logits(logits, tp)
    correct = (logits.argmax(-1) == target_logits.argmax(-1)).float()
    count = torch.ones_like(token_loss, dtype=torch.float32).sum()
    return token_loss.sum(), correct.sum(), count
