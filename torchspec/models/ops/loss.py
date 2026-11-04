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
    # q=softmax(z) 时 log q_i=z_i-logsumexp(z)，且 sum(p)=1，故 CE 可写成下面两项。
    # 前提是 target_p 在最后一维已归一化；直接传裁剪后未归一化的 p 会漏掉质量系数。
    # 对 z 的梯度为 q-p；KL(p||q)=CE-H(p)，冻结 teacher 时二者梯度相同、日志值不同。
    # 例 p=[0.75,0.25]、q=[0.5,0.5]：CE=log(2)，梯度=[-0.25,0.25]。
    logits_f32 = logits.float()
    return torch.logsumexp(logits_f32, dim=-1) - (target_p * logits_f32).sum(-1)


# 先转 float32，用 exp(z-logsumexp(z)) 求概率，稳定处理大 logits 并避免低精度归一化误差。
def _softmax_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # 减去 logsumexp 后再 exp，避免直接 exp(1000) 这类溢出；它与 softmax 的数学定义相同。
    # float() 保留 autograd 链，不是 detach；这里只改变计算精度，不决定 teacher 是否冻结。
    logits_f32 = logits.float()
    return torch.exp(logits_f32 - torch.logsumexp(logits_f32, dim=-1, keepdim=True))


# dense 头一次线性得到 V 维；low-rank 头先到 rank 再经 up 到 V，不显式构建稠密权重。
def _project(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_up: Optional[torch.Tensor],
) -> torch.Tensor:
    """Draft logits. ``lm_head_up`` is None for a dense head, the up projection otherwise."""
    # F.linear(x,W) 实际算 x @ W.T：dense W=[V,H] 将 [N,H] 映到 [N,V]。
    # 低秩时第一个 W=[R,H]，第二个 up=[V,R]；中间 [N,R] 的 R 是秩，不是 token 数。
    # 分两次矩阵乘法避免显式构造 [V,H] 的乘积权重，但最终 [N,V] logits 仍然存在。
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
    # RMSNorm 不像 LayerNorm 那样减均值。每一行独立算 mean(h*h)，不跨 token 或 batch 混合。
    # h=[3,4]、忽略 eps、gamma=[1,1] 时，除数是 sqrt(12.5)，不是标准差 0.5。
    # 先将归一化结果转回输入 dtype 再乘 gamma，顺序影响舍入；解读时不能任意交换。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp_tilde = target_p_flat.index_select(0, valid_idx)
    coverage = coverage_flat.index_select(0, valid_idx)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)

    # coverage=[N] 扩成 [N,1]，逐行恢复保留词表的真实概率质量，不沿 token 轴相乘。
    # 例完整 p=[0.5,0.2,0.3]、只保留前两词：tp_tilde=[5/7,2/7]，coverage=0.7。
    # q=[0.6,0.4] 时 alpha=min(0.5,0.6)+min(0.2,0.4)=0.7；第三词的 q 为零。
    # 此处的 alpha 是分布重合质量，不是这批 token top-1 准确率，也不是实际接受长度。
    tp = tp_tilde * coverage.unsqueeze(-1)
    alpha = torch.min(tp, q).sum(-1)
    # clamp 防止 log(0)；低于阈值的区间经过 clamp 后不再提供 alpha 方向的梯度。
    # alpha 是逐 token 的 [N]，先取 -log 再求和；不等同于对平均 alpha 取一次 -log。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    tp_tilde = target_p_flat.index_select(0, valid_idx)
    coverage = coverage_flat.index_select(0, valid_idx)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)
    log_q = F.log_softmax(logits.float(), dim=-1)

    tp = tp_tilde * coverage.unsqueeze(-1)
    alpha = torch.min(tp, q).sum(-1)
    mean_alpha = external_mean_alpha if external_mean_alpha is not None else alpha.mean()
    # lambda 是本次深度共享的标量。detach 阻止模型通过改变混合权重来“钻目标的空子”，
    # 梯度仍可经下方 KL/TV 流向 q；lambda 的数值则会随下一次前向的 alpha 更新。
    # eta>0 时 alpha 越高 lambda 越小，TV 权重越大；eta=0 时退化为纯 KL。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # 教师 head 接受的是已符合其 final-norm 契约的 ths；此处不会替教师再做 RMSNorm。
    # 只为 N 个有效位置生成 [N,Vfull]，省掉无效位置的词表投影；并非完全不物化 logits。
    # 教师权重与特征都应由调用链冻结：仅冻结 W 并不能阻止梯度流向 requires_grad 的 ths。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # 教师 head 接受的是已符合其 final-norm 契约的 ths；此处不会替教师再做 RMSNorm。
    # 只为 N 个有效位置生成 [N,Vfull]，省掉无效位置的词表投影；并非完全不物化 logits。
    # 教师权重与特征都应由调用链冻结：仅冻结 W 并不能阻止梯度流向 requires_grad 的 ths。
    target_logits = F.linear(ths, target_lm_head_weight)
    tp = _softmax_from_logits(target_logits)

    logits = _rmsnorm_logits(hs, norm_weight, lm_head_weight, lm_head_up, norm_eps)
    q = _softmax_from_logits(logits)

    alpha = torch.min(tp, q).sum(-1)
    # clamp 防止 log(0)；低于阈值的区间经过 clamp 后不再提供 alpha 方向的梯度。
    # alpha 是逐 token 的 [N]，先取 -log 再求和；不等同于对平均 alpha 取一次 -log。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # 教师 head 接受的是已符合其 final-norm 契约的 ths；此处不会替教师再做 RMSNorm。
    # 只为 N 个有效位置生成 [N,Vfull]，省掉无效位置的词表投影；并非完全不物化 logits。
    # 教师权重与特征都应由调用链冻结：仅冻结 W 并不能阻止梯度流向 requires_grad 的 ths。
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
    # lambda 是本次深度共享的标量。detach 阻止模型通过改变混合权重来“钻目标的空子”，
    # 梯度仍可经下方 KL/TV 流向 q；lambda 的数值则会随下一次前向的 alpha 更新。
    # eta>0 时 alpha 越高 lambda 越小，TV 权重越大；eta=0 时退化为纯 KL。
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
    # 展平行号 r=b*T+t。例 B=2,T=3，valid_idx=[1,5] 对应 (0,1) 与 (1,2)。
    # 两侧用同一索引取 [N,H] / [N,V] 或 [N,Htarget]，否则形状正确也会监督错 token。
    # gather 的 backward 会把所选行的梯度散回原行；未选行不通过此损失贡献梯度。
    hs = prenorm_hidden_states_flat.index_select(0, valid_idx)
    ths = target_hidden_states_flat.index_select(0, valid_idx)

    # Target probs (detached weights → no grad flows through target)
    # 教师 head 接受的是已符合其 final-norm 契约的 ths；此处不会替教师再做 RMSNorm。
    # 只为 N 个有效位置生成 [N,Vfull]，省掉无效位置的词表投影；并非完全不物化 logits。
    # 教师权重与特征都应由调用链冻结：仅冻结 W 并不能阻止梯度流向 requires_grad 的 ths。
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
