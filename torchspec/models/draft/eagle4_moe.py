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

# EAGLE4 的稀疏 FFN 与服务权重布局。路由选择使用带偏置 sigmoid 分数，混合权重使用未加偏置的分数。
# 训练步骤结束再更新负载平衡 bias，不增加 auxiliary loss；latent 变体在较小宽度运行 experts。
"""EAGLE-4 MoE draft MLPs, in the weight layout SGLang's EAGLE-4 loader reads."""

import re
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

LATENT_MOE = "nemotron_latent_relu2"
_PACKED_LATENT = re.compile(r"(.*\.experts)\.(up_proj|down_proj)")
_PER_EXPERT_LATENT = re.compile(r"(.*\.experts)\.(\d+)\.(up_proj|down_proj)\.weight")


class Eagle4Router(nn.Module):
    """Sigmoid router whose expert-selection bias balances load without a loss."""

    # 分配专家路由权重、持久 correction bias 和不持久 load 计数；update_rate 决定每步偏置调整幅度。
    def __init__(self, config):
        super().__init__()
        n = config.n_routed_experts
        self.weight = nn.Parameter(
            nn.init.normal_(torch.empty(n, config.hidden_size), std=config.initializer_range)
        )
        self.register_buffer("e_score_correction_bias", torch.zeros(n))
        self.register_buffer("load", torch.zeros(n, dtype=torch.long), persistent=False)
        self.update_rate = getattr(
            config,
            "expert_bias_update_rate",
            getattr(config, "moe_router_bias_update_rate", 1e-3),
        )

    # 执行常规模型设备/dtype 变换后强制 bias 回 float32，避免 bf16 无法表示小幅增量。
    def _apply(self, fn, recurse=True):
        # The bias stays fp32 through model.to(bf16): bf16 cannot take 1e-3 steps past |0.25|.
        super()._apply(fn, recurse)
        self.e_score_correction_bias = self.e_score_correction_bias.float()
        return self

    # 全 rank 求和专家负载、清空计数，低于均值的专家加 bias，高于均值减 bias。
    # 返回最大/平均负载供监控；应按 optimizer step 调一次而非每个 micro-batch 调用。
    @torch.no_grad()
    def update_bias(self) -> float:
        """Called once per optimizer step; returns max / mean expert load."""
        load = self.load.float()
        if dist.is_initialized():
            dist.all_reduce(load)
        self.load.zero_()

        # Under-loaded experts step up, over-loaded ones down.
        self.e_score_correction_bias += self.update_rate * (load.mean() - load).sign()
        return (load.max() / load.mean()).item()


class SharedExpert(nn.Module):
    # 构建共享 up/down，可选 gate；gated=False 用单路激活以匹配 latent 架构。
    def __init__(self, hidden_size, intermediate_size, act_fn, gated):
        super().__init__()
        if gated:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    # 共享专家对每个 token 都执行：gated 时 act(gate(x))*up(x)，否则直接 act(up(x))。
    def forward(self, x):
        h = self.up_proj(x)
        gated = hasattr(self, "gate_proj")
        return self.down_proj(self.act_fn(self.gate_proj(x)) * h if gated else self.act_fn(h))


class Eagle4MoE(nn.Module):
    """``sparse_swiglu`` (default) or ``nemotron_latent_relu2`` MoE replacing the dense MLP."""

    # 创建路由器、打包专家权重和可选 latent/shared 投影；routed down 初始化为零。
    # 因此新增专家库初始贡献为零，适合从已有 dense/shared 输出开始训练。
    def __init__(self, config):
        super().__init__()
        self.latent = getattr(config, "moe_architecture", "sparse_swiglu") == LATENT_MOE
        self.top_k = config.num_experts_per_tok
        self.n_group = getattr(config, "n_group", None) or 1
        self.topk_group = getattr(config, "topk_group", None) or self.n_group
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.act_fn = ACT2FN[config.hidden_act]
        n, hidden, inter = config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
        width = config.moe_latent_size if self.latent else hidden

        self.gate = Eagle4Router(config)

        # Packed experts: SwiGLU as w13 (gate | up) and w2, latent ReLU^2 as up and down.
        # Zero down projections make a new routed bank a no-op, so an upcycled draft starts
        # from its dense output.
        up = torch.empty(n, inter if self.latent else 2 * inter, width)
        up = nn.Parameter(nn.init.normal_(up, std=config.initializer_range))
        down = nn.Parameter(torch.zeros(n, width, inter))
        self.expert_names = ("up_proj", "down_proj") if self.latent else ("w13_weight", "w2_weight")
        self.experts = nn.ParameterDict(dict(zip(self.expert_names, (up, down))))

        # Latent experts run between these projections.
        if self.latent:
            self.fc1_latent_proj = nn.Linear(hidden, width, bias=False)
            self.fc2_latent_proj = nn.Linear(width, hidden, bias=False)

        # n_shared_experts are folded into one full-width expert.
        n_shared = getattr(config, "n_shared_experts", None) or 0
        shared_inter = config.moe_shared_expert_intermediate_size if self.latent else inter
        self.shared_experts = (
            SharedExpert(hidden, shared_inter * n_shared, self.act_fn, gated=not self.latent)
            if n_shared
            else None
        )

    # 先按每组最高两个 biased 分数筛组，再 top-k 选专家；最终权重从未加 bias 的 sigmoid 分数读取。
    # 可归一化所选权重后乘 routed_scaling_factor，平衡偏置只改变选择而不直接改变混合概率。
    def route(self, x):
        """DeepSeek-V3 routing: top-k experts per token and their combine weights."""
        scores = F.linear(x.float(), self.gate.weight.float()).sigmoid()
        biased = scores + self.gate.e_score_correction_bias

        # Group-limited routing keeps the topk_group groups with the best top-2 biased scores.
        if self.n_group > 1:
            groups = biased.unflatten(-1, (self.n_group, -1))
            best = groups.topk(2, dim=-1).values.sum(-1).topk(self.topk_group, dim=-1).indices
            dropped = torch.ones_like(groups[..., 0], dtype=torch.bool).scatter_(1, best, False)
            biased = groups.masked_fill(dropped[..., None], float("-inf")).flatten(1)

        # The bias only selects experts; the unbiased scores weight them.
        idx = biased.topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, idx)
        if self.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        return idx, weights * self.routed_scaling_factor

    # 把 token-expert 对按专家排序，用 counts.cumsum 作 grouped GEMM 边界，算各专家 FFN。
    # 按路由权重缩放后 index_add 回 token；同 token 的多个专家相加，最后加 shared 分支并恢复原形状。
    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        idx, weights = self.route(x)
        counts = torch.bincount(idx.flatten(), minlength=self.gate.weight.shape[0])
        if self.training:
            self.gate.load += counts

        # (token, slot) pairs sorted by expert feed one grouped GEMM per projection.
        order = idx.flatten().argsort()
        # flatten 后第 p 个 token-expert 对的 token id 为 p//top_k；排序只重排执行顺序，不改变归属。
        tokens = order // self.top_k
        offs = counts.cumsum(0).int()
        h = self.fc1_latent_proj(x) if self.latent else x
        up, down = (self.experts[name] for name in self.expert_names)
        inner = F.grouped_mm(h[tokens], up.transpose(1, 2), offs=offs)
        if self.latent:
            inner = self.act_fn(inner)
        else:
            gate, value = inner.chunk(2, dim=-1)
            inner = self.act_fn(gate) * value
        out = F.grouped_mm(inner, down.transpose(1, 2), offs=offs)

        # Weight each expert output and sum it back onto its token.
        out = out * weights.flatten()[order, None].to(out.dtype)
        # 同一 token 可能路由到多个专家，必须加和而非 index_copy 覆盖；路由权重已在上一步乘入。
        out = h.new_zeros(h.shape).index_add_(0, tokens, out)
        if self.latent:
            out = self.fc2_latent_proj(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out.view_as(hidden_states)


# 把打包 [E,...] latent 权重拆成逐专家键并 clone，每个导出张量独立持有存储。
def split_latent_experts(tensors: dict) -> dict:
    """Packed latent experts to the one-key-per-expert layout SGLang serves."""
    out = {}
    for key, value in tensors.items():
        m = _PACKED_LATENT.fullmatch(key)
        out.update(
            {f"{m[1]}.{i}.{m[2]}.weight": w.clone() for i, w in enumerate(value)}
            if m
            else {key: value}
        )
    return out


# 按专家数字 id 排序再 stack，恢复训练所需打包张量，未匹配的键保持不变。
def stack_latent_experts(tensors: dict) -> dict:
    """Inverse of ``split_latent_experts``."""
    out, groups = {}, defaultdict(dict)
    for key, value in tensors.items():
        m = _PER_EXPERT_LATENT.fullmatch(key)
        if m:
            groups[f"{m[1]}.{m[3]}"][int(m[2])] = value
        else:
            out[key] = value
    out.update({k: torch.stack([g[i] for i in sorted(g)]) for k, g in groups.items()})
    return out
