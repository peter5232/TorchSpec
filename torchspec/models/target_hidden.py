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

# 用临时 forward hook 验证教师最后一层与 final norm 的精确边界，避免把 pre/post-norm 状态混用。
# 返回 detach 的二者；每次调用都检查真实张量关系，而不是依赖模型输出字段的名字。
"""Capture a Hugging Face decoder's hidden states on both sides of its final norm.

Which side of the final norm a hidden state comes from is easy to get wrong:
``output_hidden_states[-1]`` and ``last_hidden_state`` are post-norm in most
Hugging Face decoders, while vLLM layer captures are often pre-norm. Forward
hooks on the final norm and the last decoder block capture the exact tensors on
both sides, and the boundary is checked on every call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FinalNormBoundary:
    """Tensors on both sides of the final norm, detached from any graph."""

    pre_norm: torch.Tensor
    post_norm: torch.Tensor


# 注册 norm 和最后 block 的 hook，执行一次 no_grad forward，并在 finally 中移除所有 hook。
# 逐项验证 norm 输入等于末层输出、重算 norm 一致、返回状态是 post-norm；恒等 norm 无法观测边界则报错。
@torch.no_grad()
def capture_final_norm_boundary(decoder: nn.Module, **forward_kwargs) -> FinalNormBoundary:
    """Run ``decoder`` and return the input and output of ``decoder.norm``.

    ``decoder`` must expose ``layers`` and ``norm`` and return an object with
    ``last_hidden_state`` (e.g. a Hugging Face ``*Model`` or ``*TextModel``).
    Raises if the norm input is not bitwise the last block's output, if
    re-applying the norm does not reproduce the captured output, or if the
    decoder's returned state is not the post-norm tensor.
    """

    captured: dict[str, torch.Tensor] = {}

    # 同时捕获 norm 的输入与输出引用，确保比较的是同一次 forward 的两侧张量。
    def norm_hook(module, args, output):
        captured["pre_norm"] = args[0]
        captured["post_norm"] = output

    # 兼容 block 返回 tuple 或 Tensor，取其隐藏态与 final norm 输入逐位比较。
    def block_hook(module, args, output):
        captured["block"] = output[0] if isinstance(output, tuple) else output

    # 临时 hook 只覆盖本次 forward，捕获真实调用链；不会修改模块 forward 或参数。
    handles = [
        decoder.norm.register_forward_hook(norm_hook),
        decoder.layers[-1].register_forward_hook(block_hook),
    ]
    # 即使 decoder forward 抛异常也要 remove hook，否则以后重复运行会不断积累捕获副作用。
    try:
        output = decoder(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    # 四份张量即使都是 [B,S,H] 也不能仅凭 shape 判等：这里验证的是数值和调用边界。
    # 对 RMSNorm，重复应用一般不幂等（可学习的逐通道权重尤其如此）；
    # 把 pre_norm 直接送 head 或把 post_norm 再 norm 一次，都会改变教师监督分布。
    pre_norm = captured["pre_norm"]
    post_norm = captured["post_norm"]
    if not torch.equal(pre_norm, captured["block"]):
        raise RuntimeError("final-norm input is not the last decoder block's output")
    # 此时 hook 已被移除，重算 norm 不会覆盖 captured；torch.equal 是逐元素精确比较，
    # 用来检验同一次确定性 norm 的重放，不是跨后端数值误差容忍测试。
    if not torch.equal(decoder.norm(pre_norm), post_norm):
        raise RuntimeError("re-applying the final norm does not reproduce its captured output")
    # 若两侧完全相等，无法证明抓到了可区分边界，主动拒绝给调用方一个看似可靠的判定。
    if torch.equal(pre_norm, post_norm):
        raise RuntimeError("final norm is an identity on this input; boundary is not observable")
    if not torch.equal(output.last_hidden_state, post_norm):
        raise RuntimeError("decoder output is not the final norm's output")
    # detach 只切断 autograd，不复制底层存储；返回值也不是任意后续原地修改的独立快照。
    return FinalNormBoundary(pre_norm=pre_norm.detach(), post_norm=post_norm.detach())
