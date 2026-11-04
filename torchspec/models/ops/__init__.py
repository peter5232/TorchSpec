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

# 训练共用算子入口：FlexAttention mask、蒸馏损失与 assistant 内容监督 mask。
# 注意 attention mask 限制可见性，loss mask 限制监督位置，两者不能互换。
from torchspec.models.ops.flex_attention import (
    build_eagle3_block_mask,
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
    eagle3_block_mask,
    generate_eagle3_mask,
)
from torchspec.models.ops.loss import (
    compiled_forward_kl_loss,
    compiled_lk_alpha_loss,
    compiled_lk_lambda_loss,
)
from torchspec.models.ops.loss_mask import compute_assistant_loss_mask

__all__ = [
    "build_eagle3_block_mask",
    "compile_friendly_create_block_mask",
    "compile_friendly_flex_attention",
    "eagle3_block_mask",
    "generate_eagle3_mask",
    "compiled_forward_kl_loss",
    "compiled_lk_alpha_loss",
    "compiled_lk_lambda_loss",
    "compute_assistant_loss_mask",
]
