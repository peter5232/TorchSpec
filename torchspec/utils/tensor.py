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

# 训练目标对齐的小型张量操作；padding 在这里表示定长位移，并不是扩展序列长度。

import torch


# 沿序列维 1 移一位并补零，保持 [B,S,...] 形状；left=False 把 [a,b,c] 变为 [b,c,0]。
# left=True 则变 [0,a,b]；no_grad 表示这个目标对齐操作不建立反向图。
@torch.no_grad()
def padding(tensor, left=True):
    # zeros_like 保留最后一槽的 dtype/device/尾随形状；[B,S,H] 生成 [B,1,H] 零向量，
    # 而 [B,S] 生成 [B,1] 零 ID。这里的 0 是占位值，是否有效由同向移动的 mask 决定。
    # S=1 时切片为空，结果只剩一槽零；S=0 时仍为空，不会凭空补出一个有效 token。
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    # cat 生成新结果，原输入不被原地移动；不过 no_grad 已切断这次位移的反向图。
    # 它适用于目标/ID/mask 对齐，不应不加判断地复用于需要跨步回传梯度的 draft hidden。
    return tensor
