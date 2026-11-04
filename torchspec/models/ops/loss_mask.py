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

# 按 token 序列中的 assistant header/end 标记建立内容监督 mask，CPU 上由 Numba 扫描。
# 这是 loss 监督范围，不是 attention 可见性：用户 prompt 虽不计 loss，仍能作为上下文。
import numba
import numpy as np
import torch


# 顺序查找完整 header，跳过 header 后指定 token，再标记直到 end 前的内容。
# 若未找到 end，则标记剩余尾部；输出原地写入 out，header/end 本身不监督。
@numba.njit(cache=True)
def _numba_loss_mask(ids, header, header_len, end, end_len, out, skip_after):
    # 这是按整数 token 子串匹配的扫描器，不会解析角色语义或重新 tokenize。
    # 调用方应给出非空、与当前聊天模板一致的 header/end；内容里出现同样的 end 子串也会结束该段。
    # 复杂度还含 header/end 的逐项比较成本；只有模板长度视作常数时才可简称 O(N)。
    n = len(ids)
    i = 0
    while i <= n - header_len:
        match = True
        for k in range(header_len):
            if ids[i + k] != header[k]:
                match = False
                break
        if not match:
            i += 1
            continue
        # i 指 header 起点，j 指第一个可监督内容位置。skip_after 跳过的是实际 token 数，
        # 不是字符数；若模板换行已与正文合并成一个 BPE token，跳过它会同时跳过那部分正文。
        j = i + header_len + skip_after
        found_end = False
        while j <= n - end_len:
            end_match = True
            for k in range(end_len):
                if ids[j + k] != end[k]:
                    end_match = False
                    break
            # 先测试完整 end，再给 out[j] 置 1，所以多 token end 的每个组成 token 都不会被监督。
            if end_match:
                found_end = True
                break
            out[j] = 1
            j += 1
        # 当 end 长度为 2，扫描在 j=n-1 时已无法做完整匹配；剩余尾 token 仍按正文标 1。
        # 因此截断在回答中途且缺少 end 的样本会监督到尾部，不会整段丢弃。
        if not found_end:
            for k in range(j, n):
                out[k] = 1
        i = j + end_len
    return out


# 必要时将 CUDA token 搬到 CPU、转 int64 并连续化，确保 Numba 获取稳定的一维数组布局。
def _to_contiguous_int64_numpy(t: torch.Tensor) -> np.ndarray:
    if t.is_cuda:
        t = t.cpu()
    if t.dtype != torch.int64:
        t = t.to(torch.int64)
    t = t.contiguous()
    return t.numpy()


# 把标记 id 转 NumPy，调用扫描器后可只保留最后一段连续 1，再搬回原输入设备。
# last_turn_only 实际按最后连续内容段截取；skip_after_header 用于处理模板换行与 BPE 合并。
def compute_assistant_loss_mask(
    input_ids: torch.Tensor,
    assistant_header_ids: list[int],
    end_token_ids: list[int],
    last_turn_only: bool = False,
    skip_after_header: int = 0,
) -> torch.Tensor:
    """Compute loss mask where 1s mark assistant content tokens only.

    Uses a JIT-compiled single-pass O(N) scan to find assistant header
    sequences and pair them with end token sequences.
    Computes on CPU and returns result on the same device as input_ids.

    Args:
        input_ids: 1-D tensor of token IDs (CPU or CUDA).
        assistant_header_ids: Token ID sequence marking the start of assistant content.
        end_token_ids: Token ID sequence marking the end of assistant content.
        last_turn_only: If True, only the last assistant turn is marked.
        skip_after_header: Number of tokens to skip after header match before
            marking content. Use 1 when the header excludes a trailing newline
            that BPE may merge with subsequent content.

    Returns:
        1-D long tensor on the same device as input_ids, with 1s for assistant
        content tokens and 0s elsewhere.
    """
    device = input_ids.device if isinstance(input_ids, torch.Tensor) else torch.device("cpu")
    ids_np = (
        _to_contiguous_int64_numpy(input_ids)
        if isinstance(input_ids, torch.Tensor)
        else np.asarray(input_ids, dtype=np.int64)
    )
    header_np = np.array(assistant_header_ids, dtype=np.int64)
    end_np = np.array(end_token_ids, dtype=np.int64)
    out = np.zeros(len(ids_np), dtype=np.int64)
    _numba_loss_mask(ids_np, header_np, len(header_np), end_np, len(end_np), out, skip_after_header)

    # 例 ids=[9,10,11,20,21,99,10,11,30,99]，header=[10,11]、end=[99]、skip=0：
    # 全轮 mask=[0,0,0,1,1,0,0,0,1,0]；last_turn_only 最终只留下索引 8。
    # 这是从现成 mask 找最后一段连续的 1；如果最后一个回答为空，留下的是前面最后一段非空正文。
    if last_turn_only and out.any():
        last_one = len(out) - 1 - np.argmax(out[::-1])
        first_of_last = last_one
        while first_of_last > 0 and out[first_of_last - 1] == 1:
            first_of_last -= 1
        out[:first_of_last] = 0

    # from_numpy 与 out 共享 CPU 存储，但 tensor 会持有该存储引用；函数返回不会使它悬空。
    # 原输入若在 GPU，上面的 .cpu() 和这里的 .to(device) 形成往返传输，不能当成纯设备端 mask 内核。
    result = torch.from_numpy(out)
    return result.to(device) if device.type != "cpu" else result
