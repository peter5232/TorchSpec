# USP 输入切片：每个 SP rank 获得主块加 ttt 尾部，以支持多步监督跨块边界读取。
# 输出 attention mask 描述本地有效输入，position_ids 描述 Ring 中 Ulysses 聚合后的主块。

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# 统一 token/mask 的 batch 维后，把全序列按 ceil(S/SP) 切片并额外读取 ttt_length 行。
# 例如 S=10、SP=2、ttt=2，rank0 取 [0:7]，rank1 取 [5:10] 后补 2 行零。
# 返回 token、attention/loss mask、两种特征和位置；补齐位置的 mask 标零。
# mask 是否被后端使用必须沿调用链核对，USP attention 当前不会读取传入的 attention_mask。
def split_usp_batch(
    *,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    ttt_length: int,
    sp_rank: int,
    sp_size: int,
    ring_rank: int,
    sp_ring_size: int,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)

    batch_size, input_len = input_ids.shape
    global_len = min(max_len, input_len) if max_len is not None else input_len
    # 例 S=10,SP=4,U=2,Ring=2,TTT=2：主块 C=ceil(10/4)=3，本地存储长 C+TTT=5。
    # 各 SP rank 起点是 0/3/6/9，而不是按“有效 token 数均分”成 3/3/2/2；等长布局便于 collective。
    # max_len 先限制全局前缀：若从 10 裁成 8，C 也改为 2，所有切片与位置边界都随之改变。
    chunk_size = (global_len + sp_size - 1) // sp_size
    sp_ulysses_size = max(1, sp_size // sp_ring_size)
    start = sp_rank * chunk_size
    local_len = chunk_size + ttt_length
    end = min(start + local_len, global_len)

    # 对全局 mask 先 clone，再做局部切片，避免调用方后续改动原 mask 影响本次准备的数据。
    # 其他输入主要通过连续切片/补齐形成输出；是否共享底层存储要看 contiguous 是否实际产生拷贝。
    # 尾部重复只为未来深度取目标：rank0 保存位置 0..4，但本轮主计算只拥有 0..2。
    # rank1 的主块 3..5 与 rank0 的尾部 3..4 重合，不应将这部分重复计作独立训练样本。
    loss_mask = loss_mask[:, :global_len].clone()

    # 按 axis 处理 [S,H] 或 [B,S,...]，先裁全局上限，再取当前片段，补到统一 local_len。
    # 返回 contiguous 张量及真实 valid_len，后者用于 attention mask，不把补零算成有效 token。
    def _slice_and_pad(tensor: torch.Tensor, axis: int, pad_value: int = 0):
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        # F.pad 的参数从最后一维往前解释：[S,H] 的 (0,0,0,p) 不补 H，只在 S 尾部补 p。
        # [B,S,H] 用同一参数也只补 S；[B,S] 则用 (0,p)。不能把 B 或 H 当序列轴补齐。
        if axis == 0:
            tensor = tensor[:global_len, :]
            sliced = tensor[start : min(end, tensor.shape[0]), :]
            valid_len = sliced.shape[0]
            if valid_len < local_len:
                sliced = F.pad(sliced, (0, 0, 0, local_len - valid_len), value=pad_value)
        else:
            tensor = tensor[:, :global_len]
            sliced = tensor[:, start : min(end, tensor.shape[1])]
            valid_len = sliced.shape[1]
            if valid_len < local_len:
                pad_len = local_len - valid_len
                if tensor.dim() == 2:
                    sliced = F.pad(sliced, (0, pad_len), value=pad_value)
                else:
                    sliced = F.pad(sliced, (0, 0, 0, pad_len), value=pad_value)
        return sliced.contiguous(), valid_len

    # attention 的 valid_len 只取 token 切片长度；这里不会比较 hidden/target 的实际有效长度。
    # 因此上游必须保证时间轴一致，不能依赖各字段独立补零来修复错位的 teacher 特征。
    input_ids, valid_len = _slice_and_pad(input_ids, axis=1, pad_value=0)
    loss_mask, _ = _slice_and_pad(loss_mask, axis=1, pad_value=0)
    if hidden_states.dim() == 2:
        hidden_states, _ = _slice_and_pad(hidden_states, axis=0, pad_value=0)
        hidden_states = hidden_states.unsqueeze(0)
    else:
        hidden_states, _ = _slice_and_pad(hidden_states, axis=1, pad_value=0)
    if target_hidden_states.dim() == 2:
        target_hidden_states, _ = _slice_and_pad(target_hidden_states, axis=0, pad_value=0)
        target_hidden_states = target_hidden_states.unsqueeze(0)
    else:
        target_hidden_states, _ = _slice_and_pad(target_hidden_states, axis=1, pad_value=0)

    # attention_mask 由实际输入长度构造，而 loss_mask 来自样本监督范围。
    # 即使某个 prompt token 的 loss_mask=0，它仍可作为后续 token 的上下文，attention_mask 应为 1。
    attention_mask = torch.zeros((batch_size, local_len), dtype=torch.long, device=input_ids.device)
    attention_mask[:, :valid_len] = 1

    usp_chunk_size = max(local_len - ttt_length, 0)
    ring_chunk = usp_chunk_size * sp_ulysses_size
    # 续上例每 Ring 段有 U*C=6 个主位置：SP0/1 都使用 position_ids=0..5，SP2/3 都是 6..11。
    # 本地 tensor 长 5 与位置向量长 6 并不矛盾：后者给 Ulysses 拼序列之后的 Q/K 使用。
    # 10/11 是等长布局中的补齐坐标，不代表原序列真有这些 token；loss 仍须屏蔽无监督位置。
    ring_start = ring_rank * ring_chunk
    # position_ids 长度为主块长度×U，而不是 local_len=主块+TTT。
    # 例如主块=5、TTT=2、U=2，本地 token 长 7，但聚合注意力的位置向量长 10；
    # 形状差异来自不同阶段的布局，而非漏补了位置。
    position_ids = torch.arange(
        ring_start, ring_start + ring_chunk, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    if batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)

    return (
        input_ids,
        attention_mask,
        loss_mask,
        hidden_states,
        target_hidden_states,
        position_ids,
    )
