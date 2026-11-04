# 用小张量手算模型包装器

这是 [模型章节](models.md) 的配套演算，按真实函数中的变量顺序走一遍。以下小尺寸用于纸面检查，不代表 GPU kernel 支持这些 tile，也不是运行测试结果。`x_t` 表示位置 t 的 token，`h_t` 表示教师 final norm 后的状态；`head(h_t)` 是下一 token 的分布。

## EAGLE：先固定一行，再观察两轮

打开 [Eagle3Trainer._forward](../../../torchspec/training/eagle3_trainer.py)，设原序列为 `[x0,x1,x2,x3,x4,x5]`。trainer 左移 token 与末层教师状态，给 wrapper 的分别是 `[x1,x2,x3,x4,x5,0]` 和 `[h1,h2,h3,h4,h5,0]`，辅助特征 `[f0,...,f5]` 保持原位。这里零是补位，不是自然语言 token；是否监督仍由 mask 决定。

接着打开 [Eagle3Model.forward](../../../torchspec/models/eagle3.py)，跟踪槽 t=1。假设三路教师特征每路宽 8、草稿宽 4，则输入特征 `[1,6,24]` 经 `project_hidden_states` 变为 `[1,6,4]`。投影在循环外，因为后续输入已经是草稿自己的隐藏态。

| 槽 t=1 的内容 | idx=0 | idx=1 |
|---|---|---|
| 输入 token | x2 | x3 |
| hidden 输入 | project(f1) | next_depth_hidden(上一轮输出槽1) |
| 教师目标所在行 | padded_target[1] | padded_target[2] |
| 教师分布 | head(h2)，预测 x3 | head(h3)，预测 x4 |
| 输出张量形状 | [1,6,4] | [1,6,4] |

沿 `hidden_states = hidden_states_out` 向下读：先算当前 loss，再产生下一轮反馈 hidden。不要把这两处 norm 合并：loss 所需的输出归一化与反馈接口是两份约定。输入 token 仍用真值，所以该循环没有执行线上 token 采样。

`cache_keys` 在循环外初始化、在每轮 backbone 内追加。以 Flex 布局为例，两轮之后缓存是 `[B,Hkv,12,D]`；槽 1 的第二轮查询读取首轮键 0、1，以及本轮物理键 7。键 6 属于槽 0 的另一条预测链，即使更早写入也不合法。这里缓存保持计算图，后轮损失可以通过它回传。

### `_calculate_loss` 中三个维度如何对齐

另取 `B=2,T=3,Hdraft=4`，监督 mask 为：

```text
sample 0: [0,1,0]
sample 1: [1,0,1]
```

`flatten().nonzero()` 得 `[1,3,5]`；draft hidden 展成 `[6,4]`。目标先沿时间切 `idx:idx+3`，再按同样 batch 优先顺序展平。编译 loss 内仅选这三行，再 norm/head。这避免为其余三行分配词表 logits；不是先把所有 logits 算完再给 loss 乘零。

若教师词表 5 个 token，只保留 1、3，`t2d=[False,True,False,True,False]` 会选 head 的第 1、3 行。某行全概率为 `[0.1,0.5,0.1,0.2,0.1]` 时：

```text
position_mask = 1             # 全词表 argmax 是 token 1，在子词表内
coverage = 0.7                # 教师在子词表上的真实概率质量
target_p_padded 的该行 = [5/7,2/7]
```

这不是把词表 token 1 重编号成全词表 token 0；它只是 draft 输出的列 0。LK loss 乘回 coverage 后用 `[0.5,0.2]`，而条件分布 KL 用 `[5/7,2/7]`。另一个完整分布即使 coverage 很高，只要完整 argmax 落在子词表外，仍会被 position_mask 排除：行可用性与概率质量是独立量。

## DFlash：同一批张量的三套坐标

打开 [DFlashModel](../../../torchspec/models/dflash.py)，设 `B=1,S=8,N=2,K=3`，输入 token id 是 `[10,11,12,13,14,15,16,17]`。假设本次采样返回 `anchors=[2,5]`、两个 keep 均为 True；并非每次随机采样都返回此结果。

| 步骤 | 第一个块 | 第二个块 |
|---|---|---|
| `_create_noise_embed` 的 token 输入 | [12,MASK,MASK] | [15,MASK,MASK] |
| draft 内存槽 | [0,1,2] | [3,4,5] |
| `_create_position_ids` 的文本位置 | [2,3,4] | [5,6,7] |
| 拼接 context 后的 KV 下标 | [8,9,10] | [11,12,13] |
| `forward` 的标签 | [12,13,14] | [15,16,17] |

[_draft_backbone](../../../torchspec/models/dflash.py) 将这些数据交给 [draft attention](../../../torchspec/models/draft/dflash.py)。Q 只从 draft 产生，长 6；K/V 从 context 与 draft 两种来源投影后拼接，长 14。位置编码描述文本位置，BlockMask 描述是否可见，两者不能互相替代。

以第二块中间查询为例，`q_idx=4` 得块号 1、块内偏移 1、文本位置 6。默认非因果块内策略允许：

```text
context KV: 0,1,2,3,4       # 必须严格早于 anchor=5
draft KV:   11,12,13         # 仅第二块，输入是 [15,MASK,MASK]
```

第一块 KV=8..10 不可见。context 的位置 5..7 也不可见，尽管它们已经由教师算出；如果允许读取 context[6]，查询就可能获得包含真实未来 token 的特征。块内可以双向，是因为未来输入放的是 MASK。

再代入 `_create_dflash_mask_mod` 的滑窗条件，令 `W=3`：query 的文本位置是 6，context 只剩 4，因为 `6-kv < 3` 且 `kv < 5`。块内位置 5、6、7 都满足距离条件；若还启用 `is_causal`，再去掉位置 7。滑窗的距离比较在文本位置上进行，不能拿物理 KV 下标 11..13 直接减查询位置 6。

### 从标签 gather 到日志的同一条算例

`label_indices` 是 `[[[2,3,4],[5,6,7]]]`，`target_ids` 是 `[[[12,13,14],[15,16,17]]]`。全序列均监督时，`weight_mask` 是 `[[[0,1,1],[0,1,1]]]`；第 0 槽输入已知 anchor，因此不优化。

若启用 L1，预测 token13 对应教师 `h2`，预测 token14 对应 `h3`。这正是 `tgt_idx = safe_label_indices - 1` 的原因：DFlash 同位置槽与普通自回归 head 的下一位置预测之间相差一位。

单看一个块的两个预测槽，令 CE 为 `[2,4]`、目标权重为 `[1,0.5]`：

```text
loss_numerator = 2*1 + 4*0.5 = 4
loss_denominator = 1 + 0.5 = 1.5
loss = 8/3
```

若只有第一槽预测正确，accuracy 是 1/2，而非加权后的 2/3。`loss_per_position` 记录未乘 decay 的 CE；启用 L1 也不会把 L1 加入这条 CE 曲线。trainer 收到最后的 `(numerator,denominator)`，才能按整个累积窗口池化，避免不同有效 token 数的 batch 被赋予相同权重。

D-PACE 则先从 CE 恢复正确 token 的概率。假设 `[0.6,0.2]`、alpha=0.5，依次得到平滑置信度 `[0.8,0.6]`、前缀乘积 `[0.8,0.48]`、反向累和权重 `[1.28,0.48]`。权重可以超过 1；早槽影响多个更长前缀。整个权重计算无梯度，所以训练不会通过调整权重本身绕过 token 拟合。

## 将 EAGLE 的 token 可见性压成块表

打开 [_build_eagle3_block_mask_tensors](../../../torchspec/models/ops/flex_attention.py)，用 `Q_LEN=8,KV_LEN=16,Q_BS=4,KV_BS=2` 手算。此时 `r=2,n_q=2,n_kv=8,n_rounds=2`。这只是公式演示，实际 kernel 使用更大的 tile。

| Q 块 | query token | FULL KV 块 | PARTIAL KV 块 |
|---|---|---|---|
| 0 | 0..3 | [] | [0,1,4,5] |
| 1 | 4..7 | [0,1] | [2,3,6,7] |

Q 块 1 的 `kv_indices` 一行实际宽 8：`[2,3,6,7,0,0,0,0]`，`kv_num_blocks=4` 告诉 kernel 只读前四项。末尾零不是重复访问 KV 块 0。`full_kv_indices` 独立存 `[0,1,0,0,0,0,0,0]`，FULL count=2。

反向需要从 KV 块反查所有使用它的 Q 块：KV 块 0 的 PARTIAL Q=[0]、FULL Q=[1]；KV 块 6 的 PARTIAL Q=[1]、FULL Q=[]。这是相同可见关系的另一方向，用于累积 dK/dV，不是另一种注意力算法。

PARTIAL 只意味着“进入这个 tile 再逐 token 判断”。例如第二轮 KV 块 6 覆盖物理键 12、13，Q 块 1 中 query 4 只能读键 12，query 5 只能读键 13，query 6/7 在此块没有合法键。把整个 tile 标为 FULL 会改变模型语义。

这种构造省掉 token 级 8×16 网格；块索引本身仍是补齐宽度的矩形数组。阅读英文 docstring 的复杂度简述时，应以这里实际分配的张量维度为准。
