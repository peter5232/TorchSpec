# 从两个输入张量走完 DFlash attention 与 DFlash2 selector

[模型推演](model-walkthrough.md)解释 wrapper 怎样选择 anchor、构造 mask 与标签。本篇从 wrapper 已经准备好输入开始，只追踪参数骨干里的投影、旋转、卷积和候选打分。

## 一个 GQA 层的形状账本

取 batch B=1、context 长 S=8、draft 长 T=6、残差宽 H=16、Q 头数 4、KV 头数 2、每头宽 D=4。教师捕获三路，每路 `[1,8,16]`。

`extract_context_feature` 先沿最后维拼成 `[1,8,48]`，经 `context_proj:48→16` 与 `context_norm` 得到 `[1,8,16]`。这一步融合教师层，不缩短序列。列表的先后顺序就是权重列分组的含义；该方法不会根据 `target_layer_ids` 重新排序输入。

[实际 attention](../../../torchspec/models/draft/dflash.py)每次 forward 走以下形状：

| 阶段 | draft 分支 | context 分支 |
|---|---|---|
| 输入 | `[1,6,16]` | `[1,8,16]` |
| Q 投影 | `[1,6,16]`，拆成 `[1,6,4,4]` | 不计算 Q |
| Q norm 后转置 | `[1,4,6,4]` | — |
| K/V 投影 | 两个 `[1,6,8]` | 两个 `[1,8,8]` |
| 按序列拼接 K/V | 各为 `[1,14,8]` | 同一拼接结果 |
| K 拆头、norm、转置；V 拆头、转置 | 各为 `[1,2,14,4]` | 同一结果 |
| attention 输出 | `[1,4,6,4]` | 没有 context 输出 |
| 合并头、输出投影 | `[1,6,16]` | — |

Q/K 的 RMSNorm 都只沿最后的四个通道计算均方，gamma 形状为 `[4]`，不是把所有头一起归一化。V 不做此 norm，也不做 RoPE。显式 `head_dim` 可使 `num_heads×head_dim` 与残差 H 不等，此时 `q_proj` 和 `o_proj` 分别进入和离开注意力内部宽度。

“共享 K/V 投影”指本层 context/draft 两种输入调用同一组 `k_proj`、`v_proj`。下一 decoder 层有自己的参数。所有层接收同一 context tensor，但重算的 K/V 不一定相同；模块也没有持久化上一调用的 context KV cache。context 不走 draft 的残差更新，梯度仍可从它流回可训练的 `context_proj`。

## GQA 的头对应关系

每个 KV 头供两个 Q 头读取。无 mask 的 SDPA 分支显式重复得到头序 `[kv0,kv0,kv1,kv1]`，对应 Q0/Q1 和 Q2/Q3；不是交替排列 `[kv0,kv1,kv0,kv1]`。`expand` 是广播视图，随后合并轴的 reshape 可能产生实际存储，不能把整个辅助函数称为零拷贝。

带 BlockMask 的 Flex 分支传 `enable_gqa=True`，不先显式重复 K/V。概念上的 attention 分数尺寸是 `[1,4,6,14]`，kernel 无需一定把它完整物化。KV 头数减少不改变 Q 头数或输出的六个 token。这里的 SDPA fallback 是无 mask 双向注意力，不能拿它的输出证明训练块隔离正确。

## RoPE：配对的是通道，查表的是文本坐标

D=4、base=10000 时，逆频率为 `[1,0.01]`，复制成两半后是 `[1,0.01,1,0.01]`。若向量为 `[a,b,c,d]`，`_rotate_half` 得到 `[-c,-d,a,b]`；旋转配对为 `(a,c)` 与 `(b,d)`。

沿用两个 anchor `[2,5]`、每块三槽的例子：draft 位置为 `[2,3,4,5,6,7]`，context 位置为 `[0,1,2,3,4,5,6,7]`。KV 的物理行 8 对应 draft 第一槽，却使用文本位置 2 的 cos/sin；不是位置 8。Q 查表结果 `[1,1,6,4]` 广播到四个头，K 查表结果 `[1,1,14,4]` 广播到两个头。

这一实现请求的 cos/sin 长度是 `S+T=14`，没有改成最大 position ID 加一。因此输入坐标必须在 `[0,14)` 内。构造器预建较长缓存也不会允许本次传入位置 1000，因为 forward 已把返回表切到 14 行。wrapper 的局部文本坐标满足要求；外部自定义坐标须自己遵守这个接口边界。

## DFlash2：卷积包在子层两侧

[DFlash2](../../../torchspec/models/draft/dflash2.py)复用父类层堆栈。GQA 仍用上述 attention；选择 MLA 时换成 `K3DSparkMLAAttention`，不能继续套用本篇的 KV 头形状。两种 attention 都被同样的块内卷积包裹。

独立取一个卷积小例子：B=1、T=6、块长3、H=8、每组2通道、核长2。`blocks` 为 `[2,3,4,2]`，依次是块数、槽数、组数、组内通道数。动态核为每个输出槽、每个 offset、每个组生成增量；该增量在组内两通道共享，基础核则仍是逐通道参数。

每个块内，固定组和通道后公式为：

```text
y[t] = coeff[t,0] × x[t] + coeff[t,1] × x[t-1]
```

t=0 时第二项补零，不读取前一个块的末尾。例如一个通道输入 `[2,3,5]`，若所有槽系数都为 `[1,0.5]`，输出为 `[2,4,6.5]`。实际系数可随 t 改变，不能把它当作固定卷积核。

`prepare(norm(x))` 一次预测“子层前”和“子层后”两套动态核。前核立刻卷积；attention/MLP 之后 `finish` 使用预留的后核，不在新输出上再次预测。前/后表示位置在子层的两侧，两次卷积都向本块左侧取值，不是一次向左、一次向右。完整顺序为 norm → 前卷积 → 子层 → 后卷积 → 原输入残差。

基础核 offset0=1、其余=0，动态投影初始为零，因此卷积初始是恒等；这不使整个 decoder 恒等，attention/MLP 仍参与计算。

## Selector：分数、候选槽和词表 ID

训练 wrapper 已先用 LM head 得到全词表 logits，再将首槽之后的位置交给 selector。两块、每块三槽时，hidden 为 `[1,2,2,H]`，logits 为 `[1,2,2,V]`；selector 的 top-k 只缩减后续候选打分，不省略全词表投影。

设某位置候选 ID 为 `[7,2,9]`，unary logits 为 `[4,6,5]`。训练正确后继是 3 且其原 logit=1，则替换最低 unary 所在槽 0，得到 ID `[3,2,9]`、unary `[1,6,5]`。因为 `topk(sorted=False)` 不排序，最低值不保证在最后一槽。若正确词本就在候选中，则保留原候选。

秩 R 的修正为：

```text
condition = predecessor_codebook[前驱ID] * hidden_projection(hidden)
score[k] = unary[k] + dot(condition, successor_codebook[候选ID[k]])
```

R=2 时，若前驱码 `[2,1]`、hidden 投影 `[0.5,3]`，条件向量为 `[1,3]`。候选码 `[0.2,0.1]` 带来修正 `0.5`。这是相加的分数，未做概率归一化；方法也没有按新 score 排序或自动解码，而是返回分数与候选 ID 的对应数组。

wrapper 用真实前驱、强制纳入正确后继，并查正确词所在**候选槽索引**做 K 类 CE。top-k 的离散选取不求梯度，但选中 logits、hidden 投影和码本仍求梯度；caller 未 detach hidden/logits，因此 selector 附加损失也训练草稿骨干。推理不提供正确后继时，自然没有这个候选替换步骤。

最后，`DFlash2DraftModel.forward` 只缩放输入 embedding，返回的仍是 final-norm 后 hidden；logits 的 `output_multiplier` 和 softcap 在 [训练 wrapper](../../../torchspec/models/dflash2.py) 中执行。把权重骨干单独调用不等于已经完成候选打分和最终解码。
