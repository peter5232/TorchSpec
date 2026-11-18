# 从一个序列读懂 padding、USP 切片与位置坐标

如果你习惯 `[B,S,H]`，容易默认每个张量里的 S 都是同一条序列长度。本仓库中原始长度、存储切片长度、当前预测主块长度、Ulysses 通信后的长度是不同的量。先把这些量写清，再看 attention 的 reshape 和位置编码。

## 1. Collator 先沿哪一维组 batch

打开 [DataCollatorWithPadding](../../../torchspec/data/utils.py)。输入是非空的样本字典列表，且每条 `input_ids` 已含 batch 维。例：

```text
A.input_ids     [1,5]          B.input_ids     [1,130]
A.hidden_states [1,5,12]       B.hidden_states [1,130,12]
```

普通路径将最大长度 130 向上取整到 128 的倍数 256，再分别补零、沿 dim=0 拼接：

```text
batch.input_ids     [2,256]
batch.hidden_states [2,256,12]
batch.loss_mask     [2,256]
```

A 新增 251 槽，B 新增 126 槽。分桶用一部分补齐计算换取更少的动态 shape；是否真正跳过 padding 的算量，要继续看后端。`paddingtensor` 中局部变量 S 实际是特征宽度，n 才是原序列长度；不要因为名字叫 S 就把最后维也当时间轴。

还有两个容易遗漏的条件：

- `torch.cat` 的 batch 维是各输入 B_i 的和。若把 `[2,S]` 的整批当一个 item，输出 batch 并非一定等于列表元素数。
- 可选 `target`/`last_hidden_states` 只有每项都具备才拼接。A 有 target、B 没有时 batch.target 为 None，不会为 B 伪造全零概率分布。

普通路径按每条 input_ids 的现有长度重建全一 attention_mask。若 A 本来已经补到 `[1,128]`，即使原 mask 仅前 5 槽为 1，再次经过此路径也会先生成 128 个 1。因此对评估缓存再次 collate 时，不能套用“输入一定是原始无 padding 序列”的假设。USP 路径则保留上游 attention_mask。

## 2. 同名 padding 函数做的是另一件事

[utils/tensor.py](../../../torchspec/utils/tensor.py) 中 `padding` 是固定长度的一位移动：

```text
输入：       [a,b,c,d]
left=False：[b,c,d,0]
left=True： [0,a,b,c]
```

它不把长度从 4 改成 5。`@torch.no_grad()` 也意味着不能把它随意拿来移动需要反向传播的草稿 hidden。序列长度为 1 时只剩零；空序列仍为空。0 是占位值，监督是否有效必须由一起对齐的 mask 决定，不能依据 token 数值判定。

## 3. 用 S=10、SP=4 手算所有切片

打开 [split_usp_batch](../../../torchspec/utils/usp.py)。设 Ulysses 度数 U=2、Ring 度数 R=2，因此 SP=U×R=4；额外目标尾长 TTT=2，原位置为 0..9。主块 C=ceil(10/4)=3，每个存储切片长 L=C+TTT=5。

| SP rank | 主块对应坐标 | 存储切片中的真实位置 | 补零槽数 | Ring rank |
|---:|---|---|---:|---:|
| 0 | 0,1,2 | 0,1,2,3,4 | 0 | 0 |
| 1 | 3,4,5 | 3,4,5,6,7 | 0 | 0 |
| 2 | 6,7,8 | 6,7,8,9 | 1 | 1 |
| 3 | 9,10,11 | 9 | 4 | 1 |

表里的 10、11 是补齐布局坐标，不是原文真实 token。各 rank 使用相同 C，便于 collective；不是把 10 个真实 token 分成 3/3/2/2 后让每卡 shape 不同。

rank 0 的尾部 3、4 与 rank 1 主块重合。它们给后续展开深度取目标使用，不能再计为新的独立样本。[Eagle3Model.forward](../../../torchspec/models/eagle3.py) 每轮将 step_input_ids、step_hidden_states、step_mask 裁到主块 C 后再送 backbone。跟踪一个位置时，应同时记录“它是否被传输了”和“它是否属于本 rank 的主计算”，二者答案可以不同。

生产端 [put_usp_shards](../../../torchspec/transfer/mooncake/eagle_store.py) 调用这个函数，写出各 `_usp{rank}` 对象；消费者[数据读取器](../../../torchspec/training/data_fetcher.py) 根据相同参数重建真实监督 mask 与 position_ids。生产端用于切片的 dummy loss_mask 不等于最后训练用的监督范围。

## 4. 为什么本地长度 5，position_ids 却长 6

本地存储是 C+TTT=5，当前位置向量却按一个 Ring 段的 U×C=6 个主位置生成：

```text
SP0 与 SP1：position_ids = [0,1,2,3,4,5]
SP2 与 SP3：position_ids = [6,7,8,9,10,11]
```

原因在 [USP attention](../../../torchspec/models/draft/llama3_eagle.py)：Q/K/V 投影时先处理每 rank 的 C=3 个主槽，随后 Ulysses all-to-all 拆头、拼序列。以 B=1、Q heads=8、KV heads=4、D=2 为例：

| 张量阶段 | 每 rank Q | 每 rank K/V |
|---|---|---|
| 本地主块投影 | [1,3,8,2] | [1,3,4,2] |
| U=2 通信之后 | [1,6,4,2] | [1,6,2,2] |
| 输出逆向通信 | [1,3,8,2] | 不适用 |

每 rank 的 Q 元素数保持 48；通信把局部工作从“短序列、全部头”变成“较长序列、一部分头”。RoPE 在通信后的 6 个位置上使用上述 position_ids。Ring 再负责跨另一个 Ring 段的注意力前缀，并不是把 TTT 尾部也当成新 query 拼进去。

此例选取的 Q/KV 头数都能被 U 整除，不能据此假设任意 GQA 配置都支持任意 U。具体后端支持条件要与初始化检查、通信实现一起读。

## 5. mask 存在，不代表所有后端都读取它

切片函数按 token 的真实有效长度生成 attention_mask：rank 2 为 `[1,1,1,1,0]`，rank 3 为 `[1,0,0,0,0]`。loss_mask 则保留原来的监督范围后补零；真实 prompt token 可以 attention=1、loss=0。

但 `LlamaUSPFlashAttention.forward` 显式丢弃传入的 attention_mask。这个事实不能被“上游 mask 标零”掩盖。因果前缀等布局约束如何处理合法查询、哪些补齐 query 仍会实际计算、loss 最后如何屏蔽，要沿具体后端逐层核对；不能声称任意带空洞的 mask 在这里都生效。也不要把标准 Flash 路径对有效前缀长度的处理无条件推广到其他后端。

切片函数还只用 input_ids 的 valid_len 生成 attention_mask，没有比较 hidden/target 的实际长度。若上游误传一份较短 teacher 特征，这里可能独立补零，不能自动发现所有时间轴错位。shape 被补得一致，不等于语义已对齐。

## 6. 用六个问题检查自己的推导

1. 将 max_len 从 10 改成 8，C、各 rank 起点、Ring 段长度分别如何变化？——C=2，起点 0/2/4/6，Ring 段长度 4；本地含尾部长度仍是 4。
2. 为什么 rank 0 读到位置 3 不等于它拥有位置 3 的主查询？——位置 3 是目标尾部，当前主计算只截取前 C 行。
3. position_ids 为什么不能简单用本地 `start:start+local_len`？——它描述通信后的主序列，既不含 TTT，也不只描述本 rank 的主块。
4. 为什么不把 loss_mask 当 attention_mask？——上文可参与条件计算而不承担监督。
5. 为什么不能从 attention_mask 的值直接断言 kernel 行为？——调用链可能不读取该参数，或只支持有效前缀等受限形式。
6. 为什么直接换成 torch.chunk 不是等价改写？——当前实现还规定统一主块长度、目标尾部、补齐与 Ring/Ulysses 位置坐标。

这些是当前源码的坐标与数据契约推导，未运行真实多卡通信。继续阅读 attention 与分布式章节时，可以沿同一组维度把每一次 transpose、view、all-to-all 和 RoPE 写在这张表旁边。
