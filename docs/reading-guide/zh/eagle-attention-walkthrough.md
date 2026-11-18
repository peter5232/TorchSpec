# EAGLE attention：同一条预测链如何经过不同后端

本文沿 [llama3_eagle.py](../../../torchspec/models/draft/llama3_eagle.py) 的张量计算走一遍。入口是文件末尾 `LlamaForCausalLMEagle3.backbone -> LlamaDecoderLayer.forward`；外层 [Eagle3Model.forward](../../../torchspec/models/eagle3.py) 反复调用这个 backbone，并保留上一轮缓存。这里的“轮”是预测深度，不是 Transformer 层号，也不是普通推理中每次只追加一个 token 的解码步。

以下小尺寸用于手算；不代表实际 GPU kernel 支持这些尺寸或本次已运行 GPU 验证。默认讨论普通文本 RoPE、连续有效前缀，不推导 mRoPE 或任意 packed 样本的支持。

## 1. 投影之前先看输入宽度

固定 `B=1,S=4,H=8,Hq=4,Hkv=2,D=2`。`fuse_layer_input` 分别 norm token embedding 与历史 hidden，沿最后维拼成 `[1,4,16]`。普通路径保留原 hidden 作残差；若配置 `attn_norm`，则先经 `eh_proj` 压回 H，attention 输入和残差起点都随之改变。

在普通路径里，Q/K/V 权重分别是 `[8,16]`、`[4,16]`、`[4,16]`。线性层逐 token 投影，没有在序列维混合信息：

| 张量 | 投影结果 | SDPA/Flex 排列 | Flash 排列 |
|---|---|---|---|
| Q | [1,4,8] | [B,Hq,S,D]=[1,4,4,2] | [B,S,Hq,D]=[1,4,4,2] |
| K/V | [1,4,4] | [B,Hkv,S,D]=[1,2,4,2] | [B,S,Hkv,D]=[1,4,2,2] |

本例 Q 的头数恰巧等于序列长度，所以两种 shape 数字一样，语义却不同。查看 transpose 和 RoPE 的 `unsqueeze_dim` 才能判断坐标；不要仅根据打印 shape 推断排列。

`repeat_kv` 将两个 KV 头展开为 `[kv0,kv0,kv1,kv1]`，Q 头 0/1 共享 kv0，Q 头 2/3 共享 kv1。标准 Flash merge 则把 Q reshape 为 `[B,S,Hkv,groups,D]`，令 K/V 在 groups 维广播；这表达相同头分组，而不先复制 KV。

## 2. 文本位置、缓存地址、候选列是三个坐标

令基础 `position_ids=[0,1,2,3]`。第三次调用之前已有两轮，因此 lck=2，当前 Q/K 用 `[2,3,4,5]` 查 RoPE。旧 K 已经旋转过，不会再按新深度旋转一次；V 不做 RoPE。

对普通 D=2 的单头向量 `[a,b]`，`rotate_half` 返回 `[-b,a]`；结合 cos/sin 后即 `[a*cos-b*sin,b*cos+a*sin]`。更高维按“前半/后半”配对，不能直接套相邻通道对的权重布局。

第三轮槽 t=1 的合法键是：

```text
首轮 k0[0], k0[1]   +   第二轮 k1[1]   +   第三轮 k2[1]
```

不同后端保存它们的方法不同：

| 后端 | 完成第三轮后的 cache K | 当前槽1如何找到键 |
|---|---|---|
| SDPA | [1,4,3,4,2]，KV 已重复为 Hq | 首轮矩阵乘，再读每轮槽1 |
| Flex | [1,2,12,2] | 物理 KV 下标 0、1、5、9 |
| 标准 FA | [1,3,4,2,2] | 第0轮前缀 + 后两轮各自槽1 |
| FA4 | [B,3,bucket,Hkv,D] | 按 bucket 间距定位同槽 |

SDPA 构造的 score 行长 `S+(rounds-1)=6`：前四列是整个首轮，后两列各对应一个链键。槽1这行前四列的后两项被 causal mask 屏蔽。Flex 的物理缓存长 12，但不是每个 query 都计算一个普通的长度 12 因果前缀。

一个后端生成的 cache 不能直接交给另一个后端；同一条链跨轮也不能随意改变 S，否则缓存长度除以 S 得到的轮数与同槽取模都将改变。

## 3. 为什么标准 Flash 要保留 LSE

从 `_FlashCachedMergeFunc.forward` 开始：前缀调用 varlen causal FA，得到 `out0` 和 `lse0`；每个后续轮只有一个合法键，分支输出就是该槽 value，分支 LSE 就是缩放后的 Q·K 分数。

用一个 query 的单个输出分量算例：

```text
前缀未归一化质量 Z0=3，前缀归一化输出 out0=2
链键未归一化质量 Z1=1，链 value=10
总质量 Z=4，输出 = (3/4)*2 + (1/4)*10 = 4
```

实现保存 `merged_lse=log(4)`，用 `exp(lse_i-merged_lse)` 求权重。前缀已归一化输出没有携带它的质量，只有 out0 无法正确合并。直接平均 2 和 10 得 6，和实际输出 4 不同。

变长路径先去 padding：B=2、S=4、有效长度 [3,2] 时，flatten 索引 `[0,1,2,4,5]` 形成五个 token，`cu_seqlens=[0,3,5]` 使 kernel 区分两个独立样本。返回时 out/LSE 按索引放回原形状。标准 FA wrapper 的有效长度来自 `attention_mask.sum()-lck`，因此 mask 必须描述连续前缀；它不能保留中间空洞的原位置。

## 4. 自定义 backward 的三种梯度

`_FlashCachedMergeFunc.backward` 读取前向保存的 Q、整个 cache、合并 out/LSE。假设上例上游梯度为 1：

```text
链 value 梯度 = w1 = 0.25
链 score 梯度 = w1 * (value1 - out) = 0.25*(10-4) = 1.5
```

后者的 `-out` 来自所有键共享的 softmax 分母。若误用链自身输出 10，就会错误得到零；单键分支虽然局部 softmax 恒等于 1，它相对整个前缀的权重仍然随 score 改变。

逐行对照代码：

1. `d_out_i = grad_out_f * wi` 是链 value 的贡献。
2. `d_lse_i = wi * sum(grad_out_f*(vi-out))` 是 score 贡献。
3. score 是 `scale*Q·K`，所以 dQ 加 `d_lse_i*scale*K`，dK 加 `d_lse_i*scale*Q`。
4. 多个 Q 头共享一个 KV 头，dK/dV 沿 groups 维求和，不能取平均。
5. 前缀仍由 FA backward 计算，但传入合并 out/LSE，让它使用全体键的分母和输出，而非前缀自己的局部统计。

返回的是整个 cache 的梯度；普通 PyTorch 的 cat 反向再把梯度传回各历史轮。缓存不是 detach 的推理缓存，因此后轮 loss 能影响前轮 QKV 投影所用的参数和输入。

FA4 的 `_EagleMaskedFlashAttnFunc` 走另一个实现：全部键一次进入带自定义 mask 的 kernel。forward 按 Q 查 KV 块，backward 按 KV 查 Q 块；两方向索引不同，但描述同一可见关系。backward 既要总 out/LSE，也必须带相同 mask；否则可能为前向没有访问的 Q-K 对计算梯度。

## 5. FA4 的 bucket 不是有效长度

取 orig_q_len=130，bucket=256。每轮先补到 256 再做投影，第三轮缓存有 `3*256` 个物理键。槽5的同链键依次在 5、261、517。输出最后裁回 130 行，缓存仍保存 256 的轮宽。

当前 `_build_eagle3_mask_pair` 的 closure 分支将 q_len 同时作为有效长度；FA4 forward 没有读逐样本 attention_mask。真实有效前缀查询不会通过因果首轮或同槽 suffix 读到右侧占位槽，但占位查询本身不因此成为零。输出裁切只移除 bucket 新增的 126 行，原 batch 中已有的样本 padding 仍可能有输出。

## 6. USP：四个 rank 的具体交换

取 Ulysses=2、Ring=2、本地序列长2、Hq=4、Hkv=2、D=2，采用 [distributed.py](../../../torchspec/utils/distributed.py) 的低位 Ulysses 排列：

| SP rank | Ulysses 坐标 | Ring 坐标 | 输入位置 |
|---|---|---|---|
| 0 | 0 | 0 | 0,1 |
| 1 | 1 | 0 | 2,3 |
| 2 | 0 | 1 | 4,5 |
| 3 | 1 | 1 | 6,7 |

`LlamaUSPFlashAttention.forward` 分别投影 Q/K/V，再分别调用 SeqAllToAll4D，scatter 头维2、gather 序列维1。每个 Ulysses 组内：Q 从 `[1,2,4,2]` 变成 `[1,4,2,2]`，KV 从 `[1,2,2,2]` 变成 `[1,4,1,2]`。元素总数不变，每 rank 现在拿更长序列的一部分头，GQA 比率仍是 2。这要求头数满足相应划分条件；这段代码没有自动填头补救。

RoPE 在交换之后执行。ring0 的 position_ids 是 [0,1,2,3]，ring1 是 [4,5,6,7]；外层 wrapper 默认按 ring_rank 构造它们。不能仍使用通信前仅两个位置的 ID。第 d 轮完成后的缓存为 `[1,d+1,4,1,2]`，持续留在通信后的坐标系。

Ring attention 只接收首轮 cache。ring1 的查询5需要远端首轮0..3与本地4..5；后续轮只需本 rank 已持有的槽5链键。跨 rank 的 send/recv 在 yunchang 内实现，这个文件调用其 forward/backward，并没有自行展开传输循环。后续 suffix 循环是本地计算，逐次通过 logaddexp 合并 out/LSE。

最后逆向 all-to-all 交换 scatter/gather 维号，将输出 `[1,4,2,2]` 还原为 `[1,2,4,2]`，合头后执行 o_proj。缓存不做这次逆变换。SeqAllToAll4D 本身带 autograd，反向须做对应通信；这与训练框架另行进行的参数梯度同步不是同一阶段。

## 7. “等价”应检验哪些范围

SDPA 缓存分支要求预先构造的加性 mask；标准 FA 用有效前缀长度去 padding；Flex 与 FA4 不读取传入的 attention_mask，USP 更显式删除它。因此相同签名不保证任意输入 mask 下行为一致，尤其不能承诺所有 padding 行输出相同。

合理对照需要相同投影权重、RoPE/缩放、输入链与有效查询，既比较前向，也比较多轮反向。可继续阅读 [Flex/标准 FA 测试](../../../tests/test_flex_attention.py)、[FA4 测试](../../../tests/test_fa4.py)、[USP 测试](../../../tests/test_usp_attention.py)。这些测试本身的形状、mask 与依赖替代范围也要逐项核对，不能据一个通过的算例推断所有后端、混合并行布局或 mRoPE 都已验证。
