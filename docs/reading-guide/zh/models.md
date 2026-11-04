# 从 PyTorch 前向到投机草稿训练

本章面向已经理解 `nn.Module`、交叉熵和反向传播，但没有实现过推理引擎的模型研究者。先把 TorchSpec 看成一个**用大模型产生特征、训练小模型预测未来的系统**。在线推理时，草稿提出 token，大模型验证；本目录主要实现前者的训练计算，并不包含完整的线上接受/拒绝调度器。

建议按三个层次阅读：先掌握张量和监督对齐，再比较草稿架构，最后研究注意力后端与分布式细节。这里的代码链接都指向当前仓库；源码里的中文注释解释具体分支，本章串起跨文件关系。

## 第一层：只跟踪一份训练样本

### 1. 教师提供两种不同的隐藏态

从 [HFTargetModel.generate_eagle3_data](../../../torchspec/models/target/eagle3_target_model.py) 开始。它给指定 decoder blocks 注册 forward hook，运行一次冻结教师，再将捕获到的若干路隐藏态拼接。设：

| 符号 | 含义 | 小例子 |
|---|---|---|
| B | batch 中样本数 | 2 |
| S | padding 后序列长度 | 8 |
| Ht | 教师隐藏宽度 | 16 |
| H | 草稿隐藏宽度 | 12 |
| A | 教师捕获层数量 | 3 |
| V | 教师词表大小 | 32 |

则 `hidden_states` 是 `[2,8,48]`，每个 token 有三个教师层的特征。`last_hidden_states` 则是 `[2,8,16]`，用途是通过冻结教师 LM head 得到概率分布。它们不能互换：前者是草稿输入特征，后者是监督目标的来源。

HF 后端捕获 **final norm 的输出**作为 `last_hidden_states`。其他来源若传的是 final norm 输入，训练端必须先应用正确的冻结 norm。不要凭 `last_hidden_state` 或“最后层”这样的名字判断是否已经 norm。[capture_final_norm_boundary](../../../torchspec/models/target_hidden.py) 实际抓取 norm 两侧并验证：最后 block 输出等于 norm 输入，重算 norm 等于捕获输出，decoder 返回的状态等于 norm 输出。

[TargetLMHead](../../../torchspec/models/target/target_utils.py) 只加载 head，以及按需加载 final norm。它的 `forward()` **只执行 head**，不会自动调用 `self.norm`；norm 是否需要、何时执行由 trainer 明确处理。`load_synced_target_lm_head` 让 rank 0 读权重，其他 rank 构造同样参数结构后接收广播。初始化失败要先让所有 rank 达成共识，否则一个 rank 报错后其余 rank 会卡在 broadcast。

### 2. 三种 mask 是三件事

| 名称 | 控制什么 | 例子 |
|---|---|---|
| attention mask | 一个查询能读取哪些键 | 未来真实 token 不可见；不同草稿块隔离 |
| loss mask | 哪些位置参与监督 | assistant 正文为 1，prompt/padding 为 0 |
| block keep mask | 固定预算里的锚点槽是否真实有效 | 要 512 个锚点却只有 20 个可用，剩余 492 槽禁用 |

[compute_assistant_loss_mask](../../../torchspec/models/ops/loss_mask.py) 在 CPU 上扫描 header/end token 序列，标记 assistant 内容。prompt 不计 loss，不代表模型不能读 prompt。`last_turn_only` 实现为保留最后一段连续的 1；不要把任意带空洞的监督 mask 和 attention 的有效前缀长度混同。

### 3. EAGLE3 第一轮为什么已经是未来预测

先读 [Eagle3Trainer._forward](../../../torchspec/training/eagle3_trainer.py)，再读 [Eagle3Model.forward](../../../torchspec/models/eagle3.py)。trainer 先把 `input_ids` 与教师 `last_hidden_states` 左移一位；辅助层特征不做同样左移。

令原始 token 为 `x0,x1,x2,x3,...`，教师辅助特征为 `f0,f1,f2,f3,...`，final-norm 后状态为 `h0,h1,h2,h3,...`。教师 `head(h_t)` 预测的是 `x_(t+1)`。进入 wrapper 时，槽 t 同时拿到：

- 辅助特征 `f_t`，经 `fc` 融合成草稿状态；
- 真实输入 token `x_(t+1)` 的 embedding；
- 教师目标 `head(h_(t+1))`，对应 `x_(t+2)` 的分布。

因此不能直接把 wrapper 槽 t 的输出解释成普通 causal LM 的 `x_(t+1)` 预测；调用前已经做过对齐。wrapper 内第 d 轮再左移输入 token 与监督 mask，并把目标切为 `target[:, d:d+S]`。同一个槽 t 的预测链继续向未来推进，但其内存槽号 t 保持不变。

`length` 是训练展开的深度，不是草稿 Transformer 层数。每轮复用同一个 backbone 的参数，上一轮产生的 hidden 反馈给下一轮，误差可以沿多轮计算图反传。token embedding 仍取真实训练 token，所以这是带真实 token 条件的训练展开，不是线上采样出来的整条草稿链。

### 4. 先看最朴素的草稿参数层

从 [LlamaForCausalLMEagle3](../../../torchspec/models/draft/llama3_eagle.py) 文件末尾往上读：

1. `project_hidden_states` 将 `[B,S,A*Ht]` 压缩成 `[B,S,H]`。可选逐路 RMSNorm；Llama 实现默认用 float32 做融合投影，再转回参数 dtype。
2. `embed_input_ids` 用完整教师词表查 embedding。即使输出词表裁剪了，输入仍可能包含任意教师 token。
3. `fuse_layer_input` 分别 norm embedding 与 hidden，再拼成 `[B,S,2H]`。启用 `attn_norm` 的变体还会经 `eh_proj` 压回 H，并以该投影作为残差。
4. attention 输出加残差，pre-norm MLP 再加残差，得到本轮隐藏态。
5. loss 对此隐藏态做 final RMSNorm 和输出投影。`next_depth_hidden` 另行选择下一轮反馈 raw hidden、final-norm hidden 或 post-MLP norm hidden。

**输出归一化与反馈归一化是两份契约。** 不能因为下一轮用的是 norm 后状态，就给当前 loss 删除 norm；也不能先把状态 norm 再传给已经会做 norm 的 fused loss。

## 第二层：理解概率目标与架构差异

### 5. LazyTarget 与裁剪词表

[Eagle3Model](../../../torchspec/models/eagle3.py) 提供两种目标表示。

| 目标 | 持久保存 | 何时得到概率 | 主要取舍 |
|---|---|---|---|
| `LazyTarget` | `[B,S+length,Ht]` 教师状态与 `[V,Ht]` head | loss 内先筛有效行，再投影 | 避免常驻完整 `[B,S,V]` |
| `PrecomputedTarget` | `[B,S+length,Vd]` 子词表概率、位置 mask、coverage | 构造目标时 | 用较小词表训练，同时保存真实概率质量 |

`t2d` 在这里是长度 V 的**布尔选择表**，不是整数 token-id 映射。`target_lm_head_weight[t2d]` 按教师行顺序选出子词表。模型加载已有裁剪 head 后，不能只替换 t2d/d2t；head 第 j 行已对应固定教师 token，换映射不换权重会静默错训。[set_vocab_buffers](../../../torchspec/models/draft/base.py) 专门拒绝这种情况。

以教师完整概率 `p=[0.6,0.3,0.1]`，只保留前两个词为例：coverage `S=0.9`，存储的子词表概率为 `p_tilde=[2/3,1/3]`。如果 draft `q=[2/3,1/3]`：

- 子词表条件分布完全匹配；
- 真实重合质量仍只有 `min(0.6,2/3)+min(0.3,1/3)=0.9`；
- 若忘了乘 coverage，就错误地报告 alpha=1。

构造目标还检查教师**完整词表 argmax**是否在子词表：不在就把相应训练行 mask 掉，避免要求模型模仿它根本无法输出的首选 token。完整 logits 按有效行分块生成，降低临时内存峰值。

### 6. “forward KL” 返回的标量究竟是什么

看 [ops/loss.py](../../../torchspec/models/ops/loss.py) 的最前面：`_forward_kl_from_logits` 算的是

`logsumexp(z) - sum(p*z) = -sum(p*log q)`。

这是教师软标签交叉熵，与 `KL(p||q)` 相差教师熵常数，对 draft 梯度相同，但数值不同。`p=q` 时该函数一般不为零；不要用“loss 不到零”判断实现错误。

LK 目标另有两种：

- `lk_alpha`：`alpha=sum_v min(p_v,q_v)`，优化 `-log(alpha)`；裁剪路径先用 coverage 还原真实 p。
- `lk_lambda`：用当前深度的平均 alpha 生成 `lambda=exp(-eta*stop_grad(mean_alpha))`，混合 KL 与 TV 项。lambda 是一个深度级标量，不是每 token 独立调节。

裁剪路径的 KL 用归一化 `p_tilde`；alpha/TV 使用 `p_tilde*coverage`。实现里的 TV 只求子词表上的绝对差，完整词表外的固定质量未加进这个标量，因此不能把它直接当完整分布 TV 日志。该缺少部分对 draft 参数是常数。

每个编译 loss 先 `index_select(valid_idx)`，再做 RMSNorm/head，避免在未监督行上构造昂贵词表概率。返回的是 `loss_sum, correct_sum, count`，LK 再附 `alpha_sum`；外层按需要归一化。`correct` 比较教师与草稿 argmax，不必等于对原始数据 hard label 的准确率。

USP 的序列在多个 rank 上：决定 lambda 的 alpha 也必须先按整个 SP 组汇总。空 shard 仍须参与同一次 collective，并构造与参数连接的零 loss；“本卡没有有效 token 就 return 0”可能破坏分布式反向或造成等待。

### 7. DFlash：把多步预测摆成一个块

先读 [DFlashModel](../../../torchspec/models/dflash.py)，再读 [DFlashDraftModel](../../../torchspec/models/draft/dflash.py)。DFlash 一次构造 N 个块，每块长度 K，draft 张量为 `[B,N*K,H]`。

设原序列长度 10，锚点 `a=4`，块长 `K=3`：

| 槽 j | draft 输入 | RoPE 位置 | DFlash 标签 | 是否监督 |
|---|---|---:|---|---|
| 0 | `x4` | 4 | `x4` | 否，已知锚点 |
| 1 | MASK | 5 | `x5` | 由原 loss mask 决定 |
| 2 | MASK | 6 | `x6` | 由原 loss mask 决定 |

每块只能读**严格早于 anchor**的教师上下文，以及本块 draft 槽；看不到其他块。基础 DFlash 块内双向可见是允许的，因为未来输入是 MASK，而不是实际未来 token。不要把“块内双向”直接等同于标签泄漏。

物理下标和语义位置分开：若 anchors 为 `[2,7]`、K=3，平铺 draft 的下标为 `[0,1,2,3,4,5]`，RoPE 坐标却是 `[2,3,4,7,8,9]`。attention 的 KV 物理布局是 `[完整 context | block0 | block1]`，mask 必须同时识别来源、块号、块内偏移和真实 anchor 坐标。

`DFlashAttention` 的 Q 只由 draft 产生；K/V 对 context、draft 使用同一组线性权重，再沿序列维拼接。Q/K 都有逐头 RMSNorm，随后各自按不同位置表做 RoPE。骨干最后确实 `return self.final_norm(draft_hidden)`，因此返回的是 **post-norm** 状态；文件中保留的历史 docstring 说 pre-norm，应以实际 return 和新增说明为准。

CUDA 包装路径会构建 BlockMask；没有 mask 的底层 fallback 是无掩码双向 SDPA。CPU 上形状前向能跑，不代表已验证 CUDA 块隔离语义。理解掩码请阅读谓词与对应测试，不能只看一次 CPU 输出有限。

### 8. 锚点预算为何固定

[ops/anchors.py](../../../torchspec/models/ops/anchors.py) 只允许自身与下一位置都监督的锚点。否则孤立监督 token 无法产生有效的后继预测。有效位置先排序，再以随机相位和受限步长在有效位置环上抽取；发生回绕时，返回槽顺序不一定按数值递增。

固定 N 保持 `Q_LEN=N*K` 不随每条样本有效 token 数变化，减少编译形状变化。若可用锚点少于预算，额外槽用 `keep_mask=False`，不是重复监督真实锚点。`max_gap` 计的是监督位置之间的距离；跨长 prompt 的原始 token 距离可能很大。

[AnchoredEagle3Model](../../../torchspec/models/anchored_eagle3.py) 也复用采样器，但执行方式不同：上下文 S 行只做 K/V 的逐位置投影，attention 查询与 MLP 只算 N 行。其单层结构保证上下文 K/V 不依赖该层 attention 输出。推广到多层不能机械照搬，上层 K/V 会依赖下层的上下文计算。

Anchored EAGLE 上下文可见到 anchor 自身，DFlash 是严格小于 anchor；两者输入对齐与 mask 契约不同，不应合并成一个未经检查的统一条件。

### 9. DFlash loss 的权重不是准确率权重

DFlash 默认 hard-label CE，也可加 L1 蒸馏。标签位置为 `anchor+j` 时，教师分布必须取 `last_hidden_states[anchor+j-1]` 经 head 得到，因为教师 hidden[t] 预测 token[t+1]。

合法监督 mask 先乘块有效性、边界、非锚点、原 loss mask。之后优化目标再乘位置权重：

- decay：预测槽 1 权重 1，槽 2 权重 `exp(-1/gamma)`，依此类推；
- D-PACE：令平滑置信度 `s=(1-alpha)*q(correct)+alpha`，求前缀连乘，再做反向累计。

例如两个平滑置信度 `[0.8,0.5]`，前缀质量 `[0.8,0.4]`，权重为 `[1.2,0.4]`。早期正确性影响更长可接受前缀，所以得到更大权重。权重计算在 no_grad 下，避免模型通过降低自己的权重来降低 loss。

返回的优化量是加权分子与权重分母，trainer 在跨 rank、跨梯度累积窗口后统一处理。准确率与逐位置 CE 使用加权前的监督 mask；它们描述预测质量，而不是原样复现带权优化标量。

### 10. DFlash2：块内动态卷积与候选选择

[DFlash2DraftModel](../../../torchspec/models/draft/dflash2.py) 在 attention 和 MLP 两边分别添加前/后卷积。隐藏态 `[B,N*K,H]` 重排为 `[B*N,K,G,C]`，其中 C 是 group_size，G=H/C。每 token 生成组级动态核，组内 C 个通道共享动态偏移；基础核仍可逐通道不同。

核长为 2 时，槽 j 的输出混合 `x_j` 和 `x_(j-1)`；块首的前驱用零填充。卷积是在每个块内部执行，不能读前一个块的末尾。基础核 offset0=1、其他=0，动态投影初始化为零，因此起步为恒等映射。`prepare` 同时生成前核与后核，后核在子层计算后使用，但条件来自子层输入。

候选选择器只重排 unary logits 的 top-k：

`score(candidate) = unary(candidate) + (code(prev) * W_hidden*h) dot code(candidate)`。

这用秩 R 的码本表达条件转移，避免 V×V 参数矩阵。训练时如果 gold 后继不在 top-k，就替换候选中最低 unary logit 的项；因为 `topk(sorted=False)`，不能直接替换最后一项。

[DFlash2Model](../../../torchspec/models/dflash2.py) 的附加 CE 从第 1 槽开始，用真实前驱/后继做 teacher forcing；权重遇到首个无效槽就截断。它还按层创建 full/sliding 策略，并应用 output_multiplier 与可选 tanh softcap。导出配置必须保留这些语义，不能只导出权重然后使用任意服务默认值。

### 11. DSpark：同骨干，不同标签偏移

[DSparkModel](../../../torchspec/models/dspark.py) 复用 DFlash 的上下文、噪声和块掩码，但是槽 j 预测 `anchor+j+1`。对于前面的 a=4、K=3：

| 槽 j | 输入 | DSpark 标签 | Markov 前驱 |
|---|---|---|---|
| 0 | x4 | x5 | x4 |
| 1 | MASK | x6 | x5 |
| 2 | MASK | x7 | x6 |

所有 K 个槽都可监督，decay 因而是 `exp(-j/gamma)`，不再是 DFlash 的 `j-1`。监督 mask 用 cumprod 保留连续前缀：`[1,0,1]` 变成 `[1,0,0]`。

[VanillaMarkov](../../../torchspec/models/draft/dspark.py) 用前驱 embedding `[V,R]` 和 R→V 投影产生 bigram logits 偏置，加在基础 head logits 上。训练前驱是真值，而不是上一个 draft argmax。损失组合为：

`ce_alpha*CE + l1_alpha*L1(p,q) + confidence_alpha*BCE(confidence_logit, stop_grad(alpha))`。

对完整归一化分布，`alpha=1-L1/2=sum(min(p,q))`。例如 p=[0.8,0.2]、q=[0.6,0.4]，L1=0.4，接受质量 alpha=0.8。confidence 头预测这个软值；即使 L1 系数为零，启用 confidence 仍需教师末层状态。其 BCE 标签 detach，避免通过改动“目标接受率”而逃避拟合。

K3 DSpark 用 MLA 替换 GQA。Q 来自 draft，KV 来自 context+draft；只旋转 rope 子维，V 不旋转。RoPE 缓存先小后按需增长，避免百万上下文配置在每层初始化时就分配大表。

### 12. MTP 与 MoE 放在什么位置理解

[MegatronMTPModel](../../../torchspec/models/mtp.py) 把 batch-first `[B,S,H]` 转成 Megatron 的 `[S,B,H]`。原生 MTP 将 hidden[t] 与 token[t+1] embedding 结合；[训练目标](../../../torchspec/training/mtp.py) 将输出第 t 行与 token[t+2] 比较，最后没有标签的行不监督。不要给它套 DFlash 的同位置标签规则。

[Qwen3.5 权重导入](../../../torchspec/models/qwen3_5_mtp.py) 有三个不可忽略的转换：

1. Qwen RMSNorm 存 gamma-1，Megatron 参数存 gamma，加载必须 +1。
2. HF Q 投影逐头排列 query/gate；Megatron 按 KV group 排列 queries、gates、K、V。
3. 专家并行时只读本 rank 的专家；全局专家编号与本地列表下标不同。

数值例子：4 个 Q 头、2 个 KV 头、head_dim=1，HF q 行是 `[q0,g0,q1,g1,q2,g2,q3,g3]`，应重排成 `[q0,q1,g0,g1,k0,v0,q2,q3,g2,g3,k1,v1]`。loader 先 NaN 填充所有参数，再检查目标覆盖率与源键消费情况，避免随机初值隐藏加载遗漏。

[EAGLE4 MoE](../../../torchspec/models/draft/eagle4_moe.py) 是另一种草稿 FFN 替换方案，不等于上述原生 MTP 专家层。其 router 用 sigmoid 分数+平衡 bias 选专家，但混合权重来自未加 bias 的分数。token-expert 对按专家排序后运行 grouped GEMM，再 index_add 回 token；一个 token 命中多个专家，因此不能用覆盖写回。负载计数在 optimizer step 后汇总并调整 bias，偏置保持 float32，以免 bf16 表示不了小更新。

## 第三层：注意力缓存和 kernel 为什么越来越复杂

### 13. EAGLE 的缓存不是普通逐 token decode cache

看 [LlamaAttention.forward](../../../torchspec/models/draft/llama3_eagle.py)。设 S=4，当前已经走到 depth=2。query 槽 1 的合法键是：

`k0[0], k0[1], k1[1], k2[1]`。

首轮 k0 是原始因果上下文；k1/k2 是同一训练起点的预测链。不能看 `k1[0]`，因为它属于另一个起点的展开链。这里最容易把“sequence 槽”“训练深度”和“文本绝对位置”混在一起。

| 后端 | 主要 KV 布局 | 深度如何读取 |
|---|---|---|
| SDPA/GQA | `[B,H,depth,S,D]` | shape[2] |
| FlexAttention | `[B,Hkv,depth*S,D]` | 缓存长度 // S |
| 标准 FA / FA4 | `[B,depth,S,Hkv,D]` | shape[1] |
| MLA SDPA | K 与 V 各自 `[B,H,depth,S,Dk/Dv]` | shape[2] |

同样的数学链可以有不同存储布局；RoPE 偏移是 depth，不是 `depth*S`。GQA 的 Hq 与 Hkv 不同，MLA 的 Dk 与 Dv 也可能不同。调试 reshape 时先写全维度含义，再看数字乘积是否相等。

### 14. 为什么标准 FA 需要自定义 backward

高效 FA 可以先算首轮因果前缀，链节点另算单键分数。每个分支除了归一化输出，还必须提供对数分母 LSE。

若前缀的未归一化质量是 3，一个新链键质量是 1，则新输出为 `0.75*out_prefix + 0.25*v_chain`。不能对两个分支输出简单相加或平均。代码用 `logsumexp(LSE_i)` 合并总分母，再以 `exp(LSE_i-LSE_total)` 加权。

反向也受总分母影响。链键 score 的导数包含：

`w_i * dot(dout, v_i - out_total)`。

`-out_total` 是概率归一化竞争产生的项，不能漏掉。前缀 FA backward 同样必须收到合并后的 out/LSE，而不是前缀自己局部的 out/LSE。[`_FlashCachedMergeFunc`](../../../torchspec/models/draft/llama3_eagle.py) 就是在显式维护这份契约。GQA 多个 query heads 共享一份 K/V，它们的 K/V 梯度还需要按组求和。

FA4 的自定义 mask 路径则一次处理完整 KV。其 autograd wrapper 必须把同一 mask 传给 forward 和 backward；只在前向屏蔽正确并不能保证训练梯度正确。

### 15. 块稀疏 mask 与编译缓存

[ops/flex_attention.py](../../../torchspec/models/ops/flex_attention.py) 根据 EAGLE 的已知拓扑直接生成块索引。首轮严格下三角块全可见，标为 FULL；对角块与 suffix 对角只部分可见，标为 PARTIAL，需要逐 token 谓词。反向还要从 KV 块反查有哪些 Q 块读取它。

例如 Q block=256、KV block=128，则每个 Q 块覆盖两个 KV 子块。两轮缓存时，每个 Q 块有首轮两个 PARTIAL 块和第二轮两个对角 PARTIAL 块。解析构造避免物化巨大的 token 级 Q×KV 布尔阵，但当前索引数组仍按块级矩形宽度补齐，不是严格按非零项紧凑存储。

[FA4 mask 模式](../../../torchspec/models/draft/llama3_eagle.py) 展示了编译器视角的另一种取舍：

| 模式 | Q_LEN 来源 | 取舍 |
|---|---|---|
| closure/simple | Python 闭包常量 | 每个 bucket 可能不同 kernel，常量计算快 |
| dynamic | aux tensor | 跨长度复用，但额外读取与向量化约束 |
| seqlen | kernel 序列信息 | 无 aux，仍有运行时整数取模 |
| seqlen_po2 | kernel 序列信息，长度补到二次幂 | `diff % Q_LEN` 可用 `diff & (Q_LEN-1)` |

padding 后的轮宽必须和 mask 模数一致；最后只裁输出，缓存继续保留 bucket 布局。预热不仅跑 forward，也跑 backward，避免首个训练 step 才触发反向 JIT。具体性能随硬件/版本/形状变化，阅读源码里的历史耗时不要当成当前设备基准。

### 16. USP 是切序列，不是复制普通 batch

[LlamaUSPFlashAttention](../../../torchspec/models/draft/llama3_eagle.py) 先通过 Ulysses all-to-all 拆 heads、拼 sequence；必要时再通过 ring attention 交换更远的因果前缀。输出 all-to-all 做逆变换，恢复本地序列和完整头。

例如本地 S=256、Ulysses=2、ring=2：Ulysses 变换后一个 rank 看到 512 个位置和一部分头，但 RoPE 必须覆盖全局 1024 个位置。后续同槽链节点留在相应本地分片；ring 主要补齐首轮完整前缀。其 loss count、LK 调度和 collective 顺序都不能简单套用“每卡一个独立 batch”的直觉。归一化与优化器部分继续读 [训练章节](training.md)。

## 用现有测试检验自己的理解

以下是阅读练习，不表示本次注释任务已经在 GPU 上运行这些测试。

| 阅读材料 | 先自己回答，再看测试 |
|---|---|
| [test_eagle3_loss.py](../../../tests/test_eagle3_loss.py) | 为什么 p=q 时 forward_kl 等于教师熵？coverage=0.7 时 alpha 能否到 1？ |
| [test_anchored_eagle3.py](../../../tests/test_anchored_eagle3.py) | N 比 S 小很多时 RoPE 表为什么仍按 S 建？同槽链可见集合是什么？ |
| [test_build_eagle3_block_mask.py](../../../tests/test_build_eagle3_block_mask.py) | 矩形 tile 的 FULL/PARTIAL 前反向索引如何对应？ |
| [test_deepseek_eagle.py](../../../tests/test_deepseek_eagle.py) | MLA 的 K/V 末维为何可不同？Flex 与 SDPA 应比较哪些输出和梯度？ |
| [test_dflash.py](../../../tests/test_dflash.py) | anchor 槽为何不计 loss？标签是同位置还是下一位置？ |
| [test_dflash2.py](../../../tests/test_dflash2.py) | 无序 top-k 为什么要 argmin 找替换项？卷积怎样保证不跨块？ |
| [test_dspark.py](../../../tests/test_dspark.py) | 为什么第0槽有效？L1=0 但开 confidence 时为何仍要教师 hidden？ |
| [test_mtp.py](../../../tests/test_mtp.py) | 输出行 t 的标签为什么是 t+2？Q/gate 权重如何按 KV 组重排？ |
| [test_target_hidden.py](../../../tests/test_target_hidden.py) | 如何用真实张量判定 final norm 边界，避免重复 norm？ |
| [test_eagle4_moe.py](../../../tests/test_eagle4_moe.py) | 路由 bias 改选择还是改混合权重？为何需要 index_add？ |

再做三个纸面检查：

1. **空监督**：一个 SP rank 的有效 token 数为 0，另一个为 10。列出本深度两边必须共同执行的 collective，解释为什么不能提前 return。
2. **标签对齐**：给定原 token `[0,1,2,3,4,5]`、anchor=2、K=3，分别写出 DFlash、DSpark、MTP 的目标。DFlash 是 `[2,3,4]` 且首槽排除；DSpark 是 `[3,4,5]`；MTP 行0的目标为 token2，尾部无目标行排除。
3. **分布重合**：p=[0.6,0.3,0.1]，子词表只保留前两项，q=[0.5,0.5]。正确 alpha=0.5+0.3=0.8；直接比较 q 和重归一化 p 会得到不同结果。

## 模型目录索引与继续阅读

| 文件组 | 职责 |
|---|---|
| [models/__init__.py](../../../torchspec/models/__init__.py) | 训练包装器的公共导出 |
| [eagle3.py](../../../torchspec/models/eagle3.py)、[anchored_eagle3.py](../../../torchspec/models/anchored_eagle3.py) | 多深度训练、目标表示、抽样查询 |
| [dflash.py](../../../torchspec/models/dflash.py)、[dflash2.py](../../../torchspec/models/dflash2.py)、[dspark.py](../../../torchspec/models/dspark.py) | 块构造、标签、损失与指标 |
| [mtp.py](../../../torchspec/models/mtp.py)、[qwen3_5_mtp.py](../../../torchspec/models/qwen3_5_mtp.py) | 原生 MTP 接口及权重导入 |
| [draft/__init__.py](../../../torchspec/models/draft/__init__.py)、[auto.py](../../../torchspec/models/draft/auto.py)、[base.py](../../../torchspec/models/draft/base.py) | 架构分派、统一参数层契约、共享加载器 |
| [draft/llama3_eagle.py](../../../torchspec/models/draft/llama3_eagle.py)、[deepseek_eagle.py](../../../torchspec/models/draft/deepseek_eagle.py) | GQA/MLA、RoPE、缓存、attention 后端 |
| [draft/dflash.py](../../../torchspec/models/draft/dflash.py)、[dflash2.py](../../../torchspec/models/draft/dflash2.py)、[dspark.py](../../../torchspec/models/draft/dspark.py) | 双源 KV、动态卷积、候选/Markov/confidence 头 |
| [draft/eagle4_moe.py](../../../torchspec/models/draft/eagle4_moe.py)、[keymap.py](../../../torchspec/models/draft/keymap.py) | 专家 FFN 与训练/服务键转换 |
| [ops/__init__.py](../../../torchspec/models/ops/__init__.py)、[anchors.py](../../../torchspec/models/ops/anchors.py)、[flex_attention.py](../../../torchspec/models/ops/flex_attention.py) | 固定预算锚点与稀疏 mask |
| [ops/loss.py](../../../torchspec/models/ops/loss.py)、[loss_mask.py](../../../torchspec/models/ops/loss_mask.py) | 编译目标函数与内容监督范围 |
| [target/__init__.py](../../../torchspec/models/target/__init__.py)、[eagle3_target_model.py](../../../torchspec/models/target/eagle3_target_model.py)、[target_utils.py](../../../torchspec/models/target/target_utils.py)、[target_hidden.py](../../../torchspec/models/target_hidden.py) | 教师特征采集、冻结头与 norm 边界 |

理解 tensor 前向后，再到 [训练与优化](training.md) 看累积窗口、分布式规约和保存；到 [推理与传输](inference-transfer.md) 看教师特征怎样跨进程到达 trainer。完整阅读入口见 [阅读指南](README.md)。
