# 从 PyTorch 模型训练读懂 TorchSpec

这份教程面向能读懂 `nn.Module`、attention、autograd 和训练循环，但尚不熟悉推理服务、投机解码和分布式数据传输的读者。阅读目标是能够解释：**一个草稿模型为何接收这些张量、为何采用这种监督方式，以及这些张量为什么要跨进程流动。**

源码中的中文 `#` 注释与本教程配套：函数附近解释局部契约，复杂计算内部解释阶段和索引；教程负责跨文件的关系。原始英文注释和文档字符串保留。遇到与旧说明冲突之处，以当前代码和新增的边界说明为准。

## 1. 先把仓库放进你熟悉的训练框架

你熟悉的监督训练可以写成：

```text
数据 → model(x) → loss(prediction, label) → backward → optimizer.step
```

TorchSpec 的关键变化是 label 不总是现成的类别编号。它还需要一个昂贵目标模型提供中间特征和输出分布：

```text
已知 token 序列 → 冻结目标模型 → 辅助层特征、最后层特征或 logits
                     ↓
同一 token 序列 + 辅助特征 → 可训练草稿模型 → 预测 → 与目标对齐 → backward
```

**目标模型负责提供训练信号；草稿模型才是主要优化对象。** 在某些草稿算法中还会使用真实 token 的硬标签、置信度或其他损失，不能把全仓都概括为同一个 KL 公式。先掌握一条 EAGLE3 路径，再读算法差异。

| 你熟悉的对象 | TorchSpec 中在哪里 | 新增的理解负担 |
|---|---|---|
| 数据集、tokenizer | [data/dataset.py](../../../torchspec/data/dataset.py) | 不同聊天协议、监督 mask、媒体展开 |
| `nn.Module.forward` | [models/eagle3.py](../../../torchspec/models/eagle3.py)、[models/draft/](../../../torchspec/models/draft/) | 多步草稿展开、teacher 特征对齐 |
| loss | [models/ops/loss.py](../../../torchspec/models/ops/loss.py) | 软分布、裁剪词表、分块计算 |
| 训练循环 | [training/trainer.py](../../../torchspec/training/trainer.py) | 累积、跨 rank 归一化和延迟指标 |
| DataLoader | [training/data_fetcher.py](../../../torchspec/training/data_fetcher.py) | 队列描述→远程张量→本地 batch |
| 主程序 | [train_entry.py](../../../torchspec/train_entry.py) | Ray 资源分配、依赖等待和并行初始化 |
| 保存模型 | [training/checkpoint.py](../../../torchspec/training/checkpoint.py) | 分片权重、优化器、FP32 master 与恢复语义 |

## 2. 为什么训练一个“小助手”能帮助推理

普通自回归生成需要不断执行“读取前文→算下一个 token→接回前文”。即使使用 KV cache 避免重算历史 K/V，后一个 token 仍依赖前一个 token 的结果。

投机解码让便宜的草稿模型先提出若干候选 token，再由目标模型验证。若连续候选符合验证规则，一轮昂贵计算能推进多个位置；若早早不接受，额外草稿成本可能抵消收益。实际速度还取决于硬件、batch、模型结构和验证实现。**预测准确率、接受长度、tokens/s 是不同指标。**

这里仅需理解训练动机，不必先实现一套接受/拒绝采样算法。TorchSpec 的主要工作是训练草稿模型；服务时具体的候选验证由接入的推理引擎承担。训练中的多步展开也不能直接当成在线采样过程：它会使用已有 token 和目标特征来构造监督。

两个基础词先记住：

- **Prefill**：一次处理一段已知 token，产生这些位置的状态。训练数据已给出整段回答时，也可以靠 prefill 采集回答位置的 teacher 特征，并不一定先生成回答。
- **Decode**：利用已有前文继续逐步生成新 token。本仓库另有 `train_with_decode` 路径，生产的数据来源和引擎调用与纯 prefill 不同。

KV cache 保存 attention 的历史 key/value；Mooncake 在本训练流水线里主要搬运教师特征。它们可以共用类似的存储技术，却不是同一种数据语义。

## 3. 分四轮读，不要先从启动脚本一路跳到最底层

### 第一轮：证明自己理解一个 batch 的数学意义

1. 看 [utils/types.py](../../../torchspec/utils/types.py)，认清 `InferenceInput`、`InferenceOutput`。
2. 看 [data/utils.py](../../../torchspec/data/utils.py) 的 `pack_loss_mask`、`resolve_loss_mask`、collator。
3. 看 [Eagle3Model.forward](../../../torchspec/models/eagle3.py) 和 [草稿基类](../../../torchspec/models/draft/base.py)。
4. 配合 [模型教程](models.md)，追踪 hidden state、位置偏移、mask 与 loss。
5. 阅读 [test_eagle3_loss.py](../../../tests/test_eagle3_loss.py)、[test_vocab_mapping.py](../../../tests/test_vocab_mapping.py) 的小规模输入和断言。

这一轮先把 Ray、Mooncake 当成“能把正确张量送来”的黑盒。完成标志：能画出 token 位置与教师目标位置的对应，能解释哪些参数有梯度。

### 第二轮：把一个 batch 放回完整训练步骤

按 [训练教程](training.md) 阅读 `Eagle3Trainer`、`Trainer`、优化器和 FSDP。回答：`micro_batch_size`、一次 dispatch、梯度累积、`optimizer.step` 四者分别是什么？哪个分母使不同长度样本和不同 rank 的梯度具有正确意义？

完成标志：给出 DP=4、micro-batch=2、累积=3 时一次更新消费 24 条样本的推导；再解释为什么 USP 模式不能照搬这条公式中的物理 GPU 数。

### 第三轮：解释张量是如何准时、安全地抵达训练器的

按 [流水线教程](pipeline.md) 读 controller，随后读 [推理与传输教程](inference-transfer.md)。先看 HFRunner 理解普通 PyTorch teacher 前向，再比较各高吞吐后端的特征捕获机制。

完成标志：能区分输入预取、结果池、训练队列、存储段、GPU 接收 buffer；能解释“拿到了 view”为什么不一定意味着数据永远归你所有。

### 第四轮：横向比较算法和工程实现

回到 [模型教程](models.md)，依次比较 anchored EAGLE3、DFlash、DFlash2、DSpark、MTP。再选择与你目标模型对应的 draft 架构和 attention 后端。最后使用 [全文件导航](source-map.md) 查缺补漏，并按 [测试与工具阅读路线](tests-and-tools.md) 阅读转换、性能和部署脚本。

完成标志：遇到新的 `forward`，能从输入契约、监督对齐、可见性 mask、归一化和状态生命周期五方面解释它，而不是只复述每条 Python 语句。

### 深入轮：跟着小张量走一次真实实现

完成四轮概念阅读后，按以下顺序检验自己是否能推导中间结果：

1. [监督与损失手算](loss-walkthrough.md)：聊天 mask → 锚点坐标 → gather → CE/KL → 子词表概率质量。
2. [模型逐步演算](model-walkthrough.md)：EAGLE 两轮预测链、DFlash 块与文本位置、BlockMask 前反向索引。
3. [跨进程时序演算](transfer-walkthrough.md)：六 token 请求、物理页地址、异步复制、远端发布、接收区复用。

训练更新衔接见[训练教程](training.md)的完整样本生命周期与梯度窗口手算；源码内进一步说明冻结 head 的输入梯度、加权分母、FP32 master 和预取结束边界。

## 4. 读每个函数时固定问这六个问题

如果已经完成概念阅读，下一步用[监督与损失手算](loss-walkthrough.md)对照真实数组，逐项核对聊天 mask、锚点坐标、gather、CE/KL 和裁剪词表概率质量。不要只记住公式；尝试解释每个中间量在源码中来自哪一行，以及改掉它会破坏什么条件。

1. **谁调用它？** driver、Ray actor、CPU 线程还是 GPU kernel？是否有远程边界？
2. **输入的单位是什么？** token、样本、batch、字节、GPU rank、草稿深度，不要只看 `int`。
3. **张量每个轴是什么？** 先写 `[B,T,H]`，再写发生 reshape/gather/shift 后每个轴代表什么。
4. **返回值归谁拥有？** 独立 tensor、共享 buffer view、未来结果 ref，还是仅描述一个远程对象的 key？
5. **有谁参与梯度或通信？** `detach/no_grad` 切断什么；空样本 rank 是否仍必须参加 collective？
6. **哪个不变量让下一步成立？** mask 与 ID 同长、SP 同组使用同一序列、所有 rank 参与保存、teacher norm 仅应用正确次数。

推荐随手记一张表，而不是复制一大段源码：

| 名称 | 形状/单位 | 生产者 | 消费者 | 梯度 | 生命周期 |
|---|---|---|---|---|---|
| `input_ids` | `[B,T]`，整数 | tokenizer/engine | draft embedding | 无 | 当前 batch |
| 辅助特征 | 通常按层或拼接组织 | target 指定层 | draft 特征投影 | 教师侧冻结 | 存储→本地 batch |
| 最后层特征 | `[B,T,H_target]` | target final norm 边界 | teacher lm_head | 无 | 可延迟求概率 |
| `loss_mask` | `[B,T]`，零/一 | parser/动态 matcher | loss | 无 | 与真实 token 对齐 |
| `mooncake_key` | 字符串 | store writer | data fetcher | 不适用 | 读完后释放 |

具体 layout 以函数为准，表中的“通常”不是所有后端都相同的承诺。

## 5. 三个可以手算的例子

### 监督范围不等于可见范围

假设 token 为 `[用户头, 问题, 助手头, 答案A, 答案B]`，没有 padding 时 attention 有效位置可以都是 1，而 loss mask 为 `[0,0,0,1,1]`。模型仍应看见用户问题，才能学会回答；只是不给问题位置算输出监督。

再补两个 padding：attention 变为 `[1,1,1,1,1,0,0]`，loss 为 `[0,0,0,1,1,0,0]`。因果关系则由 attention 实现里的另一层 mask 表达。三者不要混成一个 mask。

### 为何不总把全词表概率搬到训练端

设 `B=2,T=4096,H=4096,V=128000`，只算一个 BF16 张量：

- 最后层特征：`B×T×H×2` 字节 = 64 MiB。
- 全词表 logits：`B×T×V×2` 字节 = 2000 MiB。

因此传 hidden state 并在训练端按块应用冻结 lm_head，可把通信与常驻中间张量压力换成训练端计算。实际总内存还包括辅助层、mask、buffer 和计算临时量；上面不是显存预算器，也不意味着概率必须用 BF16。

### 词表映射不应根据名字猜

选中目标 token ID `[2,5,9]`，草稿输出列编号为 `[0,1,2]`。当前 `process_token_dict_to_mappings` 返回：

```text
d2t = [2,4,7]        # 差值编码
目标ID = 草稿列 + d2t[草稿列]
t2d[2], t2d[5], t2d[9] = True，其余为 False
```

`d2t` 不是直接 ID 表，`t2d` 不是整数反向表。继续追踪模型如何使用它们，可以避免一种“shape 完全对、语义全错”的隐蔽问题。

## 6. 如何判断自己真的读懂了

先独立回答，再去相应章节核对：

- 输入都给出了，为什么还需要推理引擎？——为了产生昂贵目标模型的状态与监督分布。
- 控制器把 key 从 sample_pool 移走后，数据能立即删吗？——不能据此判断；消费、GPU 拷贝和 SP 分片仍有自己的生命周期。当前 USP 各 rank 读取并清理独立分片 key，基础 key 出账不代表它们都已读完。
- 推理越快训练就越快吗？——到训练消费能力上限后只会增加积压，需要背压。
- token top-1 一致率高，部署就一定快吗？——不一定；连续接受、草稿成本与验证系统都会影响速度。
- 全仓是否只有 EAGLE3？——没有；按真实 trainer/model 工厂选择代码路径，不能把旧文件简介当完整支持矩阵。

[验证说明](verification.md) 记录本注释分支实际做过的检查。静态等价检查证明没有改执行代码；它不等同于在 GPU 集群上验证了全部运行组合。
