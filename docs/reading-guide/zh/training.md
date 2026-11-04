# 从 PyTorch 训练循环到异步分布式蒸馏

这一章沿“取到一个 teacher 样本 → 对齐目标 → backward → 更新 → 保存”阅读源码。你可以保留熟悉的 `forward/loss/backward/step` 直觉，但需要额外追踪三件事：数据来自哪个进程、哪个张量拥有内存、梯度的分母对应哪些 token。

## 第一层：先找到自己熟悉的训练循环

从 [Trainer._train_core_from_queue](../../../torchspec/training/trainer.py) 开始。这里仍然是 PyTorch 训练，只是 `DataLoader` 的上游换成异步 teacher 服务。控制器给训练 Actor 一次 `train_from_queue(step, num_batches)`，其中 `num_batches` 是本次优化器更新的 microbatch 数。每次调用子类 `_train_step` 执行前反向，最后一次才 `optimizer.step()`，然后 `global_step += 1`。

以上描述以取得完整累积窗口为前提。当前实现若遇到队列提前终止，可能没有执行 optimizer 更新，却仍推进 `global_step`，并留下未完成窗口的梯度。排查时应同时检查投递数、实际消费数与 sentinel，不能只看步数日志。

读取时把以下对象分开：

| 对象 | 它负责什么 | 它不代表什么 |
| --- | --- | --- |
| `TrainerActor` | Ray 进程、GPU 与 distributed 初始化、远程接口 | 模型损失的实现 |
| `Trainer` | 数据流、累积窗口、检查点、缓存、性能计时 | teacher 的推理引擎 |
| `Eagle3Trainer` / `DFlashTrainer` | 模型特定的目标构造和统计 | 单独的一套 Ray 调度器 |
| `draft_model` | 真正要训练并导出的小模型 | 包含完整 teacher 的模型 |
| `self.model` | 带训练目标的 wrapper，外面可能有 FSDP/replicate/compile | 一定可直接访问内部字段的普通 Module |

[TrainerActor](../../../torchspec/training/trainer_actor.py) 根据配置类型选择 Trainer，先判更具体的 DFlash2/DSpark，再判 DFlash。Ray 已分配进程并不意味着 PyTorch 进程组已存在；`init()` 才建立 NCCL/CPU backend、USP 和 Gloo 组。

## 第二层：一个 batch 从哪里来

[数据读取器](../../../torchspec/training/data_fetcher.py) 的链条是：控制器给 Ray Queue 放 `TrainSample`，Dataset 根据 `mooncake_key` 读取真实张量，再由 collator 组 batch。Ray 队列中的形状、dtype、packed loss mask 是元数据，不是完整 hidden state。

常见形状约定：

| 字段 | 普通训练 batch | 意义 |
| --- | --- | --- |
| `input_ids` | `[B,S]` | teacher 使用的 token 序列 |
| `hidden_states` | `[B,S,L*H]` | L 个辅助层沿特征维拼接 |
| `last_hidden_states` | `[B,S,H]` | teacher 末层特征，可能为 pre-norm |
| `loss_mask` | `[B,S]`，部分入口允许 `[B,S,1]` | 哪些位置参与监督 |
| `attention_mask` | `[B,S]` | 哪些输入位置有效，与监督区域不是一回事 |

普通 Dataset 会补齐生产端省略的 batch 维；DFlash 随后把 `[B,S,L*H]` 拆成 L 个 `[B,S,H]`。不要把“有效输入 token”和“参与 loss 的 assistant token”混淆：prompt 可能参与 attention，但不承担 loss。

### 内存所有权比 tensor 名字更重要

Mooncake 的读取结果可能是可复用接收 buffer 的视图。对于普通多样本 batch，读取样本 N 后 collator 仍持有它，此时读取 N+1 可能复用接收区，因此代码先 clone。单样本路径保留 pinned 视图，便于后续非阻塞 H2D；这依赖当前消费流程，不能把视图随意保存给更晚的异步任务。

vLLM PP 分层结果更严格。`_load_vllm_pp_layers` 逐层 GET，再调用 `_clone_fragment_before_receive_buffer_reuse`。GPU 上 clone 是异步的，但 GPUDirect 的下一次写入不遵循 CUDA stream 顺序，所以必须等 clone stream 完成。否则“已经调用 clone”仍可能得到下一层覆盖后的数据。所有层 token 必须一致，辅助层才可以沿特征维拼接；末层特征按 manifest role 单独保留。

[PrefetchedDataFetcher](../../../torchspec/training/data_fetcher.py) 用有界线程队列建立背压。开启预取时 inner 在 CPU staging，主线程消费时才搬 GPU。预取线程跨多个 `islice` 训练步存活，None/sentinel 表示结束；后台异常保存原 traceback 并在消费者重新抛出。它不是增加 DataLoader worker 数，也不会替训练器多执行优化器更新。

### 为什么空监督样本不能直接过滤

如果 rank 0 丢弃一条全零 loss mask 样本，而 rank 1 继续 backward，两个 rank 的 FSDP collective 次数或顺序可能错位。实现保留样本，把 loss mask 置零。这样数据没有贡献梯度，但参与同样的分布式计算结构。`min_loss_tokens` 同样使用 neutralize，而不是普通数据清洗中常见的 drop。

## 第三层：teacher 特征如何变成监督

### Eagle：对齐一位，再多步展开

阅读 [Eagle3Trainer._forward](../../../torchspec/training/eagle3_trainer.py)，配合 [padding](../../../torchspec/utils/tensor.py)。这里 `padding(..., left=False)` 不是增加长度，而是把 `[a,b,c]` 变成 `[b,c,0]`。token 与 teacher 末层特征左移一位，辅助层特征留在原始位置，让当前状态与下一 token 的监督对齐。

如果末层特征在 teacher 最后 norm 之前截取，需要先通过冻结的 verifier norm，再乘 teacher head。已经 post-norm 的特征不应再次 norm。初始化时 `last_hidden_states_prenorm` 决定是否加载 norm，前向读取同一状态。

词表裁剪时 `t2d` 选择 teacher 词表中保留的 token，目标概率必须对应 draft 输出维度。全词表路径使用惰性目标，不需要始终物化 `[B,S,V]` 概率。训练 wrapper 返回各展开深度的 loss/accuracy/count/alpha；这里不是在线推理的逐 token 接受/拒绝循环。

Eagle 的反向标量是 `sum(w_i * loss_i) / K`，K 为累积次数，默认 `w_i=0.8**i`。日志 `avg_loss` 另除权重和，所以日志曲线值不等于 `.backward()` 的标量。若两个深度 loss 为 2 和 3，权重为 1 和 0.8，未除 K 的反向量是 4.4，日志加权均值约为 2.444。

### DFlash：block 与 anchor 的统计约定

[DFlashTrainer](../../../torchspec/training/dflash_trainer.py) 将层拼接特征拆开，再交训练 wrapper 采 anchor、构造 block 目标。`uses_target_hidden_states` 决定当前目标是否使用 teacher 末层特征；只有真的使用且输入 pre-norm 时才加载并应用 norm。

DFlash 的位置统计包含第 0 个 anchor 槽，汇总时从索引 1 开始；[DSparkTrainer](../../../torchspec/training/dspark_trainer.py) 把 `_anchor_slot_offset` 设为 0，因此首槽本身就是预测位置。不能把不同模型的原始数组索引直接当成同一预测深度。

[DFlash2Trainer](../../../torchspec/training/dflash2_trainer.py) 复用父类训练流程，替换网络和 wrapper，增加 selector loss。它显式核对网络与训练配置的 block 大小、目标层数。DSpark 则提供 CE、L1、confidence 的分子分母对，父类统一汇总。

### MTP 的另一种偏移

[MTP objective](../../../torchspec/training/mtp.py) 用第 t 行隐藏状态监督 `x[t+2]`。`input_ids[:,2:]` 和 `hidden_states[:,:-2]` 对齐，mask 也取 `[:,2:]`。例如输入 a,b,c,d，前两行监督 c,d，最后两行无目标。空目标返回 `hidden_states.sum()*0`，而非独立的常量零，以保留 backward 路径。可传全局有效目标数作为 normalizer，实现累积窗口统一归一化。

## 第四层：梯度分母与分布式组

### 不等长监督不能简单平均 batch 均值

DFlash 前向提供原始 `(numerator, denominator)`。每个 microbatch 对 numerator/K 做 backward，整个窗口累加 denominator。最后跨 DP 求和总分母 D，再把当前梯度乘 `R*K/D`，R 是归约组大小。原因是此前 backward 除了 K，FSDP/replicate 同步又平均了 R；重标定后得到 `sum(grad numerator)/D`。

例：R=2，K=4，全窗口 D=80。同步后梯度为总分子梯度/8，再乘 8/80，最终是总分子梯度/80。若先对每个 microbatch 的 loss 均值求平均，就会把只有 1 个有效 token 的 batch 与有 100 个有效 token 的 batch 赋同样权重。

全零窗口用 `torch.where` 产生零缩放因子，避免每步把 CUDA 标量读到 Python。这个约定也说明不能只看 `loss.item()` 就推导梯度归一化；应追到 `_train_step` 与 `_normalize_accumulated_gradients`。

### DP、SP、Ulysses、Ring

[distributed.py](../../../torchspec/utils/distributed.py) 建立 USP 二维 mesh `(draft_dp,draft_sp)`，但梯度同步 mesh 覆盖全部训练 rank。不同 SP rank 处理同一个样本的不同序列块，因此要相加局部贡献；不同 DP 副本处理不同样本，才按 DP 平均。REPLICATE 的 USP 通信 hook 按 DP 数除，不是盲目按 world size 除。

U=2、Ring=2 时，一个 SP 组 `[0,1,2,3]` 的 Ulysses 组为 `[0,1]`/`[2,3]`，Ring 组为 `[0,2]`/`[1,3]`。`get_usp_rank_coords` 使用 `%U` 和 `//U`，这是当前布局约定；通信组创建后还会验证成员不越过所属 SP 样本组。

[split_usp_batch](../../../torchspec/utils/usp.py) 的主块长度是 `ceil(S/SP)`，额外读取 `ttt_length` 行作为跨片目标尾部。S=10、SP=2、ttt=2 时，rank 0 取位置 0..6；rank 1 取 5..9 再补两行零。主块归属仍各 5 个位置，额外尾部不代表新增独立训练样本。position_ids 对应 Ring 中 Ulysses 聚合后的主块，长度不一定等于本地输入加尾部的长度。

这个切片工具由生产端 `EagleMooncakeStore.put_usp_shards` 调用。消费端 `_usp_get_sharded_item` 直接读取 `_usp{rank}` key，并拒绝未标记 `usp_sharded` 的样本；不是每个 rank 下载完整样本后再切片。

### 专家并行采用另一种所有权

[Megatron 辅助模块](../../../torchspec/megatron.py) 固定 TP=PP=1，只借 Megatron 建立 EP 与相关进程组。dense 参数在 dense-DP 中归约，专家参数在 expert-DP 副本间归约，两者都除 dense-DP 大小。梯度范数则只对专家部分跨 EP 求和，复制参数只计一次。这里的组和除数不能直接套用 USP。

## 第五层：权重精度、检查点与延迟指标

[BF16Optimizer](../../../torchspec/training/optimizer.py) 持有计算用参数、FP32 master 和可复用 FP32 gradient buffer。每步拷梯度、裁剪、fused AdamW 更新 master、回写模型。非有限梯度标记跨 rank MAX 归约，确保所有 rank 同时跳过参数更新。注意调度器在这条路径上仍推进，跳过 AdamW 不等于整步完全没有状态变化。

[学习率调度器](../../../torchspec/training/lr_scheduler.py) 使用优化器步。WSD=Warmup-Stable-Decay，例如总 100 步、衰减 20 步，最后 80..100 才衰减。constant 分支一直保持峰值；其他曲线超过总步数回到最小值。

[FSDP 初始化](../../../torchspec/training/fsdp.py) 让 rank 0 在 CPU 真实构造权重，其他 rank 在 meta 建形状，再通过完整 state_dict 广播。FULL_SHARD 按子模块再按根模型包装；REPLICATE 保留完整参数只同步梯度。参数计算 dtype 与梯度归约 dtype 是不同设置。

[checkpoint.py](../../../torchspec/training/checkpoint.py) 有三种不同语义：

| 操作 | 恢复什么 | 阅读入口 |
| --- | --- | --- |
| warm start | 发布格式 draft 权重，允许特定新增模块初始化 | `load_initial_draft_weights` / `load_initial_dflash_weights` |
| resume | 模型、匹配 optimizer/master、可选 scheduler、进度与 RNG | `load` + `finalize_load` |
| continual training | 旧模型权重与尽可能精确的 master，新超参数和新 Adam 状态 | `_restore_fp32_master_params` |

只恢复 BF16 模型却保留随机初始化时的 FP32 master 是危险的：下一步 optimizer 会把旧 master 写回，覆盖刚加载的权重。因此普通 resume 缺优化器要报错。continual training 可临时读旧 optimizer 的精确 master，再恢复新参数组并清 moments；失败才从 BF16 权重同步，这是有损兜底。

训练 DCP checkpoint 与 serving 导出不同。后者需要完整 CPU state_dict 和推理引擎期望的 config/参数名；FULL_SHARD 收集失败不能拿本 rank 局部分片冒充完整模型。

Eagle 指标用两个 pinned CPU buffer 和 CUDA Event 延后一轮兑现。[DeferredTrainMetrics](../../../torchspec/training/trainer.py) 保留 CUDA 源引用直到拷贝完成；本步先提交计算，再等待上步指标。第一步可能无返回指标，结束要 flush。`_metrics_step` 指明真实归属步，不能以接收时的步号替代。

准确率按有效 token 加权。例如 1 token 全对、9 token 全错，结果是 0.1，不是 batch 均值 0.5。Eagle 的 vloss 仍是 microbatch/chunk 均值再跨 rank AVG，不能把 accuracy 的分母规则推广到所有日志。`simulated_acc_len = a0 + a0*a1 + ...` 是由位置准确率构造的近似指标，既不是线上实际接受长度，也不能直接换算 tokens/s。

## 第六层：调试时沿哪条证据链走

1. 用 [train_dump.py](../../../torchspec/utils/train_dump.py) 核对输入 token/mask/hidden state，再看各深度 loss、梯度、权重。整数要求相等，浮点用误差容限。目录比较只比较共同文件，缺失文件单独列出；`match=True` 且比较数为 0 不代表验证通过。
2. 用 [profiling.py](../../../torchspec/utils/profiling.py) 区分 CPU 取数等待、CUDA 前反向和 optimizer 时间。CUDA Event 计的是设备时间，不能把 Python 函数返回时刻当完成时刻。
3. 用 [memory.py](../../../torchspec/utils/memory.py) 区分 GPU free/used 与 PyTorch allocated/reserved；张量净字节估算不含 allocator/通信开销。清缓存包含同步，不应当作免费的性能操作。
4. 用 [logging.py](../../../torchspec/utils/logging.py) 的 role/node/rank 日志追踪各进程，结合 [wandb.py](../../../torchspec/utils/wandb.py) 的 train/inference/eval 独立 step 轴理解异步进度。
5. 若 mask 错位，追 [processing.py](../../../torchspec/utils/processing.py) 中 renderer 优先级和 BPE 换行边界；若 actor 环境不同，检查 [env.py](../../../torchspec/utils/env.py) 的白名单与覆盖顺序。
6. 启动环境问题看 [misc.py](../../../torchspec/utils/misc.py)：端口探测不是预留，GPU 物理编号与进程局部编号要分清。跨服务协议看 [types.py](../../../torchspec/utils/types.py)：输出 key 必须配套 shapes/dtypes 使用。

读完后可以自测：为什么空 loss mask 仍需要 forward/backward？为什么 clone 后有时仍需 synchronize？为什么恢复模型不能遗漏 master？为什么 loss 曲线、accuracy 曲线和梯度不一定共享同一个分母？这四个问题串起了训练与推理系统交界处最容易遗漏的约束。

## 一条样本的完整追踪：从远程特征到可重复评估

下面使用缩小的示意形状 `S=6,L=3,H=4`，只为追踪布局；不表示任何真实模型使用这个 hidden size。先把模型数学暂时看成黑盒，每经过一个函数只问“哪些值变了，谁还持有存储”。

1. **生产端完成特征并发布元数据。** 训练侧收到 `TrainSample(mooncake_key=..., tensor_shapes=...)`。普通样本包含 token `[6]`、辅助特征 `[6,12]`、末层特征 `[6,4]`。数据是 teacher 特征，不是训练 draft 的参数；teacher 的前向图不会跨 Ray/Mooncake 传进训练器。
2. **`MooncakeDataset.__iter__ → _load_from_mooncake`。** 后者把 dtype 字符串还原成 dtype，按 key 请求张量。若有 `vllm_pp_layer_manifest`，改调用 `_load_vllm_pp_layers`：每层临时读 `[6,4]`，验证 token 一致，复制接收 buffer 并等待 clone 完成，最后拼三个辅助层成 `[6,12]`。
3. **`_resolve_and_neutralize_loss_mask`。** 例如仅末三个 token 属于 assistant，得到 `[0,0,0,1,1,1]`。若 min_loss_tokens 大于有效监督数，样本仍保留，只把 mask 清零。这个步骤改变监督资格，不改变样本在 collective 序列中的位置。
4. **`MooncakeDataset.__iter__` 补 batch 维。** token/mask 变 `[1,6]`，特征变 `[1,6,12]` 与 `[1,6,4]`。这一层只统一单样本表示，还没有把两个不同长度的样本拼成一个 batch。
5. **`create_mooncake_dataloader` 的 collator。** 若第二条样本长度为 4，普通非 USP 路径会把最大长度 6 向上对齐到 128，得到 `[2,128]` token 和 `[2,128,12]` 特征；第一条后 122 行、第二条后 124 行 padding 都不应成为有效输入/监督。长度分桶用 padding 算力换取较少的编译形状。读取器在 batch_size>1 时先 clone，是因为 collator 要同时持有这两条样本；不能让第二次 GET 覆盖第一条。
6. **可选 `PrefetchedDataFetcher.__next__`。** 后台把已经整理好的 CPU batch 放入有界队列；主线程拿到它才执行 `_to_device`。这里发起的是传输，不是 forward；函数返回不保证 H2D 已完成，后续使用依赖 CUDA stream 的顺序约束。`non_blocking=True` 允许异步执行，但不意味着任意来源都能实现同样的重叠效果，pinned 来源才支持相应快速路径。
7. **`DFlashTrainer._forward`。** `[2,128,12]` 沿末维拆成三个 `[2,128,4]`。是否读取末层 `[2,128,4]` 取决于 objective；若读取 pre-norm 特征，先应用冻结 teacher norm。训练 wrapper 再计算 anchor/block 监督，返回 loss、逐位置统计和 `(numerator,denominator)`。这是数据形状示例，能采出多少有效 block 还取决于真实监督长度和 block 大小。
8. **`DFlashTrainer._train_step`。** 本条 microbatch 的分子参与 backward，分母 detach 后累计。梯度进入 draft 参数；teacher norm/head 不会因此加入 optimizer。返回的统计也 detach，窗口汇总列表不应拖住整张前向图。
9. **最后一个 microbatch 后更新。** 梯度同步和窗口分母归一化完成，公共循环可先 dump，再让 BF16Optimizer 更新 master、回写模型、清梯度。`global_step` 增加一次；本次输入张量随后可随引用释放，而远端 key 已在成功读取后清理。
10. **评估是另一条消费安排。** `Trainer.cache_eval_samples` 把评估 fetcher 产物搬 CPU 缓存，`save_eval_cache` 用单线程快照写盘。`eval_from_cache` 重复使用缓存并按 eval batch 大小重新 collate；它不用再次请求 teacher，也不 backward。缓存保存先写 `.tmp` 再 replace，但列表快照不等于深拷贝所有 tensor，写盘期间不应原地改缓存。

对应源码入口：[读取器与 Dataset](../../../torchspec/training/data_fetcher.py)、[公共 Trainer](../../../torchspec/training/trainer.py)、[DFlashTrainer](../../../torchspec/training/dflash_trainer.py)。如果要改为 Eagle 路径，仅第 7 步替换为 `Eagle3Trainer._forward` 的 token/末层特征左移与多深度目标，不改变上游元数据协议和公共 optimizer 边界。

## 一个优化器窗口的逐步算账

这里用两副本 `R=2`、每副本两个 microbatch `K=2` 举例。为了能手算，假定只有一个标量参数 θ，并令每份原始损失分子是 `c*θ`，分母是固定有效权重。真实神经网络的分子梯度由 autograd 计算，分母也可能包含目标权重；下面只演示归一化代数，不假设每个 DFlash objective 都是未加权 token 数。

| DP 副本 | microbatch | 原始分子 | 分母 | 对 θ 的原始分子梯度 |
| --- | --- | --- | --- | --- |
| 0 | 0 | `6θ` | 2 | 6 |
| 0 | 1 | `3θ` | 1 | 3 |
| 1 | 0 | `8θ` | 4 | 8 |
| 1 | 1 | `0θ` | 0 | 0 |

真实全局池化目标应为 `(6θ+3θ+8θ+0θ)/(2+1+4+0)=17θ/7`，梯度应为 `17/7≈2.42857`。

- `Trainer._train_core_from_queue` 在第一个 microbatch 禁止同步。`DFlashTrainer._backward` 除 K，所以副本 0 第一份梯度为 3，副本 1 为 4。
- 第二个 microbatch 开启同步。副本 0 累积到 `(6+3)/2=4.5`，副本 1 累积到 `(8+0)/2=4`。
- 普通 DP 平均后，每边梯度是 `(4.5+4)/2=4.25=17/4`。
- `_normalize_accumulated_gradients` 把本地分母 3 与 4 求和为 D=7，乘系数 `R*K/D=4/7`，得到 `17/7`。
- `BF16Optimizer.step` 把这个已经正确归一化的梯度复制到 FP32 buffer，再裁剪。若 max_grad_norm=0.5，标量梯度变为约 0.5，日志返回的 grad_norm 仍为约 2.42857。
- AdamW 用裁剪后梯度更新 master；是否包含 weight decay、moment 怎么改变，属于优化器更新本身，不能把上一行的 0.5 当作参数直接减去的量。
- scheduler 前进一步，master 转回模型 dtype，模型参数 `.grad=None`，公共循环 `global_step += 1`。

如果先分别求每个 microbatch 的均值再平均，在这个例子中会得到 `(6/2+3/1+8/4+0)/4=2`，与正确梯度 `17/7` 不同。差异来自监督密度，不是浮点误差。Eagle 使用其自身的多深度 loss 合约，不应把 DFlash 的重标定逻辑直接复制过去。

若本窗口所有分母都为 0，DFlash 缩放系数为 0，梯度归零。AdamW 仍可能因已有的 moments 或 weight decay 改变参数，即使 weight_decay=0 也不能忽略历史动量；“无监督梯度”不必然意味着“所有状态与参数绝对不变”。若梯度非有限，fused AdamW 按 found_inf 跳过参数/moment/weight-decay 更新，但这份封装的 scheduler 仍前进。这两种情形应分别理解。

## 三个时间轴：队列、CUDA 与指标

训练调用返回可以早于 GPU 工作完成。把 Python 的“发出命令”、CUDA stream 的“执行完成”和指标的“搬到 CPU 并可读”分开，就能理解性能代码为何显得比普通脚本复杂。

以 Eagle 为例：

| 时刻 | CPU/控制逻辑 | CUDA/指标存储 |
| --- | --- | --- |
| step 0 提交 | 取 batch，发前反向与更新 | 完成后把打包指标异步拷到 CPU buffer 1 |
| step 0 返回 | 暂无上一步指标，可能返回空字典 | DeferredTrainMetrics 保留源张量与 ready_event |
| step 1 提交 | 先发本步计算 | 本步指标使用另一个 CPU buffer |
| step 1 返回 | 等 step 0 事件并构建其字典 | step 0 源张量引用此时才释放 |
| 训练结束 | `flush_pending_metrics` | 读取最后一份尚未兑现指标 |

双缓冲只解决“当前步拷贝不能覆盖仍待读取的上一份”，并不免除事件等待。读取 CPU tensor 之前必须确认 D2H 已完成。若在前向后随手插入大量 `.item()`，GPU/CPU 会频繁同步，延迟指标原本希望隐藏的等待就重新暴露出来。

计时也有覆盖范围：[公共循环](../../../torchspec/training/trainer.py) 的最后一个 compute Event 结束点在 optimizer 之后，所以 `compute_time` 包含最后一段 optimizer；`optimizer_time` 是细分项，不能再次相加。DFlash 的 forward/backward Event 是另一组更细粒度区间。`data_time` 是主机等待，可能与之前已经提交的 GPU 工作重叠。

## 带答案的源码练习

### 练习 1：位置与维度

输入 token `[10,11,12,13]`，调用 `padding(left=False)` 后是什么？MTP 在这条序列上有几行可监督？

**答案：** padding 得 `[11,12,13,0]`，长度仍为 4。MTP 的候选监督是 token 12、13，对应前两行 hidden state；真正有效行数还取决于 `loss_mask[:,2:]`。不要把 Eagle 的一位移位与 MTP 的两位硬标签偏移混为一谈。看 [tensor.padding](../../../torchspec/utils/tensor.py) 与 [compute_mtp_objective](../../../torchspec/training/mtp.py)。

### 练习 2：准确率的分母

两个 eval chunk 的有效 token 分别为 2 和 8，accuracy 分别为 1 和 0.25。Eagle 汇总准确率是多少？

**答案：** `(2*1+8*0.25)/(2+8)=0.4`，不是 `(1+0.25)/2=0.625`。但不能因此断言该文件所有 loss 也用同一分母：`_aggregate_eval_metrics` 的 vloss 是 chunk 均值再跨 rank AVG。

### 练习 3：跨片尾部

`S=9,SP=2,ttt=2,U=1,Ring=2`，主块长度、本地长度以及第二个 SP rank 的有效输入数是多少？

**答案：** 主块 `ceil(9/2)=5`，本地长度 7。第二片从 5 开始，实际读取 5..8 共 4 行，再补 3 行零。其 attention_mask 前四位为 1；position_ids 对应该 Ring 主块 5..9，共 5 个位置，不是长度 7。看 [split_usp_batch](../../../torchspec/utils/usp.py)。

### 练习 4：检查点里的两个副本

模型 BF16 权重已加载，optimizer master 仍是初始化值。若直接训练一步会怎样？continual training 如何避免？

**答案：** AdamW 更新旧 master 后回写，可能覆盖正确加载的模型。普通 resume 要求匹配 optimizer 状态；continual training 先尝试恢复旧精确 master，再恢复本次新参数组并清 Adam state，不可用时从 BF16 模型同步 master。看 [load、_restore_fp32_master_params、finalize_load](../../../torchspec/training/checkpoint.py)。

### 练习 5：检查一份“全通过”的报告

两个 dump 目录没有同名文件，`compare_eagle3_batches` 的 match 会是什么？

**答案：** 当前实现初始化 match=True，空交集不进入循环，因此仍可能是 True。应同时检查 files_compared 和 files_only_in_dir1/2。这个工具给出数值比较结果，不保证测试样本覆盖。看 [compare_eagle3_batches](../../../torchspec/utils/train_dump.py)。

### 练习 6：为什么 clone 后还要同步

PP 第一层 GPU buffer 调用了 `clone()`，马上 GET 第二层，为什么仍可能错？

**答案：** clone 只是向 CUDA stream 提交复制，GPUDirect 对接收区的写入不受同一 stream 顺序保护。第二层可能在复制完成前覆盖源区，因此 `_clone_fragment_before_receive_buffer_reuse` 等当前 stream 完成后才继续下一次 GET。复制的发起与数据所有权真正转移完成是两件事。

### 练习 7：一次异常跳步是否完全没有变化

一张 GPU 检测到 Inf，其他 rank 没有。谁更新参数？学习率调度器是否前进？

**答案：** found_inf 跨 rank MAX 后大家一致跳过 AdamW 参数/moment/weight-decay 更新；本封装仍执行 scheduler.step 并清梯度。看 [BF16Optimizer.step](../../../torchspec/training/optimizer.py)。日志中的优化器窗口推进与实际成功参数更新次数不能无条件等同。

### 练习 8：模拟接受长度

三个深度准确率 `[0.9,0.8,0.5]` 对应 simulated_acc_len 为多少？能从它直接得到在线 tokens/s 吗？

**答案：** `0.9+0.9*0.8+0.9*0.8*0.5=1.98`。不能；这只是用边际准确率连乘的近似，未测 teacher verify 成本、draft 成本、调度等待和路径相关性。线上性能需要实际推理测量。
