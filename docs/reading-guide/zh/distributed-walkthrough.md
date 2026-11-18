# 八个 rank 到底在共同计算什么

这里把熟悉的单机 `loss.backward()` 拆成三件事：谁持有参数存储、谁持有某条序列的位置、哪些梯度需要相加或平均。阅读入口是 [distributed.py](../../../torchspec/utils/distributed.py) 与 [fsdp.py](../../../torchspec/training/fsdp.py)，调用链经过 [TrainerActor.init](../../../torchspec/training/trainer_actor.py)、[Trainer._setup_device_mesh](../../../torchspec/training/trainer.py) 和 [Eagle3Trainer.init_model](../../../torchspec/training/eagle3_trainer.py)。

## 1. 一张二维表，三种通信范围

设 world_size=8，Ulysses=2，Ring=2，于是 SP=4、DP=2。`init_usp_groups` 的二维 mesh 按 rank 排成：

| 独立数据副本 | SP 位置 0 | SP 位置 1 | SP 位置 2 | SP 位置 3 |
| --- | --- | --- | --- | --- |
| DP 0 | rank 0 | rank 1 | rank 2 | rank 3 |
| DP 1 | rank 4 | rank 5 | rank 6 | rank 7 |

沿行的 `draft_sp` 组共同处理一份序列，沿列的 `draft_dp` 组连接不同数据副本。例如 rank 6 同时属于 SP 组 `[4,5,6,7]` 与 DP 组 `[2,6]`。它的全局 rank 是 6、组内 SP rank 是 2、组内 DP rank 是 1。调用 `get_usp_rank_coords` 应传 2，得到 Ulysses 坐标 0、Ring 坐标 1，不能直接传全局编号 6。

在每行内部，低位 Ulysses 布局进一步拆成：

| 组 | DP 0 中的成员 | DP 1 中的成员 |
| --- | --- | --- |
| Ulysses | `[0,1]`、`[2,3]` | `[4,5]`、`[6,7]` |
| Ring | `[0,2]`、`[1,3]` | `[4,6]`、`[5,7]` |

这些是 attention 的序列通信组。参数梯度另有覆盖 `[0..7]` 的一维 `grad_sync_mesh`，因为同一套参数同时接受不同序列片段、不同样本的梯度贡献。把 attention 的某个局部组直接拿去同步全部参数梯度，会漏掉贡献。

`_build_usp_group_ranks` 只返回上述 Python 列表；真正创建 mesh 和 Yunchang 组的是 `init_usp_groups`。`_validate_usp_group_composition` 再收集实际成员，检查 Ring/Ulysses 不跨出各自 SP 组。它只做集合包含检查，不验证完整的预期大小和排列；通过检查不能证明所有位置布局必然正确。

## 2. 为什么八卡梯度不一定除以八

先固定某个参数 θ，设一个 EAGLE 深度中每个 rank 已经得到局部 loss。`Eagle3Model` 在 USP 下先跨 SP 汇总有效 count，再让每个 rank 的图保留：

```text
local_loss = 本片可微损失分子 / 同一 SP 组总有效 count
```

这个阶段只共享分母，日志用的全组损失分子已经 detach，不能拿来 backward。接下来令八个 rank 对 θ 的局部梯度分别是：

```text
DP 0 的四片：[1,2,3,4]  → 同一样本的总梯度 10
DP 1 的四片：[2,4,6,8]  → 同一样本的总梯度 20
两份独立数据的均值：(10+20)/2 = 15
```

`_allreduce_with_divide_factor_hook` 先把 bucket 除 DP=2，再对全八卡 SUM，得到 15。若直接按 world_size=8 平均则得到 3.75，额外除了一次 SP=4。SP 片段应相加，独立数据副本才平均。

上面的 1..8 是局部 loss 已确定之后的梯度贡献，不是裸 token loss。如果还有 K 次梯度累积，Trainer 的 loss 缩放与最后一次同步是另一层，不能把本 hook 当成所有分母都在这里处理。这个路径先在各 SP 组内得到按有效位置的均值，再平均 DP 副本；若两份独立数据监督数量不同，也不能直接把它描述为“所有 DP token 的全局池化均值”。DFlash 的累积窗口分子/分母重标定是另一份模型契约。

hook 中 `async_op=True` 返回通信工作，Future 把结果交回 DDP 的依赖链。它允许主机继续提交工作，但不是说 bucket 此刻已可随意读取；最终梯度必须等通信依赖满足。

## 3. REPLICATE 与 FULL_SHARD 改的是参数管理

在 `apply_fsdp2` 中，REPLICATE 包装完整模型，让各 rank 各持参数副本并同步梯度。它直接返回，不使用后面 FULL_SHARD 的 `MixedPrecisionPolicy`、`CPUOffloadPolicy` 或 `modules_to_shard`。模型 dtype 已由 trainer 的上游转换确定，不能只看到参数 `cpu_offload=True` 就断言本分支应用了 FSDP 的 CPU offload policy。

FULL_SHARD 则先对指定子模块 `fully_shard`，再包装根模型。子模块划分建立参数聚合/释放的单元，根模型管理剩余参数。它不是“rank 0 算第 0 层、rank 1 算第 1 层”的流水线并行；每个 rank 仍执行模型各层，计算需要时取齐相应参数。Eagle 的调用者选择 midlayer 的 Linear 子模块，DFlash 选择 draft layers，因此也不应概括成固定的“一层一个通信单元”。

参数计算 dtype 和梯度通信 dtype 是两项配置。例如 BF16 前向、FP32 归约并不矛盾：前者降低计算开销，后者减少累加误差。`fsdp_reduce_dtype` 在这里仅对字符串 `bfloat16` 特判，其余值走 FP32；不是任意 dtype 名都能直接映射。

注意实际实现范围：USP 的 SUM/DP 自定义 hook 只在 REPLICATE 分支注册，FULL_SHARD 分支在这个文件没有设置同等梯度除数。Eagle 调用者传 `grad_sync_mesh`，DFlash 调用者传 `dp_mesh`。这些事实不构成任意模型、策略、SP 配置可互换的保证；分析某个组合必须核对模型 loss 和调用者实际传入的组。

## 4. meta 模型只有结构，广播才给它数值

初始化时，`Trainer._get_init_weight_context_manager` 让全局 rank 0 在 CPU 上构造/读取真实权重，其他 rank 用 `init_empty_weights`。默认 `include_buffers=False` 通过临时替换 `Module.register_parameter` 把新参数转成 meta，保留形状和 requires_grad；普通 buffer 不自动全变 meta。参数注册前创建的临时 tensor 也仍可能真实分配，所以“meta 模型没有参数存储”不等于整个构造过程绝对不占内存。

这个替换作用于进程内全局的 Module 方法，`finally` 恢复它；它不是供多个线程同时任意构造模型的隔离机制。`include_buffers=True` 则直接走设备上下文并提前返回，不能把下面遗留的 buffer patch 分支当作它实际执行的路线。

rank 0 在分布式包装前取得完整 `full_state`，其他 rank 传空字典。`fsdp2_load_full_state_dict` 中：

1. rank 0 把已有权重搬到当前 GPU；其他 rank 用 `to_empty` 分配未初始化存储。
2. `set_model_state_dict(...broadcast_from_rank0=True)` 将完整值恢复到包装后的参数布局。
3. 显式广播 buffers，包括可能不在普通 state dict 中的 non-persistent buffer。
4. 如开启本函数的 offload，把模型搬 CPU，再把 buffers 搬回当前 GPU。

`to_empty` 不生成可用的初始化值；不能在第 1、2 步之间就做 forward。这个函数的 `device_mesh` 参数没有在函数体中使用，显式 buffer broadcast 也没有传 group：源是全局 rank 0，通信沿默认组。各 rank 必须以一致顺序枚举兼容形状的 buffer，而不是各自只初始化某个局部 SP 组。

还有一个接口细节：本加载函数按 `cpu_offload is not None` 判定启用，直接传 False 也会触发；现有 trainer 用 True/None 表达开关。`apply_fsdp2` 的同名参数则按布尔值判定，两个入口不能混读。

## 5. 三个可以自检的推导

- rank 6 的 DP 组是 `[2,6]`，为什么不能在这里汇总该序列所有 token 的 count？因为同一序列还分在 `[4,5,7]`，应沿 SP 行汇总。
- 两份样本四片的局部梯度总和为 10、20，为什么八卡不能默认除 8？因为样本数是 2，片段数不是额外数据副本。
- buffer 已广播且 barrier 已到达，是否能断言所有训练组合都有相同梯度尺度？不能。参数初值同步、通信参与范围、梯度分母是三份独立契约。
