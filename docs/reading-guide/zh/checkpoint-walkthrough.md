# 检查点恢复的四本账：权重、优化器、进度与文件

先假定你熟悉 `torch.save(model.state_dict())`。TorchSpec 的训练检查点还要解决一个额外问题：前向模型与优化器更新的权重不是同一块存储，训练状态又分布在多个进程里。以下按 [checkpoint.py](../../../torchspec/training/checkpoint.py) 的实际实现逐层核对，不把“恢复成功”当作整条异步训练流水线精确重放的保证。

## 1. 一个参数为什么要保存两次

阅读 `OptimizerState.state_dict`，再对照 [BF16Optimizer](../../../torchspec/training/optimizer.py)。取一个标量参数作为例子：

| 状态 | 示例 | 谁使用它 |
| --- | --- | --- |
| 模型参数 | BF16 `1.0` | draft 前向和反向 |
| FP32 master | FP32 约 `1.001` | AdamW 真正更新的参数 |
| Adam 一阶、二阶矩 | `exp_avg`、`exp_avg_sq` | 决定下一次更新的方向和尺度 |
| Adam 参数 step | 500 | Adam 偏差校正 |
| scheduler 状态 | 当前曲线位置与配置 | 下一次学习率 |

`1.001` 转为 BF16 可以舍入成 `1.0`；把 `1.0` 再转为 FP32 得到的仍是 `1.0`。所以模型快照不能重建 master 里尚未跨过 BF16 舍入边界的小更新。Adam 的 state dict 又不代替参数值快照，因此 `OptimizerState` 额外保存 `fp32_params`。

这里 master 按 `named_parameters()` 的名字保存，而非只存列表下标。名字来自可训练参数，顺序须与优化器构造时的 `model.parameters()` 筛选保持一致。这样至少可以明确发现“旧运行冻结了 head，新运行却需要其 master”的缺项；它不代表任意改网络、改参数组都能直接续训。底层 Adam 仍先加载自己的状态，然后才检查并拷贝 master。

## 2. 普通 resume 与 continual training 分别保留什么

普通路径依次执行 `load` 和 `finalize_load`：先读模型，再读 optimizer/master，尝试恢复 scheduler，最后应用 RNG 与进度。模型加载成功后若缺少 optimizer，函数会抛错。原因可以手算：若模型已经恢复成 `1.0`，但 master 仍是初始化时的 `-0.3`，下一步 AdamW 更新的是 `-0.3` 附近的值，再回写到模型，刚恢复的权重就被覆盖。

continual training 则从旧权重开始新的优化过程。`_restore_fp32_master_params` 在成功路径上这样做：

1. 暂存本次新建优化器的参数组超参数，例如 `lr=1e-4`。
2. 临时加载旧 optimizer，借出精确 master；这一步也会顺带载入旧 `lr=2e-5`、step 和 moments。
3. 恢复本次参数组超参数，保留实际参数对象引用。
4. 清空 Adam state，舍弃旧 step 与 moments；新 scheduler 和进度不恢复旧值。

旧 optimizer 不可用时，从已恢复的模型同步 master。这样纠正陈旧 master，但无法恢复 BF16 已丢失的低位。

还有一个需要按源码理解的失败边界：临时加载如果先改变了参数组/moments，之后才抛异常，异常分支只记录日志并同步 master，没有 `finally` 回滚参数组和清 state。因此“保留新配置、清空旧 moments”是上述成功路径的结果，不能无条件推广到所有 fallback。普通 `OptimizerState.load_state_dict` 中若收到缺少或空的 `fp32_params`，也不会主动同步 master；新格式正常应携带这一字段。

## 3. 完成 100 步，目录为什么可能叫 101

沿 [controller/loop.py](../../../torchspec/controller/loop.py) 读一次正常保存：训练 RPC 返回后，控制器先 `completed_steps += 1`，再调用 `save_model(completed_steps)`。`checkpoint.save` 又令 `step_id = step + 1`。因此从零开始完成 100 个窗口时，可能得到：

```text
控制器 completed_steps = 100
actor.global_step     = 100
save(actor, step=100)
目录                   = iter_0000101
meta.step             = 100
meta.next_step        = 101
meta.global_step      = 100
```

`finalize_load` 把 `global_step` 恢复到 actor，把 `next_step` 写到 `args.start_step`。控制器启动时实际通过 `get_global_step()` 获取完成数，再据它计算 epoch 和跳过的样本量。不要只拿目录编号当作“已经完成了多少次更新”。

若 meta 缺失或 JSON 解析失败，兼容分支仅尝试用目录编号填 `args.start_step`，不把它写到 `actor.global_step`；因此这条路径也不能被理解成与新 metadata 等价的恢复。非空 metadata 若缺少 `global_step`，会保留 actor 原值。快照中的 `world_size` 是记录字段，这个文件并未用它执行拓扑一致性校验。

## 4. 三种“已经完成”不要混在一起

从 [RayTrainGroup.save_model](../../../torchspec/ray/train_group.py) 进入，先向所有 actor 提交远程调用，得到 ObjectRef，再统一 `ray.get` 等待。各 actor 内部执行同步 `dcp.save`，没有使用 `dcp.async_save`，也没有 checkpoint 后台 Future。只调用 rank 0 会破坏这里依赖的 collective 参与方式。

`Trainer.save_eval_cache` 才使用线程池 Future：它保存的是 CPU 评估缓存，而不是模型/Adam 检查点。`Future.result()` 等待该线程写盘并传播异常；它不能证明模型检查点已完成。`force_sync=True` 在 `Trainer.save_model` 也只是兼容参数，没有切换到另一种保存实现。

checkpoint 的正常时间顺序为：

```text
每 rank CUDA synchronize
    → rank 0 创建目录 → barrier
    → 全 rank dcp.save(model)
    → 全 rank dcp.save(optimizer)
    → 全 rank dcp.save(scheduler)
    → rank 0 写 rng.pt、meta.json、latest tracker
    → barrier → actor 返回 → driver 的 ray.get 返回
```

这保证正常返回前完成了代码里的同步保存调用，不代表目录树从外部观察始终只有“完整旧版/完整新版”两种状态：

| 中断点 | 可能留下什么 | 读者应如何解释 |
| --- | --- | --- |
| 创建目录后、模型未存完 | 已存在的 model 目录 | `exists()` 不是完整性检查 |
| model 完成、optimizer 未完成 | 可读模型与缺失/不完整 optimizer | 普通 resume 不能安全继续 |
| metadata 发布后、tracker 更新前 | 新目录存在，默认入口仍指旧步 | 显式 ckpt_step 可选择另一目录 |
| 直接覆盖 tracker 期间 | 空串或损坏的步号文本 | `int()` 可能失败，没有自动寻找上一份完整快照 |

`meta.json` 先写同目录 `.tmp` 再 `replace`，保护的是单份 JSON 的发布。tracker 使用直接 `write_text`，RNG 用直接 `torch.save`，三个 DCP 子目录又是分次调用。没有整目录事务、显式 fsync 或并发写者锁；barrier 协调进程到达，不承担这些职责。显式 `ckpt_step` 还会绕过 tracker，因此 tracker 最后写也不是不可绕过的完整性标记。

## 5. 恢复 RNG 不等于复现后续一切

`save` 只让 rank 0 写 `rng.pt`。`torch.cuda.get_rng_state_all()` 的 all 指该进程可见的 CUDA 设备，不是收集所有训练 rank。恢复时各 rank 读取同一份文件。因此它不保存原本可能不同的每 rank 随机流，也没有覆盖 Python `random`、NumPy、推理引擎随机状态、Ray 队列中的样本与预取位置。

此外，Eagle/DFlash 在 `finalize_load` 返回后还会初始化 teacher head。此处不能保证直到第一批训练前再无随机数消耗。控制器按完成窗口数重新提交数据与跳过样本，是另一层恢复安排；这个文件不会还原故障瞬间的异步生产、调度与传输现场。

最后，加载也不是内存事务：模型 `dcp.load` 出错会记录日志并返回 `None`，没有回滚已写参数；scheduler 加载失败只警告。因而验证恢复时应分别核对模型/master、优化器与 scheduler、actor 进度、数据入口，不能只凭一条 “Loaded model” 日志推断后续训练轨迹逐位一致。
