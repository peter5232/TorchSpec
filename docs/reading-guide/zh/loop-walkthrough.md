# 从五次供料看清训练循环的边界

先读[流水线总览](pipeline.md)，再打开 [controller/loop.py](../../../torchspec/controller/loop.py)。这一章只跟踪 driver 的 `training_loop`，把远程模型计算当作已知的 `backward/step`。不需要先学会实现 Ray。

## 1. 把四种计数写在不同列

假设 DP=2，每个 DP 副本的 batch_size=2，梯度累积 K=3，prefetch_depth=5。这里的 global_batch_size=2×2×3=12：

| 事件 | 全局独立样本数 | 每个 DP rank 的工作 |
|---|---:|---|
| 一次成功 dispatch | 4 | 得到 2 条样本，构成一个 microbatch |
| 一个完整累积窗口 | 12 | 三个 microbatch，三次 backward |
| 一次正常 train RPC 返回 | 12 已参与本窗口 | 完成一次 optimizer.step |
| 目标预投递 5 个 dispatch | 20 | 部分 batch 属于下一窗口 |

如果再加 SP，同一 DP 样本被分到多个 SP rank，独立样本数不会乘 SP。`queued_batches` 是 driver 记账的 dispatch 数，不是全局样本数，更不是某个队列的 `qsize()`。后台预取线程可能已经把队列元素读到本地内存，而 driver 要等本步训练 RPC 返回后才扣除 K。

源码先计算 `prefetch_batches=max(K,prefetch_depth)`。这保证目标至少能覆盖当前完整窗口，不保证实际数据一定能到达。

## 2. 按一张状态表运行内层循环

假设正常训练尚有多步、pool 中一直有足够样本：

| 时刻 | queued_batches | target | 接下来做什么 |
|---|---:|---:|---|
| 刚开始 | 0 | 5 | 连续尝试 dispatch |
| 成功五次 | 5 | 5 | 内层 while 自然退出，进入 else |
| 第一训练步正常返回 | 2 | — | 扣 K=3，completed_steps 加一 |
| 下轮再成功三次 | 5 | 5 | 第二训练步可以启动 |
| 第二训练步正常返回 | 2 | — | 继续保留已投递的未来窗口部分 |

这个 `while...else` 的 else 不是“失败时执行”：只要 while 条件自然变为 False，就执行 else；若内部 `break`，才跳过它。

另设刚成功投递 3 次，第四次却没有足够样本。已有 K=3 足够当前更新，源码把 target 降成 queued=3，再 `continue`。下一次条件 `3<3` 为 False，立即开算，避免为了额外预取而让训练 GPU 空等。这里对吞吐的取舍是：保证当前窗口完整，同时让下一窗口的准备与当前计算重叠。

当只剩一个训练步或一个 epoch 只剩一步时，target 被限制为 K=3。这个限制约束 driver 接下来投递多少 dispatch；推理端已经提交的请求和已经产生的结果仍有独立生命周期。

## 3. 同时发出所有 rank 的 RPC，再等待

`train_futures` 的列表推导先给每个训练 actor 发 `train_from_queue`，随后统一 `ray.get`。这不是写法偏好：rank 0 的 backward 可能等待 rank 1 进入同一 collective。若先调用 rank 0 并等它结束，才发 rank 1，两个动作会互相等待。

全部 RPC 返回后，driver 扣 K 并增加 completed_steps。此处没有逐个验证 actor 内部实际完成的 backward 数；正常流程依赖完整投递窗口。遇到队列异常提前结束，应继续查看 Trainer 的 `is_last` 和 `global_step`，不能只用 driver 计数推断更新确已发生。

训练 RPC 返回值还可能是上一轮的指标。`_metrics_step` 标明日志归属，`previous_dispatch_wait` 把它与正确的等待时间配对。最终 `flush_pending_train_metrics()` 兑现剩余指标；它不会再训练一次，也不增加 completed_steps。

## 4. 恢复的是训练状态，不是异步队列快照

假设 steps_per_epoch=10、global_batch_size=12、恢复出的 global_step=13：

```text
resume_epoch = 13 // 10 = 1     # shuffle 使用零起点 epoch 索引
resume_skip  = (13 % 10)*12=36 # 在该 epoch 的输入顺序中跳过 36 条
current_epoch = 2              # 日志从 1 开始显示
```

这可以重建一个输入位置估计，但没有恢复此前 prompt buffer、结果池、远程训练队列和本地预取线程的完整快照。原来已发出但未消费的样本可能丢失，重新投递的样本也可能与旧尾部重叠。即使模型权重、优化器恢复正确，也不能承诺逐样本及位级完全复现。

循环还会在持续无法凑批时尝试重载数据。500 次失败是尝试次数，名义上的每次 sleep=0.01 秒之外还有 RPC 时间；它不是精确的 5 秒计时器。该判断查看 pool 和 prompt buffer，没有以 inference manager 的全部在途引用为空作前提，慢请求仍可能随后返回。重载也不清空已经排队的部分累积窗口。

## 5. 验证集先准备，保存点再选择最佳

[setup_eval](../../../torchspec/controller/eval.py) 在训练输入投递前准备 teacher 特征缓存。controller 会将评估集截成 DP 数的整倍数。例如原有 10 条、DP=4，只使用 8 条，每轮 dispatch 4 条，每个 DP 副本缓存一项。原有 3 条则截成 0，评估关闭。

若 inference_batch_size=4，初始提交窗口为 min(8,max(4×2,4))=8。每成功消费一轮，才补下一小块输入。`cache_eval_samples` 返回后更新 last_progress_at；300 秒无进展检测只在 dispatch 失败返回后执行，不能打断一个一直卡着的 RPC。

缓存 key 来自数据源路径、teacher 路径、最大长度和辅助层列表的字符串；同路径内容被改写不会自动改变 key，DP/SP 拓扑也未计入。`all(n>0)` 仅检查每个 rank 成功加载了正数项，不检查这些计数都相等。缓存存在、各 rank 可读、与当前配置语义一致，是三件不同的事。

取 eval_interval=2、save_interval=3：

| 已完成更新数 | 评估与保存 |
|---:|---|
| 2 | 独立评估并记录日志 |
| 3 | 保存分支先评估，再保存并更新 best 元数据 |
| 6 | 独立评估分支跳过，由保存分支评估一次 |

独立评估不会更新 best_checkpointed_iteration。best 只在保存候选中按 simulated_acc_len 严格增大时更新；相等不刷新，也不会自动改用 avg_loss 选优。该指标是模拟评估结果，不是服务端测得的 tokens/s。

loop 在完成第 20 次更新后传 `save_model(20)`，checkpoint 保存函数写 `iter_0000021`，评估元数据也使用 step+1；actor 的 global_step 可以仍为 20。读恢复逻辑时必须查看具体字段。不要从目录名称直接推出该跑多少个新训练窗口。

## 6. 成功返回、部分成功与清理是不同边界

- **保存去重：** last_saved_step 在保存 RPC 正常返回后设置，防止同一步又因 epoch/final 保存一次。它不是历史磁盘目录的完整索引。
- **保留最新：** `_cleanup_old_checkpoints` 按目录步号删旧目录，并不检查 best 指针；保留 best 文件不等于保留其指向权重。
- **decode 权重同步：** 导出后各引擎从临时目录加载。A 成功、B 失败时只记录结果，不回滚 A；若某个 RPC 抛错，其他 RPC 可能尚未结束，finally 却仍会删除临时目录。`mkdtemp` 用系统临时目录；设置共享 output_dir 本身不会改变它的实际位置。
- **退出清理：** 进入训练循环的 try 后，正常返回或异常退出都会调用 `_safe_training_cleanup`；try 前提交 `inference_manager.run.remote()` 若同步抛错，则不经过该 finally。其中 stop/run 等待没有统一超时，若一直不返回，后面的引擎与存储关闭仍无法执行。“进入清理函数”和“资源已释放”不能互换。

阅读 checkpoint 与 controller 细节时，始终保留这张分层图：输入游标 → 推理在途 → 结果池 → 已投递窗口 → optimizer 更新 → 指标兑现 → checkpoint 发布。只有确定一个计数位于哪一层，才能解释它何时增加、由谁减少、异常时是否恢复。
