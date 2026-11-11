# 四条样本如何越过控制器、队列与训练边界

本例只跟踪控制面。张量字节传输的 CUDA/RDMA 时序见 [传输推演](transfer-walkthrough.md)。这里的“条数”默认指独立逻辑样本，不是 token 数、SP 分片数或 optimizer step 数。

## 先给每个容器确定计数单位

| 位置 | 里面是什么 | 何时移出 |
|---|---|---|
| controller `prompt_buffer` | `InferenceInput`，待推理输入 | `get_prompts` 直接 pop |
| manager `_prompt_buffer` | 已预取但未提交的输入 | `_take_batch` |
| manager `_pending_tasks` | 一批一个 asyncio Task | collect 从集合移走，不是 GPU 完成时自动移走 |
| controller `sample_pool` / `eval_pool` | 推理回包的 key、shape、mask 等元数据 | 完整 dispatch 出池 |
| `train_queues` / `eval_queues` | 按消费者分配的 `TrainSample` | 训练侧 queue.get |
| Mooncake | 真正张量对象及分片 | 消费者的存储清理协议 |

前两处输入也可能包含 token tensor；“控制面”不意味着禁止传小张量。第三处即便已经得到完整回包，只要尚未收集仍计入 pending。最后三处不能合成一个池长：从 sample_pool 移进训练队列后，远端字节往往仍在。

## 一批四条，DP=2、SP=2

设一次训练 dispatch 要求每 DP 两条，因此 `dispatch_batch_size=4`，队列数是 `DP×SP=4`。四条样本 A/B/C/D 的长度分别为 8/7/2/1。

1. manager 可以用不同的推理批次得到它们。比如 `[A,B]` 比 `[C,D]` 晚完成，控制器可能先收到 C/D。等待 `FIRST_COMPLETED` 不会为了恢复数据集顺序而扣住快批；同时完成的任务存在集合里，也没有稳定的提交顺序承诺。
2. `push_inference_results` 按 data_id 是否在评估集合中分流。训练项进入 sample_pool，每条按逻辑 shape/dtype 增加字节估算。这里不下载 hidden states，也不检查每个远端对象此刻都能 GET。
3. 池里够四条时，`try_dispatch_batch` 先整体取出四条、扣减字节账本，再在锁外发送队列。长度贪心排序对本例得到 DP0 `[A,D]`、DP1 `[B,C]`，累计长度都是 9。它平衡的是 token 总数，不是实际 GPU 时间或 attention FLOPs。
4. 队列 0、1 各收到 A/D；队列 2、3 各收到 B/C。同一 DP 组内的两个 SP rank 必须看到相同样本描述，才可协作处理对应分片。四条独立样本产生八次元数据入队，不等于把训练数据量翻倍。
5. 当前 USP 消费路径从同一基础 key 派生各自的 `_usp{sp_rank}`，读取各自的分片。它们不是从同一个共享队列竞争抢走四条样本，也不是 controller 在这里复制大张量。

假设梯度累积为 3，外层 `controller.loop.training_loop` 至少备好 3 次这样的 dispatch，才足够一次 optimizer step：总计 12 条独立样本，每个 DP 处理 6 条。同一 DP 的 SP 协作并不再乘一次 global batch。控制器的 `batch_id` 只在 dispatch 后加一；它不是 optimizer step 计数。

## 为什么阈值 8 仍可能超过 8

`max_sample_pool_size` 查询的是控制器当前 train+eval 池长，没有预留在途批次的输出名额。若当前池长 7、推理批长 4，`7<8` 通过后四条结果仍可整体加入，变成 11；同时在途的其他批次也可以继续完成。这是暂停新工作的水位，不能当成远端对象总数或字节硬上限。

另一个独立水位控制 manager 输入预取：水位 32、现有 30、`inference_fetch_batch=16` 时，实际请求 `max(2,16)=16` 条，未发生并发消费时返回后是 46。它表达低于水位时补货，不表达固定容量为 32。

DP/SP 队列在控制器里以 `Queue()` 创建，没有显式 maxsize。正常训练由外层 `queued_batches` 和 prefetch 节奏限制分发；同一字段在不同层不能混读：

- `get_pool_size` 包括待分发 eval，排除训练队列、在途请求和未转发回包。
- `get_pool_bytes` 只包含待分发训练池的逻辑估算，不包括 eval。
- 字节估算不枚举 PP 重复 token 对象、USP 重叠分片、复制副本和原生分配开销。
- `training_speed` 统计成功分发条数，不能直接当作 backward/optimizer 的实测速度。

## 一条失败会在哪个时刻被谁看见

**普通批次异常：** `_dispatch_batch` 的远程 generate 抛普通 Exception，转成每个输入对应的异常结果，`_forward_results` 跳过这些条目。它不把输入重新放回队列。若只是一个输出是 `None`，仍先按原位置 zip，再过滤；先过滤会把后续样本的 data_id/mask 错配。回包条数不符则整批标为失败。

**引擎 actor 死亡：** manager 先设置 `_running=False`，等待 controller 记录致命错误，再重新抛出。collect 中的 `task.result()` 因而抛错。训练端下一次 `try_dispatch_batch` 在检查池长之前就报错，即使池里尚有完整旧批次也不会继续分发。评估分发没有同样的错误槽检查，评估缓存调用方另有空闲超时。

**不要把普通失败包装理解为完整异常兜底：** 参数准备和选引擎在 generate 的 try 之外；转发结果 RPC 也可能抛错。`run` 的取消补货与 drain 写在 while 后面，没有 finally 保护，因此异常直接退出与正常 `stop()` 的收尾路径不同。正常 stop 也只是协作式标志：容量等待返回之后的提交处没有立即重查标志，不能承诺发出 stop 后不会再提交任何一批。

**队列写入中途失败：** 四条已从 pool 移除且字节出账，然后逐队列 put。若某次 put 失败，之前的队列可能已收到样本；没有自动把全部条目退回 pool 或从先前队列撤销。`True` 表示本次队列发送代码正常完成，不表示训练已消费，更不表示 Mooncake 已释放。

## 评估优先级与 epoch 的边界

`submit_eval_chunk` 把评估输入放到控制器 prompt 队头，只能优先于还没被 manager 预取的输入，不能抢占 manager 本地训练输入或 GPU 在途工作。`generate_eval_cache` 消费一批后再补一小块，控制的是实际生产消费节奏，而非只依赖池阈值。

重新提交训练 epoch 是向现有 prompt_buffer 追加数据，没有清空旧输入、结果或训练队列。确定性的 seed+epoch 可以恢复输入排列，不能恢复异步完成顺序。评估 finalize 清除的是跟踪状态；若有尾部元数据被丢弃，这个动作本身没有删除其 Mooncake 对象。

## 对照入口

- [推理管理器](../../../torchspec/controller/inference_manager.py)：`run` → dispatch/collect → `_dispatch_batch` → `_forward_results`。
- [训练控制器](../../../torchspec/controller/training_controller.py)：`get_prompts` → 入池 → `try_dispatch_batch` → `_partition_results` → `_dispatch_to_queues`。
- [训练外层循环](../../../torchspec/controller/loop.py)：看 dispatch 计数怎样与梯度累积、预取及 optimizer step 对应。
- [队列绑定](../../../torchspec/ray/train_group.py)与[数据读取](../../../torchspec/training/data_fetcher.py)：看队列如何落到训练 actor，以及同组 SP 如何解释基础 key。
- [评估缓存](../../../torchspec/controller/eval.py)：看实际补货窗口、消费及超时。
