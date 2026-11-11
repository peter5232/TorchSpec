# 跟踪一条离线样本：从教师输出到再次训练

本文按实际字段追踪物化和回放，不需要先理解 Ray 的内部实现。先把 Ray actor 看成独立进程中的对象：`.remote()` 提交工作，`ray.get()` 等待结果。Mooncake 暂存大张量；Ray 队列传递这些张量的位置和形状。相关入口是 [materialize](../../../torchspec/offline/generate.py)、[OfflineSavingActor](../../../torchspec/offline/saving_actor.py)、[OfflineDataset](../../../torchspec/offline/dataset.py) 和 [OfflineReplayEngine](../../../torchspec/inference/engine/offline_replay_engine.py)。

## 固定一份样本，区分三个名字

设教师已经处理好一条样本，`data_id="train-1"`，长度为 4。为方便纸面追踪，假设教师宽度为 2，捕获三个辅助层：

| 字段 | 形状/值 | 用途 |
|---|---|---|
| input_ids | [1,4]，int64 | 真实 token |
| hidden_states | [1,4,6]，BF16 | 三路辅助特征沿最后维拼接 |
| last_hidden_states | [1,4,2]，BF16 | 训练时经教师 head 生成监督分布 |
| target | None | 此例没有预存 logits |
| packed_loss_mask | "2,2" | 此例的原始打包 mask 字符串，保存器不解码 |
| metadata | {"source":"example"} | 随样本保存的小字典 |

这些数值尺寸用于说明字段，不要求真实模型这么小。最后层是 pre-norm 还是 post-norm 由整份数据的 `dataset.json` 记录；张量宽度本身不能区分两者。

样本在生成时临时发布到 Mooncake key `K1`。落盘文件名则是 `samples/sha1("train-1").pt`。之后回放产生新 UUID `K2`：

```text
稳定样本 ID:     train-1      train-1      train-1
传输对象 key:   K1           K2           K3
阶段:           物化时       训练 epoch 0  训练 epoch 1
磁盘记录:       同一份 samples/<ID哈希>.pt
```

`data_id` 用于去重和检索；传输 key 用于一次对象的生命周期。不能为了“稳定”而复用传输 key，否则上一轮消费者清理对象时可能影响下一轮。

## 物化驱动循环到底数什么

[materialize](../../../torchspec/offline/generate.py) 创建真实 teacher 引擎、控制器和 CPU 保存 actor，不创建草稿训练进程。控制器 DP 大小为 1，`per_dp_rank_batch_size=1`：假设源数据五条、teacher 每批两条，推理可以分成 2+2+1，而保存器能完成五次单条消费，尾部不会因为不足训练 batch 而丢弃。

引擎产出后，[inference manager](../../../torchspec/controller/inference_manager.py) 把 `mooncake_key/tensor_shapes/tensor_dtypes/packed_loss_mask/metadata` 与原输入 ID 组成输出。控制器再将 TrainSample 描述符放入保存器对应 split 队列。

驱动先 `try_dispatch_*`，成功后等待 `save_from_queue` 返回，再增加 `processed[split]`。因此有三种不同数量：

- `expected`：本次加载的源数据条数。
- `processed`：本次成功完成保存调用的队列条目数，包括磁盘已有的 ID。
- `dataset.count`：已有清单与本次新增合计的唯一记录数。

若旧目录已有 `train-1`，这次又处理它，append 返回 False，但 processed 仍增加一。此入口没有把已有 IDs 从源数据中过滤掉，通常仍会重新运行 teacher；这是去重续写，不是保存并恢复 GPU 的执行进度。

## 保存器依次跨过哪些边界

跟踪 [save_from_queue](../../../torchspec/offline/saving_actor.py) 的一轮：

1. `get()` 从队列取走 TrainSample，检查有 data_id，并拒绝 vLLM PP 分层 manifest。这种 PP 结果不是一个普通基础 key 能读出的完整记录。
2. 将可能是字符串的 dtype 还原成 `torch.dtype`，按描述符从 Mooncake 拉取 CPU 张量。这里得到原尺寸 [1,4]、[1,4,6]、[1,4,2]。
3. 调用 dataset.append，同步完成本条文件和清单写入；之后才处理下一条。张量即使来自可复用接收缓冲，也不会被下一次 get 在本次 torch.save 期间覆盖。
4. get 成功后的 finally 删除本次 Mooncake 对象。可选字段根据 tensor_shapes 判断，因此不会假定每条样本都有 target/lhs。

注意 finally 的范围：get 在 try 外，get 自身失败不会执行这里的删除；append 失败却会执行删除。队列条目已经取走，这条路径没有把它重新塞回队列。重新生成需要重新提交源样本。

## `append` 的发布顺序与恢复边界

进入 [OfflineDataset.append](../../../torchspec/offline/dataset.py)，同 split 重复 ID 会立即返回 False。它不比较新旧张量，不更新旧 mask/metadata；跨 split 重复 ID 则报错。

新记录仅保留四种允许的张量名并转成 CPU 连续张量。`loss_mask` 或 `input_ids_cpu` 即使出现在 `to_tensor_dict()` 中也不会写入；mask 另以 packed_loss_mask 保存。可选 target=None 不占一个空张量字段。

然后按此顺序发布：

```text
同目录临时文件 --torch.save--> 完整 .pt 内容
              --os.replace--> samples/<ID哈希>.pt
              --append/flush/fsync--> manifest.jsonl 一行
              --> 更新本进程 _rows 与 _by_id
```

清单行包含 `split,data_id,file,seq_len`。本例 seq_len 是 `input_ids.numel()=4`，并非读取 hidden 的某一维；不要将 B>1 的整个 batch 当成一条样本追加。

| 中断位置 | 重启时能观察到什么 |
|---|---|
| 最终 .pt 发布前 | 清单没有该 ID；正常异常收尾会清理临时文件，进程硬退出不保证执行 finally |
| .pt 已替换，清单未成功提交 | 可能留下孤立 .pt；构造器不扫描它，后续同 ID append 可覆盖它 |
| 清单末尾只有损坏的 JSON | 读取直接报错，没有跳过残行或自动修复 |
| 清单正常引用一个缺失文件 | 构造器检查路径时失败 |
| 完整记录已提交 | 从 manifest 重建索引；同 ID append 去重 |

这不是断电事务保证：样本文件与父目录没有显式 fsync，只有清单做了 fsync；原子替换名字不等于三个文件都具有一致的掉电持久性。哈希由 ID 而非张量内容生成，也不是内容校验和。

## 回放如何重新进入普通训练消费路径

离线训练启动时，[controller._load_dataset_split](../../../torchspec/controller/training_controller.py) 只从清单建立 `data_id/seq_len/metadata` 条目，不读取所有 hidden。正常训练调度仍可打乱、按长度分组，再把一批 ID 交给 replay engine。

[OfflineReplayEngine.init](../../../torchspec/inference/engine/offline_replay_engine.py) 加载清单快照，建立 `ID -> row`。它不会监听随后追加的 manifest；正常使用是先物化完成，再启动训练。

假设 generate 请求 `['train-1','train-2']`：

1. 按请求顺序 mmap 读取每份 .pt，验证记录 ID 与两项必需 tensor 的类型。此处不校验所有时间维/hidden 宽度是否符合当前模型。
2. 为每条创建新 UUID 并调用 store.put。传输层仍会按契约将 hidden/target 转为 BF16；结果描述符取 put 返回的实际 shapes/dtypes。
3. 所有 put 后调用 flush；只有它正常返回，输出才交回 manager。请求中的新 token、mask、return_logits 等参数不改变磁盘记录。
4. manager 按位置 zip 输入与输出，以原输入 ID 补回身份。因此返回顺序必须与请求一致，不能擅自按文件名或 manifest 重排。
5. 控制器和训练端继续使用普通队列/Mooncake 消费流程；teacher 本身没有再次执行。

若第二条缺失，第一条可能已经 put。此 generate 不回滚已发布对象，整批失败也不等于完全没有副作用；它没有实现跨样本事务。

## 数据格式保证到哪里

`dataset.json` 校验 schema version，记录并覆盖运行时的 pre/post-norm 参数。它没有记录教师权重、tokenizer 或辅助层配置的完整指纹，不能据此证明任意训练配置兼容。

裁剪词表另存 `vocab_mapping.pt`。[训练入口](../../../torchspec/train_entry.py) 在需要物化映射且未选择保留初始权重映射时加载它；续写物化会按本次源数据重新计算并替换映射，没有与旧映射逐项比对。因此同目录续写应保持源数据与模型语义，不要把“ID 已去重”理解成“不同配置的数据能安全混写”。

[现有测试](../../../tests/test_offline_replay.py) 覆盖读写、重复 ID、可选字段以及回放接口等行为。本文是源码演算；注释改动的 AST/token 等价验证不代表真实 Ray/Mooncake 服务、磁盘故障注入或断电恢复测试已经运行。
