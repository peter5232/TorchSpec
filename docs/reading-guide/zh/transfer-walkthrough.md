# 用一条六 token 样本核对跨进程时序

这是一次可手算的代码跟踪练习，概念和各后端比较见 [推理与传输](inference-transfer.md)。只跟踪 vLLM connector 的一个请求，刻意把“请求结束”“页可回收”“远端对象齐备”“训练拥有副本”放在不同时间点。

## 固定输入，先算清字节

假设模型 8 层、H=8，捕获槽为 `[2,4,6,8]`，两个 PP 阶段分别执行前四层和后四层，每阶段只由 TP rank 0 发布。请求 `req_a` 的 token 为 `[10,20,30,40,50,60]`；隐状态页长 P=4，完成时页号为 `[2,5]`。这些数值只是阅读例子，不是模型配置建议。

每阶段的 connector 按捕获列表的全局位置选择层。前阶段负责位置 `[0,1]`，后阶段负责 `[2,3]`；槽 8 的角色是 `last_hidden_states`。对应的逻辑输出为：

| 内容 | shape | BF16/int64 字节数 |
|---|---|---:|
| 每个捕获槽的隐状态 | `[6,8]` | 96 |
| 每层配套的 token | `[6]` | 48 |
| 最终辅助层拼接 | `[6,24]` | 288 |
| 最后层表示 | `[6,8]` | 96 |

远端实际有四对 `_layerN_hs`、`_layerN_ids`，共 `4×(96+48)=576` 字节。不是只存一个 288 字节的总 `_hs`；相同 token 也会按层重复存。元数据的最终形状与实际对象布局是两个层次。

## 按实际执行者推进时间线

1. **调度进程接收请求。** `build_connector_meta` 记录完整 prompt，并预先生成 `pp_layer_manifest`。此时清单只是承诺将来有哪些对象，没有任何 GET 成功的保证。`request_finished` 返回这份元数据时，不需要读取 worker 的 Python 字典。
2. **模型 worker 写页。** 隐状态缓存视图是 `[物理页,捕获层,P,H]`。逻辑位置 0—3 在页 2，位置 4—5 在页 5；页 5 的后两个位置不属于本次有效数据。普通 attention KV 页号不能替代此隐状态组的页号。
3. **调度进程确认模型请求结束。** `request_finished` 把 CPU token tensor 和页号列表存入 `_PendingSave`，并返回 `True` 暂缓释放页。复制页号列表不会复制页内容；数据仍在 worker 的原缓存里。
4. **下一次 metadata 移交。** `build_connector_meta` 把 pending 列表移入本步 metadata，然后清空自己的队列。worker 的 `get_finished` 从这份 metadata 取任务；回调参数里的完成 ID 集合在这里被丢弃。
5. **worker 解除原页依赖。** 下面两条发布路线在这一点有不同的完成边界。走哪条由注册能力和配置决定，不能把二者的等待条件混用。
6. **各 PP 阶段汇合。** 本地发布异常先缓存，参与 PP barrier 后才抛出；正常路径返回允许回收的请求集合。这个 barrier 不等待其他线程中的所有异步 put，也不会把本地异常自动变成每个 rank 上的同一 Python 异常。
7. **训练进程逐层接收。** `MooncakeDataset._load_vllm_pp_layers` 按清单读每层 `[6,8]` 和 `[6]`，核对 token 一致性，再沿最后维拼接前三层。每次下一层 GET 前先保存当前层独立副本，成功收集后再删除远端片段。

## 第 5 步展开：原页到底何时可复用

**直接分页发布：** `_cache_layer_fragments` 计算每层两个源片段，长度为 `[64,32]` 字节，顺序是页 2 再页 5。若该缓存连续且 L=4，元素 stride 为 `[128,32,8,1]`，捕获位置 1 的两片地址分别为 `base+(2×128+1×32)×2` 与 `base+(5×128+1×32)×2`，即 `base+576` 和 `base+1344`。两个远隔的物理范围拼成连续的 96 字节层对象。

```text
模型写缓存 → cache_ready.synchronize → 网卡读原页 → 同步多源 put 返回
                                                       ↓
                                                原页不再是网络源
```

token ID 仍由普通 staging 发送。隐藏层 put 返回不意味着 ID 键已经可见；正常的 `get_finished` 返回也不代表整条样本所有对象都已发布。

**Gather 后 staging：** slot mapping 为 `[8,9,10,11,20,21]`，高级索引物化 `[6,4,8]`。这是独立于原页的显存；切出一个捕获层后得到 `[6,8]`，但 stride 常为 `[32,1]`，因为两个相邻 token 的同层中间隔着其余层。`HostBuffer.copy_from_tensor` 内部用 `contiguous()` 生成按层连续的字节顺序。

```text
当前 CUDA 流：gather/精度转换 → compute_event
复制 CUDA 流：                   等待事件 → 连续化/DtoH → copy_done
后台 CPU 线程：                                           等事件 → put → Future 完成
```

源张量由 `record_stream` 保护分配器复用；主机缓冲由基址对应的 Future 保护网卡读取。原页后续复用与 gather 的读取依赖 vLLM/CUDA 流的执行顺序，不能推断成 CPU 函数返回时所有 GPU 操作都已完成。

## 在同一例子中制造三种故障

**容量只允许 160 字节。** 一层 hs=96、ids=48，合计 144，下一层 hs 放不下，于是先提交当前组。如果容量只有 100，则 hs 与 ids 会分批；单张量 96 仍放得下。如果容量是 80，单个 hs 已超限，分成更多批次也无济于事，connector 直接报错。分批单位是完整对象，不是任意字节切片。

**第一批成功，第二批失败。** 错误处理清理失败调用涉及的键；没有跨所有请求、所有层和所有 put 的回滚事务。先前批次的对象可能仍存在。`wait_for_keys` 发现缺字段时继续等或超时，不应把剩余片段送入训练。若清理本身失败，日志与异步异常是排查证据，不能假设远端已经空了。

**第一层读对，读第二层后第一层变了。** GPU Direct GET 使用同一接收缓冲，`view(dtype).reshape(shape)` 不复制。`clone()` 虽分配新显存，复制本身仍是 CUDA 异步操作；下一次网络 GET 不受它的 stream 顺序约束。训练侧 `_clone_fragment_before_receive_buffer_reuse` 会等待 clone 所在流后才继续下一次 GET。删除远端对象与保住本地接收视图是两件事，删除不能解决接收区被覆盖。

## 回到代码核对

- [connector](../../../torchspec/inference/engine/mooncake_hidden_states_connector.py)：从 `request_finished` 到 `build_connector_meta`、`get_finished`，跟踪同一 `_PendingSave`。
- [Eagle store](../../../torchspec/transfer/mooncake/eagle_store.py)：看 `_put_raw_tensors` 如何把同一批对象交给同步发送区或异步主机池。
- [缓冲和 Future](../../../torchspec/transfer/mooncake/buffers.py)：核对 `submit` 用基址登记，`_do_put` 先等事件再加 put 锁，`drain` 遇错仍等待其他任务。
- [存储连接](../../../torchspec/transfer/mooncake/store.py)：核对原始地址注册与 `wait_for_keys` 的存在性检查；后者不提供对象锁或事务快照。
- [训练读取](../../../torchspec/training/data_fetcher.py)：核对逐层 clone、token 一致性、层角色、最终拼接及删除时机。
