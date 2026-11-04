# 从 teacher 前向到可训练样本：推理与传输

本章面向已经写过 `model(input_ids)` 和训练循环，但还没接触推理服务的读者。先沿 Hugging Face 路径理解一次前向，再看其他后端怎样在调度、分页缓存和跨机器传输中保留同一条样本的含义。实现索引列在文末；注释描述当前代码，不能据此假定任意版本的第三方后端都兼容。

## 1. 先分清这里的“生成”是什么

训练 draft 需要 teacher 的辅助层输出、token，以及用于构造目标分布的信息。最直接的方法是在完整已知序列上跑 teacher，取出中间张量。这个过程不需要反向传播，也不一定生成任何新 token。

- **Prefill**：给定长度为 S 的已知序列，计算这些位置的模型输出。在一般推理系统里也会填充 attention 的 KV cache。TorchSpec 用它物化训练数据。
- **Decode**：已有 prompt 后继续逐步生成 token。SGLang 的 `generate_with_decode` 会产生 completion，再构建训练数据和 mask。
- **Speculative decoding**：小 draft 提议多个 token，teacher 验证并决定接受多少。在线接受率是推理性能指标；训练代码的多步展开与 loss 是另一个过程。

不要从函数名 `generate` 推断语义：[抽象接口](../../../torchspec/inference/engine/base.py)定义的是数据生产者。HF 的实现做已知序列前向，SGLang 普通入口设 `max_new_tokens=0`，vLLM/TRT-LLM 用一个生成 token 满足后端捕获生命周期，但本章这些入口返回的训练长度仍对应 prompt。真正的 completion 数据路线见 [SGLang decode mixin](../../../torchspec/inference/engine/sgl_engine_decode.py)。

## 2. 最接近 PyTorch 的 HF 路径

从 [HFEngine.generate](../../../torchspec/inference/engine/hf_engine.py) 开始读：它接收张量列表或 `ray.ObjectRef`，必要时 `ray.get`，然后转交 [HFRunner](../../../torchspec/inference/engine/hf_runner.py)。Ray actor 是远程进程中的 Python 对象；调用 `.remote()` 返回任务引用，并不表示远端已执行完成。

`HFRunner.generate` 处于 `torch.no_grad()` 下，把输入送到 CUDA，再进入 `_run_inference`：

1. 单条 `[S]` 补成 `[1, S]`。
2. 每条单独调用 `target_model.generate_eagle3_data`，没有把不同长度样本堆成一个 padding batch。
3. 收集 `input_ids`、`hidden_states`、`target` 和 `last_hidden_states`。
4. 有 Mooncake 时逐条写入，最后 `flush()`；无存储时直接返回张量。

这里传给目标模型的 attention/loss mask 都是全一，原始 `packed_loss_mask` 作为元数据留给训练侧。因此“teacher 看哪些 token”和“训练在哪些位置计 loss”需要分开理解。真正的层捕获与最后归一化位置应继续看 [HFTargetModel](../../../torchspec/models/target/eagle3_target_model.py)，不要只凭字段名称推断。

`HFEngine` 是部署外壳，`HFRunner` 是计算与发布流程。两者分开使模型可以在 actor 到达正确节点后才加载。构造器中先设置 `_engine=None`；昂贵的加载发生在 `init`，不是驱动端构造对象时。HF 接口接受一些兼容参数但没有使用它们，支持范围以实际实现为准。

## 3. 一条样本的两个通道

普通 Eagle 样本可以由如下字段组成，实际 shape 以写入后的元数据为准：

| 字段 | 常见形状 | 含义 |
|---|---|---|
| `input_ids` | `[S]`，HF 路径也可能保留 `[1, S]` | token 编号，通常 int64 |
| `hidden_states` | `[S, K×H]` | K 个辅助层沿特征轴拼接 |
| `last_hidden_states` | `[S, H]` | 最后层表示；是否已过 norm 由后端契约决定 |
| `target` | 常见为包含词表轴的张量 | 可选 teacher 目标；不能假定所有后端都传它 |

K=3、H=4096 时，`hidden_states` 的特征轴是 12288，不是把样本数变成三倍。一个样本单独存储时通常没有 batch 维，训练 collator 才负责组装 batch。

[存储协议](../../../torchspec/transfer/mooncake/eagle_store.py)把逻辑键 `sample_x` 拆成 `sample_x_hs`、`sample_x_ids` 和可选 `_lhs`、`_tgt`。元数据含 `mooncake_key`、`tensor_shapes`、`tensor_dtypes`、mask，必要时还有逐层清单。

- **Ray 控制通道**承载任务、引用和小型结果字典；在部分入口也传输入 token。
- **Mooncake 数据通道**承载大型张量字节。它不是把 GPU tensor 对象原封不动送给另一个 Python 进程；消费者用 shape 和 dtype 解释收到的原始内存。

因此 dtype 不是可有可无的描述。`put` 会把隐状态、最后隐状态及可选 target 转成 BF16，再返回实际精度；若用 FP32 元数据解释这些字节，读取端会要求两倍长度。`put_raw_tensors` 则假定调用者已经规范好精度，不做同样转换。

## 4. 从一个引擎到多个节点

[factory.py](../../../torchspec/inference/factory.py)有两类入口：`create_inference_engines` 阻塞到就绪；`prepare_inference_engines` 返回 `(head_engines, init_refs)`，允许上层同时初始化训练端，之后再等待引用。

Ray placement group 预先规划资源位置，工厂通过重排后的 bundle 索引与物理 GPU ID 把 actor 放到对应节点。这里出现的 `num_gpus=0.2` 是 actor 的调度占位，不能解释为大模型实际只需五分之一张 GPU；真正设备范围由资源规划和后端子进程管理。

例子：总共 32 张卡，每节点 8 张，一个 SGLang 副本跨 2 节点，则每副本 16 卡，总计 2 个副本、4 个 actor，但只有 2 个 head 作为请求入口。worker 虽不接 `generate`，仍参加 TP 通信；工厂用模块级引用保活，防止 Ray 回收它们。

TP 把模型张量计算分到多卡；PP 把不同模型层放到不同阶段。对 vLLM，两节点各 8 卡、PP=4 时得到 TP=4，因为 `TP×PP=16`。这仍是一份模型副本，不能当成 16 个互不关联的样本 worker。

## 5. 各后端如何取得相同字段

| 后端 | 捕获与发布入口 | 需要特别注意的实现边界 |
|---|---|---|
| [HF](../../../torchspec/inference/engine/hf_runner.py) | HFTargetModel 前向后，由 runner 写入 | 按样本执行；最接近普通 PyTorch 阅读体验 |
| [SGLang](../../../torchspec/inference/engine/sgl_engine.py) | 带补丁的 spec_training 后端直接写入 | 普通 generate 是 prefill；接口读后端返回键和实际长度 |
| [vLLM](../../../torchspec/inference/engine/vllm_engine.py) | extract_hidden_states 加自定义 connector | 捕获索引 +1，最后槽是 pre-norm；按层发布清单 |
| [TRT-LLM](../../../torchspec/inference/engine/trtllm_engine.py) | SaveHiddenStates 加 Mooncake 重定向补丁 | 不做 +1；最后状态用 -1 槽，表示 post-norm；仅单节点 TP |
| [TokenSpeed](../../../torchspec/inference/engine/tokenspeed_engine.py) | 后端 prefill 隐状态捕获并直接发布 | 当前只支持单节点文本 prefill，要求最后隐状态；每结果必须一个键 |
| [离线回放](../../../torchspec/inference/engine/offline_replay_engine.py) | 按 data_id 读磁盘张量，再写 Mooncake | 无 teacher 前向；CPU actor，原记录的 mask 和元数据保留 |

**层坐标和 token 坐标不要混淆。** 用户选择零基第 2 层执行后的表示，vLLM 的捕获槽对应 3；这个 +1 是层输出与下一层输入的坐标转换，与训练时 next-token 的序列移位无关。vLLM 另外追加 `num_hidden_layers` 槽，保存最终 norm 之前的表示；TRT-LLM 的 `-1` 是 post-norm。训练侧若要自己重建 logits，必须知道这个差别，否则可能漏做或重复 norm。

前缀 KV 缓存能让后端跳过重复 prompt 的前向，但它不自动保存训练需要的所有辅助层。假设第二条 prompt 长 5，其中前 3 个 token 命中缓存，只执行后 2 个位置，就无法发布声明为 `[5, K×H]` 的完整张量。因此这些捕获路径会限制前缀复用；TRT-LLM 还关闭 chunked prefill 和 overlap scheduler，以保住单次前向捕获缓冲区的顺序和生命周期。具体开关应以各后端代码为准。

## 6. Decode 数据为何少一个位置

[SglDecodeEngineMixin.generate_with_decode](../../../torchspec/inference/engine/sgl_engine_decode.py)会先检查最短生成长度能否放进上下文。如果一条 prompt 已用完容量，会保留输出中的 `None`，而不是直接从返回列表删除它。

例如输入 `[甲, 乙, 丙]` 中乙被过滤，实际只把甲、丙送入引擎，最终返回 `[甲结果, None, 丙结果]`。这让编排层严格对应输入与输出，不会把丙的特征贴到乙的标签上。

假设 prompt 长 4，completion 长 3：代码设置存储长度为 `4+3-1=6`，mask 段长为 `[4, 2]`，跳过 prompt 的位置，只对 completion 的有效位置计算 loss。末 token 没有后续训练目标；completion 长度不超过 1 的请求会被丢弃，避免零 loss 样本。首个输出 token 命中停止符也会被丢弃。这里是样本构造规则，训练内的多步展开还会继续处理各步偏移。

权重同步入口 `update_weights_from_disk` 只更新 draft，且清缓存；它借助后端内部请求接口设置 `update_draft_model=True`。teacher 的模型初始化与 draft 的训练后热更新是两种生命周期。

## 7. vLLM connector：页号、完整请求与发布清单

[MooncakeHiddenStatesConnector](../../../torchspec/inference/engine/mooncake_hidden_states_connector.py)值得分成调度端与 worker 两条线读。vLLM 为两端创建不同实例，Python 字典不共享。

调度端的流程：

1. `build_connector_meta` 为新请求缓存 prompt token、预计算 shape/dtype 和 `pp_layer_manifest`。
2. `request_finished` 保存请求的物理块号，返回 `True` 暂缓回收隐状态页，同时给上层返回已有元数据。
3. 下一次 `build_connector_meta` 把 `_pending_saves` 放进跨进程 metadata。

worker 的流程：

1. `register_kv_caches` 找到隐状态专用缓存组和张量。
2. `get_finished` 消费待保存元数据，按每个请求的页号发布完整序列。
3. 发布完成或完成独立 staging 后返回请求 ID，调度器才可回收原页。

`save_kv_layer` 在当前实现中是空操作：逐层回调并不保证能看到完整请求。阅读旧英文说明时也应以实际 `get_finished` 实现为准。

缓存视图按 `[物理块, 捕获层, 页内 token, H]` 使用。页长 4、逻辑请求长 6，块号 `[2, 5]` 展开为 slot `[8, 9, 10, 11, 20, 21]`。逻辑相邻 token 可以位于不相邻的物理页，因此不能用一个普通连续 slice 恢复请求。

有两种发布方式：

- **Gather/staging**：高级索引物化 `[S, 捕获层数, H]`，再为各层生成 `_layerN_hs` 和 `_layerN_ids`，按主机缓冲容量合并多个对象提交。
- **直接分页发布**：若缓存是 CUDA BF16、RDMA 注册成功且客户端支持多源写入，`_cache_layer_fragments` 直接列出每页的指针和长度。页长 4、S=6、H=8、BF16 时，两片大小为 `[64, 32]` 字节，Mooncake 按列表顺序拼成 `[6, 8]`。

PP 各阶段只发布自己的层，TP rank 0 避免重复写相同对象。清单把不同层标成 `hidden_states` 或 `last_hidden_states`，训练侧再组装完整样本。即便本地写入异常，`get_finished` 也先参加 PP barrier 再抛出，避免其余阶段卡在同步上。barrier 不意味着所有后台 put 都已经结束；消费者仍需等待完整键集合。

## 8. CUDA 拷贝、网卡读取与三个生命周期

重点对照 [EagleMooncakeStore._put_raw_tensors](../../../torchspec/transfer/mooncake/eagle_store.py) 和 [AsyncPutManager](../../../torchspec/transfer/mooncake/buffers.py)。异步 CPU staging 的真实顺序是：

```text
计算流写完 hidden states
  → 记录 compute_event
  → 复制流等待 compute_event
  → GPU 到 pinned CPU 缓冲的复制
  → 记录 copy_done
  → 后台线程等待 copy_done
  → Mooncake/网卡读取主机缓冲
  → Future 完成，主机缓冲可复用
```

这里有三个独立资源：

| 资源 | 谁防止过早复用 | 何时可以回收或覆盖 |
|---|---|---|
| 源 GPU 张量 | `record_stream(copy_stream)` 与 CUDA 分配器 | 复制流不再读取它之后 |
| 主机 staging 缓冲 | 按基址登记的 Future | `wait_for_buffer` 确认旧 put 完成之后 |
| 远端 Mooncake 对象 | 消费者读取和删除协议 | 训练端已取得需要的数据之后 |

`non_blocking=True` 只允许复制入队；它不会自动通知网卡。`record_stream` 防止 PyTorch 分配器过早复用源显存，也不等同数据复制完成。后台线程需要选择正确 CUDA 设备再等待事件，因为线程不自动继承调用者的当前设备。

池中只有 A/B 两块主机缓冲时，第一、二批可依次使用 A/B；第三批拿到 A 后，必须等第一批 put 的 Future。这个等待就是局部背压：生产者不能无限快地产生尚未传走的数据。线程池允许事件等待重叠，但 `batch_put_from` 本身在锁内串行执行，因为客户端并不支持任意并发 put。

GPU Direct 省掉隐藏张量的 DtoH，但网卡仍在 CUDA stream 体系之外。普通 GPU send buffer 路径先同步设备内拷贝，再同步 put。vLLM 直接分页路径先同步缓存写事件，然后同步发送；返回就是原缓存页可回收的边界。token ID 较小，依然可走 staging。

## 9. 消费端不能把“拿到键”当作“拿到数据”

[MooncakeHiddenStateStore.wait_for_keys](../../../torchspec/transfer/mooncake/store.py)先轮询元数据中全部必需键的可见性，再开始字节传输。它使用单调时钟计算截止时间；非正超时表示一直等待。缺一层或缺 `_ids` 都不能以半条样本继续训练。

[EagleMooncakeStore.get](../../../torchspec/transfer/mooncake/eagle_store.py)优先使用 GPU Direct，失败则走 host buffer。需要关注两种读取结果的存储所有权：

- GPU Direct 当前代码接收到**同一块可复用 GPU 缓冲**，返回的是 `view(dtype).reshape(shape)` 视图。下次 `get` 可覆盖旧内容；跨请求保留旧结果的消费者必须先复制或转存。这一点以实现为准，不能照英文 docstring 推断为逐字段独立分配。
- host 路径先核对字节数，再 `torch.frombuffer` 借用原生主机缓冲，随后 `.to(device)`。目标为 CUDA 时会产生设备存储；同设备的 `.to(cpu)` 不保证复制。`input_ids_cpu` 专门 `clone()` 保存，避免后续为了 token 从 GPU 同步回来。

实际训练侧怎样转存、预取和组 batch，请接着读 [MooncakeDataset](../../../torchspec/training/data_fetcher.py)。传输层本身不能替所有消费者保证视图持有期限。

`flush()` 会 drain 所有在途 put，并检查之前或 drain 中缓存的异常。HF 和离线回放在返回键前显式 flush；vLLM connector 为了不阻塞整个调度步，只检查已经完成的写入错误，因此读取端等待不可省略。

## 10. 容量、重试和删除

[helpers.py](../../../torchspec/transfer/mooncake/helpers.py)估算常见传输大小：

`B × S × (2 × K × H + 8 + 可选的 2 × H)` 字节。

前两项分别是 BF16 辅助隐状态和 int64 token，最后一项是 BF16 最后隐状态。再乘安全余量并向上对齐 256 字节。它不包含词表 logits；如果传 `target`，应单独核对实际容量。B=1、S=4、H=8、K=3 且带最后隐状态时，原始大小 288 字节；乘 1.1 后对齐为 512 字节。

缓冲容量和远端 segment 容量不是同一限制。单个待打包张量大于 staging buffer 时，connector 直接失败；把多个小对象分批不能解决一个超大对象。同步/异步 put 都会检查返回码，部分失败时尽力清理整批键，避免留下看起来可读但内容不完整的记录。

消费后的 `remove_eagle3_tensors` 用 `batch_remove(force=True)` 立即删除，最多尝试三次，仅重试失败键；对象已不存在也算成功。最终失败只记录日志，因此它是尽力回收，不是事务式保证。

[deferred_delete.py](../../../torchspec/transfer/mooncake/deferred_delete.py)另提供遵守 TTL 的延迟删除器：从入队时间再等待 TTL 加缓冲时间，使用有界队列和重试计数。它不是当前 Eagle 强制删除路径的必经步骤；`stop()` 也不是排空所有待删除对象。阅读它时把它视作独立工具，而不要把所有存储清理都画成经过该后台线程。

## 11. 服务生命周期与完整模块索引

[utils.py](../../../torchspec/transfer/mooncake/utils.py)中的 MooncakeMaster 是 Ray actor 对本地子进程的管理器。启动前解析二进制路径，子进程拥有独立进程组，关闭时先 SIGTERM 后必要时 SIGKILL；日志线程持续读取 stdout/stderr，防止管道塞满。其 Linux `prctl` 行为不是可移植到任意平台的通用 Python 功能。`check_mooncake_master_available` 只做端口 TCP 连通探测，不执行对象 PUT/GET。

下面的入口补齐没有复杂算法但仍承担边界职责的文件：

| 源码 | 阅读作用 |
|---|---|
| [inference/__init__.py](../../../torchspec/inference/__init__.py) | 推理包边界，本身不加载模型 |
| [engine/__init__.py](../../../torchspec/inference/engine/__init__.py) | 后端公开名称及可选依赖导入隔离 |
| [base.py](../../../torchspec/inference/engine/base.py) | 生命周期和训练数据生产接口 |
| [factory.py](../../../torchspec/inference/factory.py) | placement、head/worker、初始化汇合 |
| [hf_engine.py](../../../torchspec/inference/engine/hf_engine.py)、[hf_runner.py](../../../torchspec/inference/engine/hf_runner.py) | 从远程输入回到普通 PyTorch 前向 |
| [sgl_engine.py](../../../torchspec/inference/engine/sgl_engine.py)、[sgl_engine_decode.py](../../../torchspec/inference/engine/sgl_engine_decode.py) | prefill 与真实 completion 两种数据来源 |
| [vllm_engine.py](../../../torchspec/inference/engine/vllm_engine.py)、[connector](../../../torchspec/inference/engine/mooncake_hidden_states_connector.py) | 层捕获、PP、分页缓存和发布完成边界 |
| [trtllm_engine.py](../../../torchspec/inference/engine/trtllm_engine.py) | 捕获补丁配置、post-norm 与请求键 |
| [tokenspeed_engine.py](../../../torchspec/inference/engine/tokenspeed_engine.py) | TokenSpeed 单节点 prefill 适配 |
| [offline_replay_engine.py](../../../torchspec/inference/engine/offline_replay_engine.py) | 磁盘 teacher 输出复用在线消费协议 |
| [transfer/__init__.py](../../../torchspec/transfer/__init__.py)、[mooncake/__init__.py](../../../torchspec/transfer/mooncake/__init__.py) | 传输包边界、延迟导出及循环依赖隔离 |
| [store.py](../../../torchspec/transfer/mooncake/store.py) | 连接、注册、键等待、资源回收 |
| [eagle_store.py](../../../torchspec/transfer/mooncake/eagle_store.py) | 样本字段、精度、发布、读取与删除 |
| [buffers.py](../../../torchspec/transfer/mooncake/buffers.py) | 内存所有权、线程池和缓冲复用背压 |
| [helpers.py](../../../torchspec/transfer/mooncake/helpers.py) | 字节预算与容量日志 |
| [deferred_delete.py](../../../torchspec/transfer/mooncake/deferred_delete.py) | 独立的 TTL 延迟回收工具 |
| [utils.py](../../../torchspec/transfer/mooncake/utils.py) | master 进程、节点定位、地址与端口探测 |

建议第一次阅读只走 HF → Eagle store → buffer；第二次用 vLLM connector 检查自己是否能回答：谁还持有原页、谁负责源 tensor 保活、什么时候主机缓冲能复用、什么时候全部远端字段可读、以及消费者何时才能删除对象。

## 12. 带着故障现象反向追代码

这组问题可以用来检验是否真正读懂数据链路，而不只是认识类名：

| 现象 | 先检查的证据 | 接着阅读 |
|---|---|---|
| GPU 很空闲但数据生产慢 | 是否在复用缓冲前等待 Future；主机池是否全在途 | `HostBufferPool.get_buffer` → `AsyncPutManager.wait_for_buffer` |
| 有 Mooncake 键但训练一直等 | 缺哪些具体后缀/层；生产端异步异常是否已经被暴露 | `wait_for_keys` → connector `get_finished` → `check_async_errors` |
| 请求成功但字节大小不符 | 实际 prompt_tokens、辅助层数、BF16/FP32，以及前缀复用开关 | 各引擎 `generate` → `_get_tensor_shapes` → host 读取长度检查 |
| 第一条样本读对，读第二条后第一条变了 | 第一条是否仍借用 GPU receive buffer 的视图 | `_get_tensors_gpu_direct` → 训练侧预取/复制逻辑 |
| PP 训练缺失某一层 | 层 ID 是否严格递增、最终槽是否正确、该架构是否支持每阶段捕获 | vLLM `_check_per_stage_capture_support` → connector `_local_layer_positions` |
| 多节点启动卡住 | 是否启动了所有节点；是否每个副本只有一个 head；地址是否一致 | 工厂 `_prepare_vllm_engines` → `_init_headless_executor` |
| loss mask 看起来错位 | 是否把 decode 过滤后的列表压缩了；是否区分 prompt 与 completion 长度 | `generate_with_decode` 的 `active_indices` 和原位回填 |
| 结束训练后显存仍被占用 | follower executor 或后端子进程是否关闭，而非只清 Python 前端变量 | vLLM `shutdown` → `_shutdown_headless_executor` |

做一个完整纸面推演：输入两条不同长度的 token 序列，经过 HF actor 的 ObjectRef 解引用、逐条 teacher 前向、BF16 字节发布、批末 flush、Ray 元数据返回、训练端键等待、GPU/host 读取、训练侧持有与删除。再将生产者换成 vLLM：增加调度/worker 两个实例、物理页号、PP 层清单和异步发布。能指出两条路线中每一段数据归谁拥有、谁等待谁，就理解了本仓库推理与传输的主要设计约束。
