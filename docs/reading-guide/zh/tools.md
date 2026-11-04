# 工具脚本阅读：权重转换、测量与数据准备

这些程序回答生产代码之外的具体问题：训练检查点怎样交给推理框架、某种 attention 实现的成本是多少、GPU 能容纳多长序列、网络是否通、如何获得 teacher 分布上的对话。本章是源码阅读路线，没有执行任何脚本。文件名中的 `test`、`check` 或 `benchmark` 不代表运行只读或成本很小。

## 全部 Python 工具及执行边界

| 文件 | 实际工作 | 资源与输出 |
|---|---|---|
| [convert_to_hf.py](../../../tools/convert_to_hf.py) | DCP → 服务权重/config，频次分析及可选词表裁剪 | 完整 CPU 权重、PyTorch DCP、safetensors；可加载 target/数据集；写权重/config/频次数组 |
| [convert_to_torchspec.py](../../../tools/convert_to_torchspec.py) | 外部权重 → warm-start DCP | 可 Hub 下载；单进程 gloo；写 iter_0000001 与 tracker |
| [extract_dflash_checkpoint.py](../../../scripts/tools/extract_dflash_checkpoint.py) | DFlash 检查点 → 简单 `.pt` | CPU DCP/torch.load，覆盖输出文件；不输出服务配置 |
| [benchmark_eagle3.py](../../../tools/benchmark_eagle3.py) | 随机 EAGLE3 的完整训练步容量/速度 | CUDA、所选 attention 后端，真实反向与优化器更新；只打印结果 |
| [max_seq_search.py](../../../tools/max_seq_search.py) | 使用 target embedding/LM head 的容量搜索 | 权重加载、CUDA、BF16Optimizer，可能触发模型下载 |
| [bench_anchored_attention.py](../../../tools/bench_anchored_attention.py) | anchored 与 dense attention 前向对比 | CUDA、FlexAttention/编译；默认开子进程，打印延迟/峰值 |
| [bench_eagle3_mask_modes.py](../../../tools/bench_eagle3_mask_modes.py) | 五种 mask 表达的首次执行与稳定延迟 | CUDA、FlashAttention/Cute 等依赖，编译及反向；修改进程内缓存和 mask mode |
| [compare_eagle3_convergence.py](../../../tools/ci/compare_eagle3_convergence.py) | TP/PP loss 轨迹差异 | 纯 CPU JSON 读取，覆盖报告文件；不训练 |
| [check_network_topology.py](../../../tools/check_network_topology.py) | 接口、RDMA、本机带宽及集群 TCP 连通 | Linux sysfs/ip，可选 perftest；开监听端口/子进程，有 RAY_ADDRESS 时连接集群 |
| [test_sglang_engine_patch.py](../../../tools/test_sglang_engine_patch.py) | SGLang → Mooncake → GPU 读回集成验证 | 补丁版 SGLang、CUDA、小模型、Mooncake master；启动服务并写删对象 |
| [generate_data.py](../../../tools/generate_data.py) | 用在线 teacher 重写 assistant 回答 | OpenAI Python 客户端及已启动兼容服务；真实请求、追加成功输出、覆盖错误输出 |
| [prepare_perfectblend.py](../../../scripts/tools/prepare_perfectblend.py) | 下载并规范 PerfectBlend 对话 | datasets/Hub，可能大量下载；覆盖 JSONL |

所谓“转换”也不一定无外部副作用：指定 teacher 路径可能访问模型 Hub，缺配置时 `convert_to_hf` 会把自动生成配置写到输入目录。需要纯本地实验时，应先准备所有本地权重/配置/数据，再读实际路径解析分支；本章不替这些程序增加下载或启动保证。

## 1. 理解三种检查点产物

**训练 DCP** 保存分布式训练需要的参数结构，张量可能分散在多个文件。转换器的 `_EmptyStateDictLoadPlanner` 先根据元数据中的完整 shape/dtype 分配 CPU 张量，再在 `no_dist=True` 下读分片。因此“不需要原来几张 GPU”不等于“不需要足够内存”。

**服务权重**通常是 `model.safetensors` 加 `config.json`，另有必要的词表映射。它需要推理端认识的键名、模型类型、捕获层索引及输出精度。只把 DCP 文件改后缀不会成为服务权重。

**warm-start DCP** 是外部权重装入 TorchSpec draft 后重新保存成训练目录。本工具写 `iter_0000001/model` 和 tracker，未恢复原优化器动量、随机数状态或原始训练进度；它适合初始化，不是精确续训。

阅读顺序建议：先看 [convert_to_torchspec._remap_to_model](../../../tools/convert_to_torchspec.py)，再看 [convert_to_hf._extract_model_weights](../../../tools/convert_to_hf.py)，最后看各自保存函数。两个方向不只是互换文件后缀。

### 键名变了，张量维度不自动变

共享 [keymap.py](../../../torchspec/models/draft/keymap.py) 包含这些前缀关系：

| 训练名 | 服务名 |
|---|---|
| `midlayer.` | `layers.0.` |
| `context_proj.` | `fc.` |
| `context_norm.` | `hidden_norm.` |
| `final_norm.` | `norm.` |

例如拼接三层 teacher 特征后送入 `nn.Linear(3H, H)`，其权重形状为 `[H, 3H]`。将 `context_proj.weight` 改成 `fc.weight` 不会转置它。反向转换时，若目标模型已经有 `layers.0`，应保留真实存在的键，不能机械改成 `midlayer`；当前 `_remap_to_model` 就先检查目标模型键集合。

服务导出还有模型专用路径：模型提供 `state_dict_for_serving` 时优先调用，否则通用重命名后执行 latent expert 拆分。配置也需同步转换：DFlash2 使用其服务配置助手，Kimi 的 vLLM 导出有捕获层 +1 调整。权重能被读成张量不代表模型配置与运行后端已匹配。

### 缺键和 dtype 的真实语义

warm-start 先构造目标模型，再 `load_state_dict(..., strict=False)`。缺失参数保持新模型初值；同名参数 shape 不兼容仍可能报错。工具拒绝零匹配或匹配不到一半键数，但这个阈值按键数算，不能保证一半参数量，更不能保证所有关键层都来自外部 checkpoint。

HF 转换先按检查点最常见 dtype 建模/载入，再执行可选输出精度转换。指定 target 后还会覆盖 embedding；这不再是保持所有参数值不变的纯容器转换。DFlash2 对缺键检查比通用分支更严格，只在允许补 teacher embedding 时放行对应缺项。

[extract_dflash_checkpoint.py](../../../scripts/tools/extract_dflash_checkpoint.py) 是更轻的提取工具：普通 `.pt` 输入先筛名字并保留原键，DCP 输入会去 `draft_model.` 前缀、排除 t2d/d2t；完全找不到 draft 键时退回保存全部项。因而“输出文件存在”并不证明拿到了可直接加载的纯 draft state_dict。

## 2. 词表裁剪：行数只是表面，行的含义才是契约

完整输出头可写成 `W ∈ R[V,H]`，每行对应一个 target token。裁成 D 个输出列意味着保留 D 行，并给出 draft 列号与 target token 的关系；embedding 是否完整、输出头是否裁剪是不同问题，不能把两个词表轴混为一谈。

`_count_token_frequencies` 只统计监督位置。token `[2,5,2]`、mask `[0,1,1]` 的计数是 `{5:1, 2:1}`。它优先取 `loss_mask`，否则展开 packed mask，再否则全一。统计前使用训练同一数据加载器，模板、最大长度和缓存会改变入选 token 分布。coverage 是这些监督位置中被保留 token 覆盖的比例，不是验证阶段接受率。

当前 [process_token_dict_to_mappings](../../../torchspec/data/preprocessing.py) 的编码特别值得注意：

- `t2d` 是长度 V 的**布尔成员表**，不是每个 target ID 到 draft ID 的整数表。
- `d2t[j] = used_target_ids[j] - j` 是**偏移编码**，恢复 target ID 时还要加上 draft 列号 j。

例如保留 target ID `[2,5,9]`，d2t 为 `[2,4,7]`。要从完整 `W` 选择对应三行，应理解真实 ID 是 `[2,5,9]`，不能只看变量名就把偏移当行号。

阅读本次版本还应识别两个既有不一致，注释没有修复它们：`_save_with_vocab_pruning` 给仅接收一个参数的 `_prepare_export_tensors` 传了两个参数；该裁剪函数还直接使用 `W[d2t]`，与上述偏移编码不一致。因此本教程不把这个分支作为已验证可用的转换命令。已裁剪的头也不能随意换数据重排，因为每行权重已经绑定原训练映射；因子化头和 DFlash2 另有明确拒绝裁剪的检查。

## 3. 看 benchmark，先问“计时圈住了什么”

三个基准的数字不能直接放进一张“谁更快”的表：

| 工具 | 计时工作 | 计时口径 |
|---|---|---|
| `benchmark_eagle3` | 合成 batch、目标构造、draft 前向、反向、优化器更新 | 前后 GPU 同步的 CPU 墙钟平均值 |
| `bench_anchored_attention` | 多深度 attention 前向与 mask 构造，可重复多层 | 整段循环前后同步的墙钟平均值 |
| `bench_eagle3_mask_modes` | mask 方案对应 attention 前向和 Q 反向 | CUDA event 的设备时间平均值 |

GPU 调用通常只是将 kernel 入队。若没有等待设备完成，CPU 计时可能只反映发射开销。`benchmark_eagle3` 用 `synchronize → t0 → step → synchronize → t1`；mask 工具用一对 CUDA event 圈住调用，最后统一同步后读取 elapsed_time。`run_fwd_bwd` 的英文描述提到 synchronize，但实际函数体没有同步，由上层测量代码完成。

预热用于触发 JIT、分配器和优化器状态等首次工作。`bench_eagle3_mask_modes` 的 compile 列包含每个唯一长度桶的随机张量创建及首次前后向，并不是编译器本身的纯耗时。正式延迟测量用 snap 后的长度：例如请求 384 被补到 512，实际张量按 512 计算，性能反映分桶后的真实工作量。

峰值显存是 `max_memory_allocated`，包含统计开始时已存活的模型与张量，不是单个 kernel 新增显存，也不是设备总使用量。anchored 工具除以 `1e9` 输出十进制 GB；EAGLE3 训练基准除以 `1024**3`，更准确说是 GiB。对比时要统一单位。

### Anchored attention 的公平性边界

[bench_anchored_attention](../../../tools/bench_anchored_attention.py) 的 dense 路径有 S 个 query、KV 每深度增长 S；anchored 路径只有 A 个 query、KV 每深度增长 A。mask 让锚点只看其之前的上下文，以及追加区域中属于自己链的列。多层 draft 会加入稠密上下文计算，因为下一层需要上一层所有上下文位置的输出。

默认每种策略单独子进程，防止前一种策略预热了编译器内部状态，给后测策略不对称优势。结果只代表这里的 attention 前向，不包含投影、loss、反向或优化器。子进程缺 RESULT 行会显示 n/a，不能解释成零成本。

### Mask 基准的梯度范围

[bench_eagle3_mask_modes](../../../tools/bench_eagle3_mask_modes.py) 中 Q 设置 `requires_grad=True`，K/V 没有。因此它不是 Q/K/V 全部反向的完整模型训练。clear cache 清除一些 Python/FlashAttention 缓存，也不保证清掉所有系统级编译缓存。

汇总表的方向也不同：compile speedup 为“基准时间/当前时间”，大于一更好；延迟比为“当前时间/基准时间”，小于一更好。不要把同一行两个 `x` 当成同一种倍率。

## 4. 最大序列长度不是模型的永久属性

[benchmark_eagle3](../../../tools/benchmark_eagle3.py) 用随机模型维度；[max_seq_search](../../../tools/max_seq_search.py) 加载真实 target embedding 和 LM head，但 teacher 中间特征仍是随机生成的 `[B,S,3H]` 与 `[B,S,H]`。两者都不包含 Ray/Mooncake 数据预取和完整 teacher 推理的并发内存。

每个候选长度都实际做前向、反向、优化器更新，才会触发 Adam 动量/方差和 FP32 主参数的成本。仅成功一次前向不能证明可训练。清空优化器 `state` 是为了减小前一候选残留影响；不会重新初始化模型或释放所有优化器常驻参数。

二分搜索假设长度越长越难容纳，在 step 网格上查边界；线性搜索从低到高，首次 OOM 就停止，更少经历“大分配失败后继续测试小长度”的碎片效应。JIT 桶、设备其他进程、分配器状态和 loss mask 都会让结果变化。

当前实现的边界也需读明白：

- 搜索边界会对齐；`max_seq_search` 下界向下取整可能产生零，上界向上取整可能超过用户输入。
- `best` 初始就设为下界，所有候选失败时仍可返回正数且峰值为零；应结合每次成功日志，不只信最终 Maximum 行。
- `benchmark_eagle3._try_time` 将所有 RuntimeError 显示成 OOM，其他后端/形状错误也可能被这种格式掩盖。
- `max_seq_search` 的 `--vocab-mapping` 当前只解析和打印，未实际加载设置映射；不能根据参数存在推断裁剪路径准备完整。
- `run_eagle3_backward_and_update` 的 accumulation_steps 只缩放 loss，仍每次调用 step，不是完整的梯度累积调度。

因此容量记录至少应附 GPU、B/H/V、TTT 深度、attention 后端、checkpointing、loss 分布、step 粒度及成功/失败日志。它是此工作负载的实测约束，不是所有训练配置都能使用的安全上限。

## 5. 网络诊断与补丁验证各证明什么

[check_network_topology](../../../tools/check_network_topology.py) 区分三类证据：

1. sysfs/ibv_devinfo 的设备状态和网卡地址，说明设备及链路可见。
2. 每 HCA 一个 ACTIVE 端口上的 `ib_write_bw` 本地 server/client，测本地数据路径 Gb/s；不是跨节点 RDMA 带宽。
3. Ray 集群节点之间两两 TCP connect，说明有向 socket 可达；报告 RTT 实际为建连时间，没有应用层 echo。

集群模式先在所有节点建立监听器，再并行 accept 与 probe。actor 的并发方法是必要条件：若 accept 挡住该 actor 的 probe，就会把调度串行问题误当网络故障。推荐的 NCCL_SOCKET_IFNAME 只是优先选第一个 RDMA 关联候选，不会改环境，也不证明所有节点接口同名或互通。

[test_sglang_engine_patch](../../../tools/test_sglang_engine_patch.py) 是更高层的一条真实链路：临时 master → 补丁 SGLang 引擎 → 零新 token prefill → 读返回键 → 按 `[S,K×H]`/`[S,H]` 拉回 GPU → 检查形状、非零、有限数和 token 类型 → 删除对象。它不与独立 teacher reference 逐元素对比，所以通过只能说明基本采集/传输契约可走通，不能证明捕获层、norm 或数值完全正确。

脚本需要 CUDA、补丁 SGLang、小模型和 Mooncake 二进制。成功路径会关闭 store/engine，main 的 finally 回收 master；中间断言失败不一定执行成功路径里的所有清理。端口查找使用 UDP bind，也不等于持续预留之后服务使用的 TCP 端口。

## 6. 数据再生成与数据格式准备

[generate_data](../../../tools/generate_data.py) 连接已有 OpenAI 兼容服务，不自行启动服务。对一段多轮对话，它保留 system/user，丢弃旧 assistant，并在每个 user 后生成新回答。下一轮因此看到的是新 teacher 回答历史。不同样本可并行，同一样本各轮依赖前一轮结果。

线程池的上限是每服务 concurrency；达到某服务上限后等待 Future 完成再提交，是请求端背压。失败记录按计数重试，连接类错误会移除该服务；所有服务不可用时持续探测，未设置总超时。健康检查也是一 token 真实生成请求，并把部分非连接业务错误当作“可达”。

成功结果及时 flush 并按完成顺序写入，data_id 是输入文件行号，用于续跑跳过；重排输入后继续用旧输出会导致 ID 语义不一致。错误文件每次覆盖，成功文件追加。输出路径应使用 `.jsonl`，因为错误路径只是字符串替换，缺少该后缀会与成功路径相同。

`context_length` 的实现是“最后一轮 prompt_tokens + 所有轮 completion_tokens 之和”，不等同最终对话重新 tokenize 的严格长度，尤其多轮历史已经含先前回答。统计时应按代码口径解释。

[prepare_perfectblend](../../../scripts/tools/prepare_perfectblend.py) 下载 train split，统一 ShareGPT 的 from/value 和 role/content 两种消息格式。先抽样，再过滤最少消息数、assistant 存在以及 assistant 总字符数至少 50；所以抽样 50000 个候选不保证输出 50000 条。50 字符只是启发式，不是保证够 16 token 的 tokenizer 校验。输出采用覆盖写入。

## 7. 用 CI 轨迹检查理解“数值接近”

[compare_eagle3_convergence](../../../tools/ci/compare_eagle3_convergence.py) 检查两条轨迹非空、长度一致、step 一致、loss 有限，再计算：

`relative_diff = |loss_tp - loss_pp| / max(|loss_tp|, |loss_pp|, 1e-8)`。

它采用对称分母并检查**最差一步**，不是只比较最终 loss 或平均误差。比如 2 与 2.01 的相对差是 0.01/2.01。通过说明这些 CI 步在给定阈值内接近，不说明数据/目标本身一定正确，也不是训练质量或在线接受率的评估。

读完工具后，可以沿 [推理与传输](inference-transfer.md) 回到真实生产链路，核对基准没有包含的 teacher 执行、远端发布、预取、张量生命周期与控制面调度成本。
