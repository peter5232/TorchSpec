# 全部生产模块导航

[返回总路线](README.md)

这里按当前源码实际文件列举，不沿用可能落后于实现的旧目录图。函数数包含类方法与嵌套函数，不包含 dataclass 自动生成的方法。覆盖数量用于发现遗漏，不代表解释质量；复杂实现需配合章节中的算例与源码正文注释阅读。

共 **106 个 Python 模块、1048 个函数/方法定义**。

| 模块 | 函数数 | 模块导读 | 配套章节 |
|---|---:|---|---|
| [torchspec/__init__.py](../../../torchspec/__init__.py) | 0 | 中文导读：公开导入入口，区分训练包装 Model、草稿 DraftModel 和结构 Config。 | [阅读](pipeline.md) |
| [torchspec/config/__init__.py](../../../torchspec/config/__init__.py) | 0 | 中文导读：导出配置 schema、加载器与扁平化转换；实际派生逻辑在 train_config.py。 | [阅读](pipeline.md) |
| [torchspec/config/inference_config.py](../../../torchspec/config/inference_config.py) | 2 | 中文导读：推理后端的配置边界。TP 是把一个模型的张量计算分到多卡，PP 是把层分成多个阶段； | [阅读](pipeline.md) |
| [torchspec/config/mooncake_config.py](../../../torchspec/config/mooncake_config.py) | 9 | 中文导读：分布式张量存储的容量与通信配置。 | [阅读](pipeline.md) |
| [torchspec/config/train_config.py](../../../torchspec/config/train_config.py) | 12 | 中文导读：配置从结构化 YAML 到历史扁平 args 的桥梁。 | [阅读](pipeline.md) |
| [torchspec/config/utils.py](../../../torchspec/config/utils.py) | 4 | 中文导读：从目标模型配置推导兼容的草稿几何，尤其保持词表和 RoPE 参数一致。 | [阅读](pipeline.md) |
| [torchspec/controller/__init__.py](../../../torchspec/controller/__init__.py) | 0 | 中文导读：导出流水线控制器、推理调度器和装配/循环入口；本文件不执行训练作业。 | [阅读](pipeline.md) |
| [torchspec/controller/eval.py](../../../torchspec/controller/eval.py) | 5 | 中文导读：评估使用固定 teacher 特征缓存，避免每次评估都重复运行昂贵目标模型。 | [阅读](pipeline.md) |
| [torchspec/controller/inference_manager.py](../../../torchspec/controller/inference_manager.py) | 26 | 中文导读：异步生产者调度器。它把控制器中的输入分批发给引擎，再把完成的 key 送回控制器。 | [阅读](pipeline.md) |
| [torchspec/controller/loop.py](../../../torchspec/controller/loop.py) | 7 | 中文导读：主循环把异步样本生产与同步 optimizer step 连接起来。 | [阅读](pipeline.md) |
| [torchspec/controller/setup.py](../../../torchspec/controller/setup.py) | 3 | 中文导读：把已经创建的推理引擎、训练组和控制器连起来。 | [阅读](pipeline.md) |
| [torchspec/controller/training_controller.py](../../../torchspec/controller/training_controller.py) | 38 | 中文导读：训练流水线的样本账本。负责“哪条数据该给谁”，不负责 teacher forward。 | [阅读](pipeline.md) |
| [torchspec/data/__init__.py](../../../torchspec/data/__init__.py) | 0 | 中文导读：汇总数据加载、模板、renderer、padding 和 mask 编解码公共接口。 | [阅读](pipeline.md) |
| [torchspec/data/dataset.py](../../../torchspec/data/dataset.py) | 11 | 中文导读：在线数据入口，将对话变成 InferenceInput 所需字段；此时尚未运行目标模型。 | [阅读](pipeline.md) |
| [torchspec/data/parse.py](../../../torchspec/data/parse.py) | 37 | 中文导读：聊天文本的“协议层”，决定角色边界、思考内容、工具调用以及监督区间。 | [阅读](pipeline.md) |
| [torchspec/data/preprocessing.py](../../../torchspec/data/preprocessing.py) | 16 | 中文导读：对话规范化、监督区间构建和草稿词表选择。 | [阅读](pipeline.md) |
| [torchspec/data/renderers/__init__.py](../../../torchspec/data/renderers/__init__.py) | 0 | 中文导读：导入并注册 kimi-k3 renderer；注册发生在包导入时，使配置名称能找到具体实现。 | [阅读](pipeline.md) |
| [torchspec/data/renderers/base.py](../../../torchspec/data/renderers/base.py) | 3 | 中文导读：renderer 比字符串模板拥有更多职责：同时负责消息编码和监督范围。 | [阅读](pipeline.md) |
| [torchspec/data/renderers/kimi_k3.py](../../../torchspec/data/renderers/kimi_k3.py) | 12 | 中文导读：Kimi K3 的 XTML 编码交给 checkpoint 自带 tokenizer，避免手工拼字符串破坏分段编码。 | [阅读](pipeline.md) |
| [torchspec/data/renderers/registry.py](../../../torchspec/data/renderers/registry.py) | 6 | 中文导读：名称到 renderer 类的注册表，把模型特定格式与通用数据加载器解耦。 | [阅读](pipeline.md) |
| [torchspec/data/template.py](../../../torchspec/data/template.py) | 4 | 中文导读：注册各模型的角色头、结束符、系统提示和 parser 类型。 | [阅读](pipeline.md) |
| [torchspec/data/utils.py](../../../torchspec/data/utils.py) | 21 | 中文导读：数据形状、监督 mask 和数据源读取的公共工具。 | [阅读](pipeline.md) |
| [torchspec/inference/__init__.py](../../../torchspec/inference/__init__.py) | 0 | 推理包将 teacher 的前向计算组织成可远程调用的训练数据生产者。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/__init__.py](../../../torchspec/inference/engine/__init__.py) | 0 | 集中导出推理接口和可用后端，方便调用方选择实现。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/base.py](../../../torchspec/inference/engine/base.py) | 6 | 定义训练数据生产者的共同生命周期；调用方通过 Ray 调度，具体后端负责前向和张量发布。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/hf_engine.py](../../../torchspec/inference/engine/hf_engine.py) | 6 | Hugging Face 后端的远程外壳：处理 actor 的 GPU 定位、日志和 ObjectRef，模型计算交给 HFRunner。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/hf_runner.py](../../../torchspec/inference/engine/hf_runner.py) | 15 | 直接调用 HFTargetModel 提取 teacher 输出，再按样本写入 EagleMooncakeStore。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/mooncake_hidden_states_connector.py](../../../torchspec/inference/engine/mooncake_hidden_states_connector.py) | 31 | vLLM 的隐状态发布桥：调度端记录请求/页号及清单，worker 从分页缓存取出完整请求并写入 Mooncake。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/offline_replay_engine.py](../../../torchspec/inference/engine/offline_replay_engine.py) | 7 | 离线回放把磁盘已有 teacher 张量重新发布到 Mooncake，复用在线训练的消费契约。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/sgl_engine.py](../../../torchspec/inference/engine/sgl_engine.py) | 11 | SGLang 适配器依赖带 spec_training 支持的后端：它执行前向并直接写 Mooncake，本层只整理请求和键。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/sgl_engine_decode.py](../../../torchspec/inference/engine/sgl_engine_decode.py) | 4 | 在 SGLang 上生成真实 completion 再提取训练样本，同时提供已训练 draft 权重的热更新入口。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/tokenspeed_engine.py](../../../torchspec/inference/engine/tokenspeed_engine.py) | 6 | TokenSpeed 适配器使用 prefill 隐状态捕获，把后端已发布的键转成 TorchSpec 样本元数据。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/trtllm_engine.py](../../../torchspec/inference/engine/trtllm_engine.py) | 14 | TRT-LLM 的 PyTorch 后端用 SaveHiddenStates 捕获训练数据，仓库补丁把结果从文件转存到 Mooncake。 | [阅读](inference-transfer.md) |
| [torchspec/inference/engine/vllm_engine.py](../../../torchspec/inference/engine/vllm_engine.py) | 17 | 通过 vLLM extract_hidden_states 与自定义 connector 采集完整 prompt 的逐层隐状态。 | [阅读](inference-transfer.md) |
| [torchspec/inference/factory.py](../../../torchspec/inference/factory.py) | 16 | 根据推理后端和 Ray placement group 创建 actor，并把慢速模型初始化发到目标节点。 | [阅读](inference-transfer.md) |
| [torchspec/megatron.py](../../../torchspec/megatron.py) | 5 | Megatron-Core 仅提供模型/专家并行组和梯度辅助函数，训练循环与优化器仍由 TorchSpec 管理。 | [阅读](training.md) |
| [torchspec/models/__init__.py](../../../torchspec/models/__init__.py) | 0 | 模型训练包装器的公共入口：EAGLE3 做多深度展开，DFlash 家族做锚点块预测。 | [阅读](models.md) |
| [torchspec/models/anchored_eagle3.py](../../../torchspec/models/anchored_eagle3.py) | 10 | EAGLE3 的锚点训练：单层结构允许全序列只投影上下文 K/V，昂贵的查询、MLP 和 loss 只在 N 个锚点运行。 | [阅读](models.md) |
| [torchspec/models/dflash.py](../../../torchspec/models/dflash.py) | 14 | DFlash 训练包装：教师多层特征作为上下文，每个锚点生成 anchor+MASK 的并行预测块。 | [阅读](models.md) |
| [torchspec/models/dflash2.py](../../../torchspec/models/dflash2.py) | 5 | DFlash2 在 DFlash 训练尾部加入候选选择器监督，并为每一层保留 full/sliding attention 策略。 | [阅读](models.md) |
| [torchspec/models/draft/__init__.py](../../../torchspec/models/draft/__init__.py) | 0 | 草稿参数层与配置的统一导出；AutoDraftModelConfig 解析架构，AutoEagle3DraftModel 实例化。 | [阅读](models.md) |
| [torchspec/models/draft/auto.py](../../../torchspec/models/draft/auto.py) | 6 | 配置 architecture 名称到具体 draft 类的路由；服务配置、训练构造和权重加载共享这个入口。 | [阅读](models.md) |
| [torchspec/models/draft/base.py](../../../torchspec/models/draft/base.py) | 24 | EAGLE draft 的参数层契约、因果掩码、输出头和检查点单张量读取。 | [阅读](models.md) |
| [torchspec/models/draft/deepseek_eagle.py](../../../torchspec/models/draft/deepseek_eagle.py) | 16 | DeepSeek MLA 版 EAGLE3：注意力输入是归一化 embedding 与隐藏态拼接的 2H 特征。 | [阅读](models.md) |
| [torchspec/models/draft/dflash.py](../../../torchspec/models/draft/dflash.py) | 21 | DFlash 参数骨干：教师多层特征融合后，每层以 context+draft 为 K/V、仅 draft 为 Q。 | [阅读](models.md) |
| [torchspec/models/draft/dflash2.py](../../../torchspec/models/draft/dflash2.py) | 15 | DFlash2 在双源 KV 骨干中加入块内动态分组卷积，并用低秩前驱/后继码本重排候选。 | [阅读](models.md) |
| [torchspec/models/draft/dspark.py](../../../torchspec/models/draft/dspark.py) | 14 | DSpark 的 Markov 偏置和接受率预测头；K3 变体用 MLA 替换 DFlash 的 GQA。 | [阅读](models.md) |
| [torchspec/models/draft/eagle4_moe.py](../../../torchspec/models/draft/eagle4_moe.py) | 10 | EAGLE4 的稀疏 FFN 与服务权重布局。路由选择使用带偏置 sigmoid 分数，混合权重使用未加偏置的分数。 | [阅读](models.md) |
| [torchspec/models/draft/keymap.py](../../../torchspec/models/draft/keymap.py) | 2 | 训练内部参数名与 HF/vLLM/SGLang 导出键之间的双向转换。 | [阅读](models.md) |
| [torchspec/models/draft/llama3_eagle.py](../../../torchspec/models/draft/llama3_eagle.py) | 83 | Llama/GQA EAGLE3/4 参数骨干及多种等价训练注意力实现。 | [阅读](models.md) |
| [torchspec/models/dspark.py](../../../torchspec/models/dspark.py) | 4 | DSpark 复用 DFlash 特征/掩码骨干，但改为每槽预测下一个 token，并组合 Markov CE、L1 蒸馏与置信度 BCE。 | [阅读](models.md) |
| [torchspec/models/eagle3.py](../../../torchspec/models/eagle3.py) | 7 | EAGLE3 的训练展开与教师分布表示。输入已由 trainer 对齐，不能把 depth 当成 Transformer 层号。 | [阅读](models.md) |
| [torchspec/models/mtp.py](../../../torchspec/models/mtp.py) | 5 | 批优先训练张量与 Megatron 原生 MTP 的序列优先接口之间的适配。 | [阅读](models.md) |
| [torchspec/models/ops/__init__.py](../../../torchspec/models/ops/__init__.py) | 0 | 训练共用算子入口：FlexAttention mask、蒸馏损失与 assistant 内容监督 mask。 | [阅读](models.md) |
| [torchspec/models/ops/anchors.py](../../../torchspec/models/ops/anchors.py) | 3 | 固定预算锚点采样器，供 EAGLE3 anchored 与 DFlash 共用。 | [阅读](models.md) |
| [torchspec/models/ops/flex_attention.py](../../../torchspec/models/ops/flex_attention.py) | 18 | FlexAttention 编译入口和 EAGLE3 causal+对角 suffix 的稀疏掩码构造。 | [阅读](models.md) |
| [torchspec/models/ops/loss.py](../../../torchspec/models/ops/loss.py) | 12 | EAGLE3 的编译损失内核：先筛有效行，再做 RMSNorm/head 与分布比较，返回求和量供外层归一化。 | [阅读](models.md) |
| [torchspec/models/ops/loss_mask.py](../../../torchspec/models/ops/loss_mask.py) | 3 | 按 token 序列中的 assistant header/end 标记建立内容监督 mask，CPU 上由 Numba 扫描。 | [阅读](models.md) |
| [torchspec/models/qwen3_5_mtp.py](../../../torchspec/models/qwen3_5_mtp.py) | 7 | Qwen3.5 MoE MTP 的 Megatron 构造与 HF 权重导入。 | [阅读](models.md) |
| [torchspec/models/target/__init__.py](../../../torchspec/models/target/__init__.py) | 0 | 教师侧公共协议：生成指定层隐藏态的数据后端，以及训练端只加载的冻结 LM head。 | [阅读](models.md) |
| [torchspec/models/target/eagle3_target_model.py](../../../torchspec/models/target/eagle3_target_model.py) | 13 | 教师数据生成协议与 Hugging Face 实现。用少量 hook 捕获指定层而不返回全部层激活，节约显存。 | [阅读](models.md) |
| [torchspec/models/target/target_utils.py](../../../torchspec/models/target/target_utils.py) | 13 | 训练端只加载冻结教师 LM head，以及输入为 pre-norm 时需要的 final norm，避免复制整台教师模型。 | [阅读](models.md) |
| [torchspec/models/target_hidden.py](../../../torchspec/models/target_hidden.py) | 3 | 用临时 forward hook 验证教师最后一层与 final norm 的精确边界，避免把 pre/post-norm 状态混用。 | [阅读](models.md) |
| [torchspec/offline/__init__.py](../../../torchspec/offline/__init__.py) | 0 | 中文导读：导出新版磁盘回放协议与 norm 元数据配置器；它不等于旧 .ckpt 数据集接口。 | [阅读](pipeline.md) |
| [torchspec/offline/dataset.py](../../../torchspec/offline/dataset.py) | 7 | 中文导读：离线回放格式由 dataset.json（语义）、manifest.jsonl（索引）和 samples/*.pt（张量）组成。 | [阅读](pipeline.md) |
| [torchspec/offline/generate.py](../../../torchspec/offline/generate.py) | 4 | 中文导读：特征物化作业。复用在线推理和控制器，但消费者换成保存 actor，不执行草稿反向传播。 | [阅读](pipeline.md) |
| [torchspec/offline/saving_actor.py](../../../torchspec/offline/saving_actor.py) | 5 | 中文导读：用“从队列取数据并写盘”替代训练消费者，证明生产与消费是通过协议解耦的。 | [阅读](pipeline.md) |
| [torchspec/ray/__init__.py](../../../torchspec/ray/__init__.py) | 0 | 中文导读：Ray 基础设施包。placement_group 预留资源，ray_actor 处理进程设备， | [阅读](pipeline.md) |
| [torchspec/ray/placement_group.py](../../../torchspec/ray/placement_group.py) | 27 | 中文导读：把“需要多少训练/推理 GPU”映射为 Ray 集群上的具体资源位置。 | [阅读](pipeline.md) |
| [torchspec/ray/ray_actor.py](../../../torchspec/ray/ray_actor.py) | 7 | 中文导读：Ray actor 共用的节点、GPU 和分布式 rendezvous 地址工具。 | [阅读](pipeline.md) |
| [torchspec/ray/train_group.py](../../../torchspec/ray/train_group.py) | 16 | 中文导读：训练 actor 的主进程端代理，批量发起远程调用并管理 rank→actor 的映射。 | [阅读](pipeline.md) |
| [torchspec/train_entry.py](../../../torchspec/train_entry.py) | 12 | 中文阅读入口：这里是训练作业的装配层，不是模型 forward。 | [阅读](pipeline.md) |
| [torchspec/training/__init__.py](../../../torchspec/training/__init__.py) | 0 | 训练包命名空间：具体 Trainer、优化器、FSDP 与检查点按需从子模块导入；本文件不启动训练。 | [阅读](training.md) |
| [torchspec/training/checkpoint.py](../../../torchspec/training/checkpoint.py) | 22 | 训练状态持久化与 warm start：DCP 保存分布式训练状态，safetensors 初始权重用于模型初始化。 | [阅读](training.md) |
| [torchspec/training/data_fetcher.py](../../../torchspec/training/data_fetcher.py) | 26 | 训练输入适配：Ray 队列只传 TrainSample 元数据，真正的大张量通过 Mooncake key 读取。 | [阅读](training.md) |
| [torchspec/training/dflash2_trainer.py](../../../torchspec/training/dflash2_trainer.py) | 3 | DFlash2 的轻量构建适配：保留 DFlash 训练循环，替换配置、网络及带 selector 损失的 wrapper。 | [阅读](training.md) |
| [torchspec/training/dflash_trainer.py](../../../torchspec/training/dflash_trainer.py) | 17 | DFlash 系列共用训练器：构造 block/anchor 目标，支持 DFlash2 与 DSpark 通过 hook 更换网络。 | [阅读](training.md) |
| [torchspec/training/dspark_trainer.py](../../../torchspec/training/dspark_trainer.py) | 3 | DSpark 在 DFlash 管线上接入 Markov/confidence head 和 L1 蒸馏；不另写一套梯度累积循环。 | [阅读](training.md) |
| [torchspec/training/eagle3_trainer.py](../../../torchspec/training/eagle3_trainer.py) | 14 | Eagle 训练适配层：把 teacher 特征变成多步训练目标，连接 draft、FSDP 和公共训练循环。 | [阅读](training.md) |
| [torchspec/training/fsdp.py](../../../torchspec/training/fsdp.py) | 9 | 参数分配与分布式包装：非主 rank 用 meta 建模，再由 rank 0 广播；支持 REPLICATE 和 FULL_SHARD。 | [阅读](training.md) |
| [torchspec/training/lr_scheduler.py](../../../torchspec/training/lr_scheduler.py) | 7 | 统一学习率曲线：warmup 后接 constant/linear/cosine/inverse-square-root/WSD。 | [阅读](training.md) |
| [torchspec/training/mtp.py](../../../torchspec/training/mtp.py) | 1 | MTP 的硬标签目标：用第 t 行隐藏状态预测 x[t+2]，与普通 next-token CE 的一位偏移不同。 | [阅读](training.md) |
| [torchspec/training/optimizer.py](../../../torchspec/training/optimizer.py) | 11 | BF16 模型配 FP32 master 权重的优化器封装：AdamW 更新高精度副本，再回写计算用参数。 | [阅读](training.md) |
| [torchspec/training/trainer.py](../../../torchspec/training/trainer.py) | 30 | 训练公共骨架：队列读取、设备 mesh、梯度累积、检查点和评估缓存在此衔接。 | [阅读](training.md) |
| [torchspec/training/trainer_actor.py](../../../torchspec/training/trainer_actor.py) | 15 | Ray Actor 边界：设置 GPU/rank 与 torch.distributed，再把远程调用委托给具体 Trainer。 | [阅读](training.md) |
| [torchspec/transfer/__init__.py](../../../torchspec/transfer/__init__.py) | 0 | 传输层连接推理生产者与训练消费者：Mooncake 承载张量字节，调用结果携带键和形状。 | [阅读](inference-transfer.md) |
| [torchspec/transfer/mooncake/__init__.py](../../../torchspec/transfer/mooncake/__init__.py) | 1 | Mooncake 子包的公共入口：轻量工具直接导出，配置和存储类按属性访问时再导入。 | [阅读](inference-transfer.md) |
| [torchspec/transfer/mooncake/buffers.py](../../../torchspec/transfer/mooncake/buffers.py) | 24 | 管理可复用的 CPU pinned/GPU 字节缓冲，以及后台 Mooncake 写入的 Future。 | [阅读](inference-transfer.md) |
| [torchspec/transfer/mooncake/deferred_delete.py](../../../torchspec/transfer/mooncake/deferred_delete.py) | 10 | 提供遵守对象租约 TTL 的后台延迟删除器，使用有界新任务队列和失败重试列表。 | [阅读](inference-transfer.md) |
| [torchspec/transfer/mooncake/eagle_store.py](../../../torchspec/transfer/mooncake/eagle_store.py) | 12 | Eagle3 样本的存储协议：一个逻辑键拆成 _hs、_ids 和可选 _lhs/_tgt，shape/dtype 通过元数据传递。 | [阅读](inference-transfer.md) |
| [torchspec/transfer/mooncake/helpers.py](../../../torchspec/transfer/mooncake/helpers.py) | 2 | 纯容量辅助函数：把模型维度转成传输缓冲字节预算，供配置和错误信息使用。 | [阅读](inference-transfer.md) |
| [torchspec/transfer/mooncake/store.py](../../../torchspec/transfer/mooncake/store.py) | 17 | 通用 Mooncake 客户端拥有网络连接、已注册缓冲、复制流和后台写入线程池。 | [阅读](inference-transfer.md) |
| [torchspec/transfer/mooncake/utils.py](../../../torchspec/transfer/mooncake/utils.py) | 13 | Mooncake 服务的进程管理与连通性工具：Ray actor 启动 master，子进程日志回流，退出时终止进程组。 | [阅读](inference-transfer.md) |
| [torchspec/utils/__init__.py](../../../torchspec/utils/__init__.py) | 0 | 公共工具包命名空间：分布式组、日志、张量对齐与诊断工具分别在子模块实现，本文件无初始化副作用。 | [阅读](training.md) |
| [torchspec/utils/distributed.py](../../../torchspec/utils/distributed.py) | 17 | 进程组注册表：Gloo 用于控制对象/CPU 通信，USP 分别建立数据副本、序列分片及梯度同步 mesh。 | [阅读](training.md) |
| [torchspec/utils/env.py](../../../torchspec/utils/env.py) | 1 | Ray actor 环境转发策略：传递明确白名单和设备可见性控制，不转发跨节点共享的 Inductor 缓存路径。 | [阅读](training.md) |
| [torchspec/utils/logging.py](../../../torchspec/utils/logging.py) | 8 | 日志和 tracking 入口：控制台按 actor 区分，文件按 role/node/rank 分开；W&B/TensorBoard 管理指标输出。 | [阅读](training.md) |
| [torchspec/utils/memory.py](../../../torchspec/utils/memory.py) | 5 | 内存估算和观测辅助：形状元数据估算传输量，CUDA API 区分设备可用内存与 PyTorch 分配/保留量。 | [阅读](training.md) |
| [torchspec/utils/misc.py](../../../torchspec/utils/misc.py) | 5 | 运行环境辅助：节点地址、端口探测、GPU 可见性映射及默认 teacher 辅助层选择。 | [阅读](training.md) |
| [torchspec/utils/processing.py](../../../torchspec/utils/processing.py) | 2 | Tokenizer 与 assistant 边界适配：为 loss mask 提供与实际 renderer/chat template 一致的 token 标记。 | [阅读](training.md) |
| [torchspec/utils/profiling.py](../../../torchspec/utils/profiling.py) | 16 | 训练性能观测：PyTorch profiler 用调度窗口采集算子，内存 profiler 记录 GPU 分配历史或 CPU 原生分配。 | [阅读](training.md) |
| [torchspec/utils/tensor.py](../../../torchspec/utils/tensor.py) | 1 | 训练目标对齐的小型张量操作；padding 在这里表示定长位移，并不是扩展序列长度。 | [阅读](training.md) |
| [torchspec/utils/train_dump.py](../../../torchspec/utils/train_dump.py) | 9 | 训练数值诊断工具：保存 CPU 张量快照，比较输入、逐深度损失、梯度和权重，帮助定位两条执行配置的差异。 | [阅读](training.md) |
| [torchspec/utils/types.py](../../../torchspec/utils/types.py) | 0 | 跨编排/推理边界的数据协议：InferenceInput 携带 prompt/token/mask，InferenceOutput 携带 Mooncake key 与形状。 | [阅读](training.md) |
| [torchspec/utils/usp.py](../../../torchspec/utils/usp.py) | 2 | USP 输入切片：每个 SP rank 获得主块加 ttt 尾部，以支持多步监督跨块边界读取。 | [阅读](training.md) |
| [torchspec/utils/wandb.py](../../../torchspec/utils/wandb.py) | 5 | 多进程 W&B 接入：主进程建 run，其他 actor 通过 run id 关联；指标显式绑定各自 step 轴。 | [阅读](training.md) |
