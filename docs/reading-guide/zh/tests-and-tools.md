# 用测试、配置和工具把理解落到实处

[返回总路线](README.md) · [工具 Python 源码详解](tools.md)

测试不是阅读完源码后的附属品。模型代码通常为大 batch、并行设备和编译优化写得复杂，测试里的小模型与手写 reference 往往更适合建立第一幅图。先读输入构造，再读断言，最后追实现。

## 1. 按“想验证哪条理解”选择测试

| 想验证的理解 | 测试入口 | 阅读时做什么 |
|---|---|---|
| 软目标损失究竟算什么 | [test_eagle3_loss.py](../../../tests/test_eagle3_loss.py) | 对照 `_reference_forward_kl_loss` 与实现；确认 perfect prediction 的 CE 仍有 teacher 熵 |
| 裁剪词表如何编码 | [test_vocab_mapping.py](../../../tests/test_vocab_mapping.py) | 手算选中 token、差值 d2t 与布尔 t2d；检查仅监督位置计频率 |
| 多步 attention 看哪些位置 | [test_build_eagle3_block_mask.py](../../../tests/test_build_eagle3_block_mask.py) | 从小尺寸 dense reference 看可见性，再读块级结构 |
| anchor 版本与密集版本何时等价 | [test_anchored_eagle3.py](../../../tests/test_anchored_eagle3.py) | 看 context rows、独立 chain slot、不同深度梯度的断言 |
| DFlash 与 DSpark 标签为何差一位 | [test_dflash.py](../../../tests/test_dflash.py)、[test_dspark.py](../../../tests/test_dspark.py) | 写出 anchor+槽位对应真实 token 的索引；不要只比较输出 shape |
| DFlash2 selector 为什么需要 gold 候选 | [test_dflash2.py](../../../tests/test_dflash2.py) | 找替换最弱候选、teacher-forced transitions、prefix mask 的测试 |
| 一个 token 对、一批 token 错怎样平均 | [test_eagle3_trainer.py](../../../tests/test_eagle3_trainer.py) | 比较 token 加权结果和 naive mean-of-means |
| 空监督 rank 是否能跳过 | [test_data_fetcher.py](../../../tests/test_data_fetcher.py)、[test_eagle3_loss.py](../../../tests/test_eagle3_loss.py) | 追 neutralized sample、空 shard collective 与梯度图 |
| CUDA clone 是否已完成 | [test_data_fetcher.py](../../../tests/test_data_fetcher.py) | 看 receive buffer reuse 之前同步的要求，与 GPU 异步执行联系起来 |
| pre/post norm 有何区别 | [test_target_hidden.py](../../../tests/test_target_hidden.py)、[test_target_lm_head.py](../../../tests/test_target_lm_head.py) | 比较 norm 输入、norm 输出、最终返回状态，不依赖名字猜测 |
| 参数恢复为何不能遗漏 master | [test_optimizer_state_keys.py](../../../tests/test_optimizer_state_keys.py)、[test_continual_training_checkpoint.py](../../../tests/test_continual_training_checkpoint.py) | 看按名称配对 FP32 master、冻结/解冻改变参数集时如何处理 |
| 逻辑序列如何对应物理页 | [test_connector_slot_mapping.py](../../../tests/test_connector_slot_mapping.py) | 手算非连续 block ID 和最后半页；再追 connector |
| 异步写入何时可暴露错误 | [test_mooncake_async_publish.py](../../../tests/test_mooncake_async_publish.py)、[test_mooncake_connector_finished.py](../../../tests/test_mooncake_connector_finished.py) | 分清回包、Future 完成、PP barrier、消费端键等待 |
| PP 改变是否仍输出相同特征 | [test_vllm_pp_equivalence.py](../../../tests/test_vllm_pp_equivalence.py) | 看真实引擎验收的输入/容差与运行条件；不要把 mock 单测当成这个实验 |
| 离线回放能否替代在线生产者 | [test_offline_replay.py](../../../tests/test_offline_replay.py) | 检查输入 ID、norm 元数据和 live-engine 返回协议 |
| 格式化有没有改变监督 | [test_kimi_k3_renderer.py](../../../tests/test_kimi_k3_renderer.py)、[test_pretokenized_dataset.py](../../../tests/test_pretokenized_dataset.py) | 看真实 mask 与 ID 的匹配、控制符隔离和拒绝静默截断 |

注意“reference”也有适用范围。某测试覆盖了一个 backend 的小尺寸 forward，不自动证明所有 attention 后端、dtype、SP 拓扑和长上下文都正确。

## 2. 怎样读一个测试，而不被 mock 带偏

以 `test_partition_results.py` 为例：它要验证分区后的样本条数和 token 负载，不需要真实 GPU。先写下输入长度与 DP 数，再自己算分组，最后看断言是否正好表达这条行为。

但 Mooncake、Ray 与引擎测试可能替换网络、进程或 GPU 依赖。那些测试能检查协议分支、错误转发、元数据，却不能证明真实网卡传输吞吐或 CUDA 流顺序在硬件上完成。需要真实系统时，继续看文件名带 integration 的脚本和 PP 验收测试。

根目录 [conftest.py](../../../conftest.py) 在缺少 torch 时会为重依赖安装 mock 导入器。知道这一点后，就不会把本机测试可收集误解为部署环境完整，也不会把 mock 的返回值当成真实 tensor 运算。

## 3. 配置如何对应到代码

先对照 [configs/README.md](../../../configs/README.md)、[default.yaml](../../../configs/default.yaml) 和 [train_config.py](../../../torchspec/config/train_config.py)。YAML 只写部分字段，其余来自 dataclass schema；CLI 最后覆盖。

| 一组配置 | 代码阅读问题 |
|---|---|
| `model.target_model_path` 与 `draft_model_config` | 一个提供 teacher/初始共享权重来源，另一个定义草稿结构，是否被混用？ |
| `inference.inference_engine_type` | 选中工厂哪一个后端？是否需要补丁/特征捕获支持？ |
| `inference.*.tp_size` 与每引擎 GPU 数 | 引擎副本内部怎么分卡，工厂如何校验拓扑一致？ |
| `training.micro_batch_size` 与累积步数 | 一次更新真正消费多少独立样本？ |
| `training.ttt_length` / `dflash_block_size` | 是递归展开深度还是块几何？是哪一个 wrapper 消费？ |
| `dataset.renderer` / `chat_template` | 是谁决定 token 和监督区间？ |
| `inference.last_hidden_states_prenorm` | 是否需要额外 verifier norm，离线数据是否覆盖此设置？ |
| `mooncake.global_segment_size` / buffer sizes | 分别是分布式存储容量、写暂存还是读接收区？ |
| `training.load_path` / `continual_training` | 恢复旧优化过程还是用旧权重开新训练？ |

可阅读以下现成对照，而不急着启动作业：

- [HF 配置](../../../configs/hf_qwen3_8b.yaml)：最接近 `model(input_ids)` 的路径。
- [vLLM 配置](../../../configs/vllm_qwen3_8b.yaml) 与 [PP2 配置](../../../configs/vllm_qwen3_8b_pp2.yaml)：从单阶段到分层捕获。
- [DFlash](../../../configs/sglang_qwen3_8b_dflash.yaml)、[DFlash2](../../../configs/sglang_qwen3_8b_dflash2.yaml)、[DSpark](../../../configs/sglang_qwen3_8b_dspark.yaml)：比较结构配置与训练目标开关。
- [anchored EAGLE3](../../../configs/vllm_qwen3_8b_anchored.yaml)：比较查询采样与密集位置展开。
- [draft 模型目录](../../../configs/draft_models/README.md)：JSON 是参数结构定义，不支持注释；中文解释集中在模型教程与源码配置类。

旧示例注释可能写过已经不适用的字段名或拓扑。最终以当前参数解析和工厂校验为准，不要凭脚本开头一句“每卡一份模型”忽略后面实际传入的 `tp_size=2`。

## 4. 启动脚本按三段读

以 [qwen3-8b-single-node/run.sh](../../../examples/qwen3-8b-single-node/run.sh) 为例：

1. **环境段**：选择可见 GPU，设置编译缓存与日志。`CUDA_VISIBLE_DEVICES` 是进程可见设备列表，不是训练模型的 DP 数。
2. **配置段**：识别 YAML、推断后端名称，确定训练/推理卡数。脚本里的 CLI 覆盖可以覆盖 YAML，继续检查末尾的 `"$@"` 是否又提供覆盖项。
3. **执行段**：`python3 -m torchspec.train_entry` 才进入 Python 装配逻辑。不要因为 run.sh 短，就忽略它传入的资源参数。

多节点例子还包含 Ray head/worker 建群及地址协商，见 [examples/README.md](../../../examples/README.md)、[docs/ray.md](../../ray.md)。脚本不是声明式文档；构建环境、启动服务、清进程和提交 Slurm 的工具都有真实副作用。本注释任务只研读它们，没有执行训练或集群操作。

## 5. 基准测试要带着假设读

详见 [工具章节](tools.md)。至少区分：

- kernel benchmark：在已准备好的 tensor 上计某段 attention 的时间。
- 模型训练 step：包含 forward、backward、optimizer 及可能的数据等待。
- 端到端推理：还包含调度、草稿计算、验证、batch 和请求长度变化。

GPU 调用异步，纯 Python 墙钟计时如果不等待 CUDA 完成可能只测到提交开销。warmup 与编译时间是否计入、batch/序列长度/精度是否一致，决定两列数字能否比较。

`max_seq_search.py` 反复探测可运行的序列长度，`convert_to_hf.py` / `convert_to_torchspec.py` 在训练检查点与发布格式之间转换；两类工具不能凭“都是 Python 脚本”当成同一种轻量验证。

## 6. 测试目录逐文件导航

下面按文件列出真实测试名称的一部分，方便从一个具体行为跳回测试。该索引不声称这些测试已经在当前机器全部通过；实际执行结果见 [验证记录](verification.md)。现有测试本体保留，避免给一个复杂 fixture 加上未经运行验证的“必然行为”解释。

- [__init__.py](../../../tests/__init__.py): 测试包初始化。
- [test_anchored_eagle3.py](../../../tests/test_anchored_eagle3.py): `test_context_rows_match_the_dense_eagle3_mask`；`test_each_anchor_sees_only_its_own_chain_slot`；`test_window_keeps_the_last_w_context_keys`。共 20 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_build_eagle3_block_mask.py](../../../tests/test_build_eagle3_block_mask.py): `test_dense_mask_matches_reference`；`test_forward_matches_reference`；`test_backward_gradients_match_reference`。共 14 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_capacity_handling.py](../../../tests/test_capacity_handling.py): `test_single_tensor_float32`；`test_single_tensor_float16`；`test_single_tensor_bfloat16`。共 36 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_ci_convergence.py](../../../tests/test_ci_convergence.py): `test_matching_trajectories_pass`；`test_divergent_trajectory_fails`。共 2 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_connector_slot_mapping.py](../../../tests/test_connector_slot_mapping.py): `test_single_block`；`test_multiple_blocks`；`test_partial_last_block`。共 10 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_continual_training_checkpoint.py](../../../tests/test_continual_training_checkpoint.py): `test_restore_fp32_master_params_keeps_fresh_optimizer_hparams`。共 1 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_conversation_normalization.py](../../../tests/test_conversation_normalization.py): `test_sharegpt_normalization_preserves_extended_message_fields`；`test_sharegpt_role_and_content_win_over_stale_duplicates`；`test_reasoning_aliases_collapse_to_reasoning_content`。共 4 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_data_fetcher.py](../../../tests/test_data_fetcher.py): `test_clone_fragment_synchronizes_cuda_before_receive_buffer_reuse`；`test_iterates_samples`；`test_stops_on_none_sentinel`。共 15 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_data_utils.py](../../../tests/test_data_utils.py): `test_is_local_data_path_true`；`test_is_local_data_path_false_for_hub_ids`；`test_is_local_data_path_existing_dir`。共 4 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_deepseek_eagle.py](../../../tests/test_deepseek_eagle.py): `test_output_shape`；`test_cache_shapes_across_steps`；`test_gradients`。共 12 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_dflash.py](../../../tests/test_dflash.py): `test_config_attributes`；`test_config_serialization`；`test_single_layer`。共 86 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_dflash2.py](../../../tests/test_dflash2.py): `test_mla_mode_builds_k3_mla_attention`；`test_mixed_layer_types_build_per_layer_mask_policies`；`test_repository_config_dispatches_to_dflash2`。共 26 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_draft_from_pretrained.py](../../../tests/test_draft_from_pretrained.py): `test_base_classes_declare_an_empty_mapping`；`test_every_family_exposes_the_mapping`；`test_round_trip_preserves_every_parameter`。共 4 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_draft_model_config.py](../../../tests/test_draft_model_config.py): `test_generate_draft_model_config_preserves_rope_fields`；`test_generate_draft_model_config_copies_rope_scaling`；`test_generate_draft_model_config_fills_yarn_defaults`。共 3 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_draft_rope.py](../../../tests/test_draft_rope.py): `test_prefers_top_level_attribute`；`test_falls_back_to_rope_parameters`；`test_default_when_declared_nowhere`。共 8 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_dspark.py](../../../tests/test_dspark.py): `test_subclasses_dflash_and_attrs`；`test_ci_config_uses_standard_dspark_shape`；`test_optional_fc_norm_normalizes_each_target_layer_before_projection`。共 29 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_dynamic_last_turn_loss.py](../../../tests/test_dynamic_last_turn_loss.py): `test_think_tag_with_content`；`test_empty_think_tag`；`test_think_tag_whitespace_only`。共 39 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_eagle3_loss.py](../../../tests/test_eagle3_loss.py): `test_matches_reference`；`test_perfect_prediction_equals_entropy`；`test_loss_non_negative_and_finite`。共 28 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_eagle3_trainer.py](../../../tests/test_eagle3_trainer.py): `test_aggregate_metrics_weights_by_token_counts`；`test_naive_mean_of_means_would_disagree`；`test_aggregate_eval_metrics_weights_by_token_counts`。共 6 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_eagle4_moe.py](../../../tests/test_eagle4_moe.py): `test_matches_reference`；`test_gradients_match_reference`；`test_serving_layout`。共 4 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_env.py](../../../tests/test_env.py): `test_vllm_v2_model_runner_is_enabled_by_default`；`test_explicit_vllm_model_runner_value_is_forwarded`。共 2 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_fa4.py](../../../tests/test_fa4.py): `test_forward_pass_comparison`；`test_forward_pass_non_aligned`；`test_backward_pass_gradient_comparison`。共 8 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_flash_attn_fwd_compat.py](../../../tests/test_flash_attn_fwd_compat.py): `test_flash_attn_fwd_tolerates_trailing_returns`。共 1 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_flex_attention.py](../../../tests/test_flex_attention.py): `test_forward_pass_comparison`；`test_backward_pass_gradient_comparison`；`test_eagle3_flex_mask`。共 6 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_freeze_lm_head.py](../../../tests/test_freeze_lm_head.py): `test_freeze_only_detaches_the_head`；`test_frozen_head_leaves_the_trainable_set`；`test_load_lm_head_copies_the_target_rows`。共 15 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_glm45_parser.py](../../../tests/test_glm45_parser.py): `test_template_registered`；`test_template_attributes`；`test_dispatches_to_general_parser`。共 6 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_hf_engine.py](../../../tests/test_hf_engine.py): `test_init_stores_config`；`test_init_with_different_ranks`；`test_init_without_base_gpu_id`。共 16 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_hf_runner.py](../../../tests/test_hf_runner.py): `test_init_mooncake_store_with_config`；`test_init_mooncake_store_with_explicit_config`；`test_init_mooncake_store_raises_without_config`。共 9 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_initial_dflash_weights.py](../../../tests/test_initial_dflash_weights.py): `test_shared_weights_loaded_new_modules_retained`；`test_reject_incomplete_or_incompatible_weights`；`test_complete_dflash2_checkpoint_loads_selector`。共 3 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_initial_draft_model_path.py](../../../tests/test_initial_draft_model_path.py): `test_accepts_the_file_itself`；`test_accepts_the_directory_holding_it`；`test_expands_user_home`。共 28 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_initial_draft_upgrades.py](../../../tests/test_initial_draft_upgrades.py): `test_base_checkpoint_preserves_shared_weights_and_new_initialization`；`test_complete_upgraded_checkpoint_restores_all_new_weights`；`test_bad_checkpoint_is_rejected_without_modifying_model`。共 6 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_kimi_k25_integration.py](../../../tests/test_kimi_k25_integration.py): `test_converted_data_to_parser`；`test_multi_turn_parsing`；`test_preprocess_single_sample`。共 12 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_kimi_k25_parser.py](../../../tests/test_kimi_k25_parser.py): `test_template_registered`；`test_template_attributes`；`test_parser_initialization`。共 42 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_kimi_k3_renderer.py](../../../tests/test_kimi_k3_renderer.py): `test_kimi_k3_renderer_is_registered_and_created_with_tokenizer`；`test_kimi_k3_assistant_match_tokens_include_outer_sep`；`test_training_matcher_reproduces_k3_mask_after_media_expansion`。共 13 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_length_grouping.py](../../../tests/test_length_grouping.py): `test_length_hint_reads_whichever_field_is_present`；`test_sorts_longest_first_within_each_chunk`；`test_trailing_partial_chunk_is_sorted_too`。共 12 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_loop_cleanup.py](../../../tests/test_loop_cleanup.py): `test_write_training_metrics_logs_each_step_loss`；`test_training_loop_prefills_and_refills_controller_queue`；`test_training_loop_prefetch_zero_preserves_one_step_dispatching`。共 16 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_loss_mask.py](../../../tests/test_loss_mask.py): `test_single_turn`；`test_two_turns`；`test_no_end_token`。共 30 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_loss_mask_cross_validation.py](../../../tests/test_loss_mask_cross_validation.py): `test_multi_turn`；`test_multi_turn_last_turn_only`；`test_leading_newlines`。共 3 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_low_rank_lm_head.py](../../../tests/test_low_rank_lm_head.py): `test_shapes_and_parameter_count`；`test_forward_matches_the_materialized_product`；`test_divisor_must_shrink_the_head`。共 16 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_minimax_parser.py](../../../tests/test_minimax_parser.py): `test_template_registered`；`test_template_attributes`；`test_parser_initialization`。共 35 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_mooncake_async_publish.py](../../../tests/test_mooncake_async_publish.py): `test_connector_wait_for_save_checks_errors_without_flush`；`test_eagle_raw_put_validates_then_delegates`；`test_gpu_direct_put_waits_for_staging_before_rdma_read`。共 13 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_mooncake_connector_finished.py](../../../tests/test_mooncake_connector_finished.py): `test_request_finished_queues_completed_save_and_delays_blocks`；`test_request_finished_all_groups_selects_hidden_state_group`；`test_build_connector_meta_moves_pending_saves_to_worker`。共 15 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_mooncake_force_delete.py](../../../tests/test_mooncake_force_delete.py): `test_enable_hard_pin_env_roundtrip`；`test_enable_hard_pin_default_off`；`test_tcp_memcpy_default_is_applied_by_export_env`。共 22 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_mooncake_master.py](../../../tests/test_mooncake_master.py): `test_uses_mooncake_build_dir_env`；`test_uses_system_path_if_available`；`test_falls_back_to_default_path`。共 17 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_mtp.py](../../../tests/test_mtp.py): `test_pack_qwen35_qkv_groups_queries_gates_key_value`；`test_qwen35_mtp_config_reads_nested_text_config`；`test_mtp_objective_scores_row_t_against_token_t_plus_2`。共 4 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_offline_replay.py](../../../tests/test_offline_replay.py): `test_offline_dataset_round_trip_and_resume`；`test_offline_dataset_accepts_dflash_hidden_states_only`；`test_offline_dataset_rejects_cross_split_ids_and_missing_tensors`。共 16 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_optimizer.py](../../../tests/test_optimizer.py): `test_finite_gradient_updates_parameters`；`test_nonfinite_gradient_skips_entire_adamw_update`；`test_remote_nonfinite_signal_skips_local_finite_update`。共 3 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_optimizer_state_keys.py](../../../tests/test_optimizer_state_keys.py): `test_names_pair_with_fp32_params_positionally`；`test_freezing_the_head_drops_only_its_name`；`test_state_dict_is_keyed_by_name`。共 5 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_partition_results.py](../../../tests/test_partition_results.py): `test_single_dp_rank_keeps_all_samples_together`；`test_one_sample_per_rank_uses_round_robin`；`test_empty_results_returns_empty_partitions`。共 11 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_placement_group.py](../../../tests/test_placement_group.py): `test_custom_ip_bundles_add_node_resources`；`test_custom_label_bundles_add_bundle_label_selectors`；`test_custom_bundles_validate_node_count`。共 12 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_pretokenized_dataset.py](../../../tests/test_pretokenized_dataset.py): `test_pretokenized_rows_are_loaded_without_rerendering`；`test_producer_columns_are_carried_through_as_metadata`；`test_non_scalar_columns_are_not_pinned_in_metadata`。共 12 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_renderer_registry.py](../../../tests/test_renderer_registry.py): `test_registry_creates_renderers_with_the_target_tokenizer`；`test_registry_rejects_duplicate_registration`；`test_cache_version_falls_back_to_the_protocol_default`。共 17 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_save_config.py](../../../tests/test_save_config.py): `test_save_resolved_config_basic`；`test_save_resolved_config_with_cli_overrides`；`test_save_resolved_config_empty_output_dir`。共 13 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_sgl_engine_decode.py](../../../tests/test_sgl_engine_decode.py): `test_generate_with_decode_passes_min_new_tokens`；`test_generate_with_decode_passes_stop_token_ids`；`test_generate_with_decode_rejects_invalid_min_new_tokens`。共 7 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_sgl_engine_factory.py](../../../tests/test_sgl_engine_factory.py): `test_single_node_4_engines`；`test_single_node_1_engine`；`test_single_node_gpu_ids`。共 16 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_sglang_engine_integration.py](../../../tests/test_sglang_engine_integration.py): 没有 test_ 命名函数；阅读文件的命令行/main 入口与集成运行条件。
- [test_target_hidden.py](../../../tests/test_target_hidden.py): `test_capture_returns_both_sides_of_the_final_norm`。共 1 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_target_lm_head.py](../../../tests/test_target_lm_head.py): `test_rms_norm_fallback_loads_weight_and_matches_reference`；`test_rms_norm_fallback_creates_matching_non_rank_zero_structure`；`test_explicit_norm_load_fails_closed_without_a_supported_norm`。共 10 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_training_config_validation.py](../../../tests/test_training_config_validation.py): `test_load_config_rejects_zero_micro_batch_size`；`test_load_config_rejects_negative_micro_batch_size`；`test_load_config_accepts_positive_micro_batch_size`。共 13 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_usp_attention.py](../../../tests/test_usp_attention.py): `test_usp_matches_flex_loss_and_gradients`；`test_usp_ring_matches_flex_loss_and_gradients`。共 2 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_vllm_engine.py](../../../tests/test_vllm_engine.py): `test_connector_assigns_each_layer_to_one_pp_rank`；`test_alphanumeric_unchanged`；`test_special_chars_replaced`。共 62 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [test_vllm_engine_integration.py](../../../tests/test_vllm_engine_integration.py): 没有 test_ 命名函数；阅读文件的命令行/main 入口与集成运行条件。
- [test_vllm_pp_equivalence.py](../../../tests/test_vllm_pp_equivalence.py): 没有 test_ 命名函数；阅读文件的命令行/main 入口与集成运行条件。
- [test_vocab_mapping.py](../../../tests/test_vocab_mapping.py): `test_packed_loss_mask_format`；`test_multi_turn_packed_loss_mask`；`test_only_masked_tokens_counted`。共 12 个 test_ 函数/方法定义（参数化可能产生更多用例）。
- [utils.py](../../../tests/utils.py): 测试共用工具；跟随具体测试调用阅读。
