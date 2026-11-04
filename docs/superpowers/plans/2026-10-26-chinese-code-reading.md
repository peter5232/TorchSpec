# TorchSpec 中文源码阅读 Implementation Plan

> **For agentic workers:** Use scoped parallel documentation tasks; each worker reads actual implementations and verifies comment-only edits. Steps use checkbox syntax for tracking.

**Goal:** 为熟悉 PyTorch 模型训练、尚不了解推理系统的读者提供源码内详细中文讲解和分层阅读教程。

**Architecture:** 以一条训练样本的生命周期贯穿全仓，从单进程张量计算延伸到异步生产消费与分布式。源码解释局部实现，教程解释跨文件设计和阅读顺序；不复制源码作为教程。

**Tech Stack:** Python 注释、Markdown、现有测试作为阅读材料、AST/token 对比验证。

**Spec:** 用户要求全仓详细中文注释、设计及架构考量、函数实现解释和复杂处示例；已明确授权按推荐方案直接执行。

## Global Constraints

- 在 `codex/torchspec-annotated-zh` 工作；不改算法、函数签名、默认参数、导入或运行行为。
- 只添加 `#` 中文注释，保留原有英文文档字符串和许可证。
- 每个非空生产模块介绍职责、上下游、关键约束；每个函数说明真实作用、关键输入输出或副作用；复杂逻辑另加就地解释和具体例子。
- 读源码后写解释，不依据函数名机械生成模板，不把猜测写成设计事实。
- `torchspec/` 全覆盖；测试、工具、脚本和配置在扩展阅读章节建立索引，并讲解关键入口。
- 不安装全局依赖，不运行训练、不访问模型权重、不推送远端。

## Review Focus

- 训练中的多步展开与在线推理的接受/拒绝不能混为一谈。
- teacher hidden states 的 pre/post norm、位置偏移和词表裁剪需与实现一致。
- Ray 元数据队列、Mooncake 张量传输以及 CUDA buffer 生命周期需区分。
- DP/TP/SP、梯度累积和 loss 分母不能用普通单卡 batch 概念掩盖。
- 注释覆盖率不等于质量；抽查复杂分支、例子和跨文件引用。

## Tasks

- [x] 模型：`torchspec/models/**`，教程 `docs/reading-guide/zh/models.md`。
- [x] 训练：`torchspec/training/**`、`torchspec/utils/**`、`torchspec/megatron.py`，教程 `training.md`。
- [x] 推理/传输：`torchspec/inference/**`、`torchspec/transfer/**`，教程 `inference-transfer.md`。
- [x] 编排/输入：入口、config、controller、ray、data、offline，教程 `pipeline.md`。
- [x] 主阅读路线、术语与数值例子、全文件覆盖索引、测试/脚本/配置扩展路线、README 入口。
- [x] 对比全部 Python AST 和有效 token；检查 Markdown 本地链接；核对覆盖和准确性；运行可用格式检查。

## Execution notes

用户已明确跳过方案问答。当前 checkout 干净且处于新注释分支，复用当前 checkout。无本地 venv；纯注释不新增行为单测，以原始代码 AST/有效 token 完全相等验证不改变行为；现有行为测试作为阅读证据。

完成记录：生产模块及 tools/scripts Python 源码均已添加详细注释；测试与配置保持原实现，通过教程索引阅读。AST、有效 token、语法与链接验证通过。pytest 因独立 import torch 即可复现的 OpenMP 冲突退出；初次环境缺少 pre-commit；2026-11-04 已通过 uv 隔离环境补跑完整 hooks，并执行 2 项无需 PyTorch 的收敛轨迹测试。详细边界见 docs/reading-guide/zh/verification.md。
