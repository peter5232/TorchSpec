# 注释范围与验证记录

本阅读分支基于 `a4e5c1085bfc5990ddcae6cf5832bb26380b38d3`，给现有 Python 文件增加中文 `#` 注释并应用注释相关格式调整，并增加阅读教程与 README 入口。保留原有许可证、英文 docstring、算法、函数签名和默认值。

## 覆盖范围

- `torchspec/` 的 106 个 Python 模块：1048 个函数/方法定义（包含嵌套函数）均有中文讲解，复杂实现另有步骤、形状、生命周期和边界说明。
- `tools/` 与 `scripts/` 的 12 个 Python 文件：118 个函数/方法，覆盖转换、数据生成、基准和诊断工具。
- 测试通过[测试阅读路线](tests-and-tools.md)建立索引；配置与启动脚本通过实际入口讲解。没有给测试或第三方补丁机械添加逐行注释，也没有改写测试行为。
- [源码地图](source-map.md)列出全部生产模块；[主阅读路线](README.md)组织从模型张量到分布式系统的学习顺序。

注释覆盖不等于正确性证明。交叉复核重点包括 token 对齐、loss 分母、生产端 USP 分片、Mooncake 接收区复用、异步 H2D 与后台保存的完成时机；发现的歧义已按实际调用链修订。源码现有的不一致仅说明，不在注释分支中修改算法，例如[转换工具的边界](tools.md)。

## 静态验证

全部 118 个修改过的 Python 文件均通过语法编译，并与基线比较：

1. `ast.dump(ast.parse(...))` 完全一致，确认语法树、常量及 docstring 没有改变。
2. 去除 COMMENT、NL、ENCODING 后的 `(token.type, token.string)` 序列完全一致；保留 NEWLINE、INDENT、DEDENT 的检查，防止注释编辑误改可执行语句与缩进。
3. 函数前中文注释覆盖与教程本地文件链接检查通过；`git diff --check` 通过。

在仓库根目录可用标准库复核核心等价性，不需要导入 PyTorch：

```python
import ast
import io
import subprocess
import tokenize
from pathlib import Path

baseline = "a4e5c1085bfc5990ddcae6cf5832bb26380b38d3"
ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}

def tokens(source):
    return [(t.type, t.string)
            for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in ignored]

files = subprocess.check_output(
    ["git", "diff", "--name-only", baseline], text=True
).splitlines()
for name in files:
    if not name.endswith(".py"):
        continue
    before = subprocess.check_output(
        ["git", "show", f"{baseline}:{name}"], text=True
    )
    after = Path(name).read_text()
    assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after)), name
    assert tokens(before) == tokens(after), name
    compile(after, name, "exec")
print("Python AST、有效 token 和语法编译检查通过")
```

这证明本次 Python 修改不改变可执行内容，不证明原项目算法无缺陷，也不等于运行过 GPU 训练或推理。

## 环境限制与未完成的运行检查

尝试执行以下现有测试，但进程在收集阶段前退出，退出码为 134：

```sh
python3 -m pytest tests/test_ci_convergence.py tests/test_target_hidden.py tests/test_conversation_normalization.py tests/test_partition_results.py tests/test_vocab_mapping.py -q
```

单独执行 `python3 -X faulthandler -c 'import torch; print(torch.__version__)'` 即可复现：`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.` 因此不能报告这些测试通过。没有通过绕过 OpenMP 检查来掩盖环境冲突，没有运行 GPU、分布式训练或需要下载权重的测试。

2026-11-04 补充验证：通过 uv 隔离工具环境运行完整的 `pre-commit run --all-files`（命令为 `uv tool run --from pre-commit pre-commit run --all-files`）。首次运行修正了两处注释相关格式，修正后重新运行，全部适用 hooks 通过（无符号链接可检查的 hook 跳过）；AST、有效 token、语法编译和 535 个本地链接再次验证通过。未安装全局依赖。

另以 `python3 -m unittest discover -s tests -p test_ci_convergence.py -v` 执行不依赖 PyTorch 的现有收敛轨迹测试，2 项通过。该结果不替代上面受阻的 PyTorch 测试。

## 第二轮：实现内部的深入演算

在首轮提交 `f51a04a` 上继续深化 15 个核心 Python 文件，增加 257 行中文注释，重点是函数体内的张量变换、坐标与字节布局、梯度通路和异步完成边界。新增三篇配套演算：监督与损失、模型小张量、跨进程传输时序；入口均在主阅读指南。

本轮修改与 `f51a04a` 的 AST、有效 token 完全相同，15 个修改文件均可编译；全部教程的 556 个本地链接有效。完整 pre-commit hooks 通过，两个现有收敛轨迹测试再次通过。手算例子的 CE、熵、KL、概率重合质量和锚点回绕结果另以标准库复算，并交叉核对源码。

这些例子是帮助理解实现的推导，不是 GPU kernel 或真实网络传输的运行结果。本轮未重新运行此前受 OpenMP 环境冲突阻塞的 PyTorch 测试，也未改变预取结束、缓存 padding 或异步错误处理等现有行为。

## 第三轮：调度、恢复与离线数据（2026-11-11）

在 `e912c2e` 上深化 10 个 Python 文件，净增加 205 行注释，并修正清理函数适用范围的旧描述。新增控制器、主循环、checkpoint、离线数据四篇演算，分别追踪逻辑样本/分片计数、累积窗口、训练状态与文件发布，以及 manifest 和回放 key 的生命周期。

10 个文件与本轮基线的 AST、有效 token 完全一致，语法编译通过；教程全部 586 个本地链接有效。完整 pre-commit hooks 与 `git diff --check` 通过，两个既有收敛轨迹测试再次通过。交叉审阅补明了 RPC 组异常返回不代表其他调用已结束，以及 try 之前的提交失败不经过 finally。

checkpoint、异步队列和离线磁盘故障场景均为依据源码的推演；没有运行 GPU/分布式恢复、网络故障注入或断电测试。现有实现的非事务边界仅加以解释，本轮不修改其行为。
