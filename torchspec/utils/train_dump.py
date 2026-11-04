# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# 训练数值诊断工具：保存 CPU 张量快照，比较输入、逐深度损失、梯度和权重，帮助定位两条执行配置的差异。
# 比较结果仅覆盖共同存在的字段/文件；不能把 match=True 当作所有训练状态完全相同的证明。

"""Debug utilities for saving and comparing training data.

Used to verify correctness between train_entry.py and train/train_eagle3.py.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from torchspec.utils.logging import logger


# 只提取已有 grad 的参数，detach+clone 后搬 CPU，避免后续 zero_grad/step 改写快照。
def extract_gradients(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract gradients from model parameters.

    Args:
        model: The model to extract gradients from.

    Returns:
        Dict mapping parameter names to gradient tensors (cloned and detached).
    """
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone().cpu()
    return gradients


# 按可选前缀筛参数，detach+clone 并搬 CPU，得到与后续权重更新隔离的调试副本。
def extract_model_weights(model: nn.Module, filter_prefix: str = None) -> Dict[str, torch.Tensor]:
    """Extract model weights from parameters.

    Args:
        model: The model to extract weights from.
        filter_prefix: Optional prefix to filter parameter names (e.g., 'draft_model').

    Returns:
        Dict mapping parameter names to weight tensors (cloned and detached).
    """
    weights = {}
    for name, param in model.named_parameters():
        if filter_prefix is not None and not name.startswith(filter_prefix):
            continue
        weights[name] = param.detach().clone().cpu()
    return weights


# 按 1 起点步号判断是否位于前 max_dump_steps；不检查路径或执行 I/O。
def should_dump_step(step: int, max_dump_steps: int = 5) -> bool:
    """Check if current step should be dumped.

    Args:
        step: Current training step (1-indexed).
        max_dump_steps: Maximum number of steps to dump.

    Returns:
        True if step should be dumped, False otherwise.
    """
    return step <= max_dump_steps


# 把路径模板中的 step/rank/batch_idx 展开，收集输入与可选损失/梯度/权重并 torch.save。
# 张量转换为 CPU 以便离线比较，可能触发设备同步，因此只用于少量调试步骤。
def dump_eagle3_batch(
    args,
    *,
    step: int,
    batch_idx: int,
    batch: Dict[str, torch.Tensor],
    plosses: Optional[List[torch.Tensor]] = None,
    acces: Optional[List[torch.Tensor]] = None,
    gradients: Optional[Dict[str, torch.Tensor]] = None,
    total_loss: Optional[torch.Tensor] = None,
    model_weights: Optional[Dict[str, torch.Tensor]] = None,
    hidden_states: Optional[torch.Tensor] = None,
    target: Optional[torch.Tensor] = None,
    last_hidden_states: Optional[torch.Tensor] = None,
):
    """Save Eagle3 training batch and metrics for debugging.

    Args:
        args: Arguments containing save_debug_train_data path template.
        step: Current optimizer step.
        batch_idx: Batch index within the step.
        batch: Dict of input tensors (input_ids, attention_mask, etc.).
        plosses: Optional list of per-position losses.
        acces: Optional list of per-position accuracies.
        gradients: Optional dict mapping param names to gradient tensors.
        total_loss: Optional total weighted loss scalar.
        model_weights: Optional dict mapping param names to weight tensors.
        hidden_states: Optional aux hidden states tensor (intermediate layers).
        target: Optional target logits tensor.
        last_hidden_states: Optional last hidden states tensor (final layer before lm_head).
    """
    path_template = getattr(args, "save_debug_train_data", None)
    if path_template is None:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # 路径模板应包含区分 rank/step/batch 的字段，否则并行写入可能覆盖同一个文件。
    # 本函数按用户模板落盘，不额外替用户构造唯一文件名。
    path = Path(path_template.format(step=step, rank=rank, batch_idx=batch_idx))
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "step": step,
        "batch_idx": batch_idx,
        "rank": rank,
        "batch": {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
    }

    # 可选项只在调用者提供时保存；没有某字段与该字段数值为零是不同情况。
    # detach 防止把计算图带入快照，CPU 化便于跨机器离线比较，但会引入传输等待。
    if plosses is not None:
        data["plosses"] = [pl.detach().cpu() for pl in plosses]
    if acces is not None:
        data["acces"] = [ac.detach().cpu() for ac in acces]
    if gradients is not None:
        data["gradients"] = {k: v.cpu() for k, v in gradients.items()}
    if total_loss is not None:
        data["total_loss"] = total_loss.detach().cpu()
    if model_weights is not None:
        data["model_weights"] = {k: v.cpu() for k, v in model_weights.items()}
    if hidden_states is not None:
        data["hidden_states"] = hidden_states.detach().cpu()
    if target is not None:
        data["target"] = target.detach().cpu()
    if last_hidden_states is not None:
        data["last_hidden_states"] = last_hidden_states.detach().cpu()

    logger.info(f"Save Eagle3 debug batch to {path}")
    torch.save(data, path)


# 读取两份 dump，比较共同输入/损失字段及参数梯度/权重，返回 match、diffs 和键集合摘要。
# 参数名字缺失会记错，但未同时存在的可选 batch/指标字段会跳过，需结合覆盖范围解释结果。
def compare_eagle3_outputs(
    path1: str,
    path2: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = False,
) -> Dict:
    """Compare two Eagle3 training dumps for correctness verification.

    Args:
        path1: Path to first dump file.
        path2: Path to second dump file.
        rtol: Relative tolerance for torch.allclose.
        atol: Absolute tolerance for torch.allclose.
        verbose: If True, print detailed comparison info.

    Returns:
        Dict with comparison results including match status and diffs.
    """
    data1 = torch.load(path1, weights_only=False)
    data2 = torch.load(path2, weights_only=False)

    results = {
        "match": True,
        "diffs": {},
        "summary": {},
    }

    # 先查形状；整数 token/mask 必须完全相等，浮点转 float32 后按 atol/rtol 比较。
    # 失败记录差异计数或最大/平均绝对误差，verbose 输出每个已检查张量的结果。
    def _compare_tensors(t1, t2, name: str):
        """Compare two tensors and record differences."""
        if t1.shape != t2.shape:
            results["match"] = False
            results["diffs"][name] = {"error": f"Shape mismatch: {t1.shape} vs {t2.shape}"}
            return

        if t1.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            match = torch.equal(t1, t2)
            if not match:
                diff_count = (t1 != t2).sum().item()
                results["match"] = False
                results["diffs"][name] = {"diff_count": diff_count, "total": t1.numel()}
        else:
            match = torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol)
            if not match:
                diff = (t1.float() - t2.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                results["match"] = False
                results["diffs"][name] = {"max_diff": max_diff, "mean_diff": mean_diff}

        if verbose:
            status = "✓" if name not in results["diffs"] else "✗"
            print(f"  {status} {name}: shape={tuple(t1.shape)}, dtype={t1.dtype}")

    # Compare batch inputs
    # 建议先核对输入再解释梯度差异：mask/位置错一位会让整条监督链都不一致。
    # 此循环只检查两边共同有的输入字段，因此阅读报告时还要人工确认两边 dump 的采集配置一致。
    for key in ["input_ids", "attention_mask", "loss_mask", "target", "hidden_states"]:
        if key not in data1.get("batch", {}) or key not in data2.get("batch", {}):
            continue
        t1 = data1["batch"][key]
        t2 = data2["batch"][key]
        _compare_tensors(t1, t2, f"batch/{key}")

    # Compare per-position losses
    for metric_key in ["plosses", "acces"]:
        if metric_key in data1 and metric_key in data2:
            if len(data1[metric_key]) != len(data2[metric_key]):
                results["match"] = False
                results["diffs"][metric_key] = {
                    "error": f"Length mismatch: {len(data1[metric_key])} vs {len(data2[metric_key])}"
                }
                continue
            for i, (m1, m2) in enumerate(zip(data1[metric_key], data2[metric_key])):
                _compare_tensors(m1, m2, f"{metric_key}[{i}]")

    # Compare total loss
    if "total_loss" in data1 and "total_loss" in data2:
        _compare_tensors(data1["total_loss"], data2["total_loss"], "total_loss")

    # Compare gradients
    # 同名梯度逐项比较；仅一边有的参数名会使整体 match=False，并最多展示 10 个名字。
    # 这有助于发现冻结策略变化，但某参数没有 grad 也可能是本 batch 未经过该分支。
    if "gradients" in data1 and "gradients" in data2:
        grads1, grads2 = data1["gradients"], data2["gradients"]
        common_keys = set(grads1.keys()) & set(grads2.keys())
        only_in_1 = set(grads1.keys()) - set(grads2.keys())
        only_in_2 = set(grads2.keys()) - set(grads1.keys())

        results["summary"]["gradient_keys"] = {
            "common": len(common_keys),
            "only_in_file1": len(only_in_1),
            "only_in_file2": len(only_in_2),
        }

        if only_in_1 or only_in_2:
            results["match"] = False
            if only_in_1:
                results["diffs"]["gradients_only_in_file1"] = list(only_in_1)[:10]
            if only_in_2:
                results["diffs"]["gradients_only_in_file2"] = list(only_in_2)[:10]

        for key in sorted(common_keys):
            _compare_tensors(grads1[key], grads2[key], f"gradients/{key}")

    # Compare model weights
    # 权重快照需要在相同训练边界采集，例如都在 optimizer.step 之前。
    # 若一个在更新前、另一个在更新后，即使算法相同，也不能期待逐参数比较相等。
    if "model_weights" in data1 and "model_weights" in data2:
        weights1, weights2 = data1["model_weights"], data2["model_weights"]
        common_keys = set(weights1.keys()) & set(weights2.keys())
        only_in_1 = set(weights1.keys()) - set(weights2.keys())
        only_in_2 = set(weights2.keys()) - set(weights1.keys())

        results["summary"]["weight_keys"] = {
            "common": len(common_keys),
            "only_in_file1": len(only_in_1),
            "only_in_file2": len(only_in_2),
        }

        if only_in_1 or only_in_2:
            results["match"] = False
            if only_in_1:
                results["diffs"]["weights_only_in_file1"] = list(only_in_1)[:10]
            if only_in_2:
                results["diffs"]["weights_only_in_file2"] = list(only_in_2)[:10]

        for key in sorted(common_keys):
            _compare_tensors(weights1[key], weights2[key], f"model_weights/{key}")

    return results


# 按同名文件求交集并逐对比较，另列目录独有文件；独有文件本身不会把 match 设为 False。
# 因此 files_compared=0 的成功不代表完成了有效数值验证。
def compare_eagle3_batches(
    dir1: str,
    dir2: str,
    pattern: str = "*.pt",
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True,
) -> Dict:
    """Compare all matching Eagle3 dumps in two directories.

    Args:
        dir1: First directory path.
        dir2: Second directory path.
        pattern: Glob pattern for finding dump files.
        rtol: Relative tolerance for torch.allclose.
        atol: Absolute tolerance for torch.allclose.
        verbose: If True, print comparison progress.

    Returns:
        Dict with overall match status and per-file results.
    """
    dir1, dir2 = Path(dir1), Path(dir2)
    files1 = {f.name: f for f in dir1.glob(pattern)}
    files2 = {f.name: f for f in dir2.glob(pattern)}

    # 以文件名而非文件顺序匹配，所以 dump 模板应把 step/rank/batch 编进名字。
    # 仅存在于一个目录的文件被列入摘要，但当前函数不把这种缺失自动计为数值不匹配。
    common_files = set(files1.keys()) & set(files2.keys())

    overall_results = {
        "match": True,
        "files_compared": len(common_files),
        "files_only_in_dir1": list(set(files1.keys()) - common_files),
        "files_only_in_dir2": list(set(files2.keys()) - common_files),
        "per_file_results": {},
    }

    # 逐文件调用单对比较并折叠 match；没有共同文件时循环为空，默认 True 保持不变。
    # 因此验证报告必须同时看 files_compared 和独有文件列表，不能只读取一个布尔值。
    for filename in sorted(common_files):
        if verbose:
            print(f"\nComparing {filename}...")
        result = compare_eagle3_outputs(
            str(files1[filename]),
            str(files2[filename]),
            rtol=rtol,
            atol=atol,
            verbose=verbose,
        )
        overall_results["per_file_results"][filename] = result
        if not result["match"]:
            overall_results["match"] = False

    if verbose:
        status = "✓ All match" if overall_results["match"] else "✗ Differences found"
        print(f"\nOverall: {status}")
        print(f"  Files compared: {len(common_files)}")
        if overall_results["files_only_in_dir1"]:
            print(f"  Files only in dir1: {overall_results['files_only_in_dir1']}")
        if overall_results["files_only_in_dir2"]:
            print(f"  Files only in dir2: {overall_results['files_only_in_dir2']}")

    return overall_results


# 输出 dump 的步号、张量 shape/dtype、loss/accuracy；梯度和权重最多列前 20 项范数。
def load_and_print_eagle3_dump(path: str, show_gradients: bool = False):
    """Load and print summary of an Eagle3 training dump.

    Args:
        path: Path to the dump file.
        show_gradients: If True, also print gradient summary.
    """
    data = torch.load(path, weights_only=False)

    print(f"=== Eagle3 Training Dump: {path} ===")
    print(f"Step: {data.get('step', data.get('inference_id', 'N/A'))}")
    print(f"Batch IDX: {data.get('batch_idx', 'N/A')}")
    print(f"Rank: {data.get('rank', 'N/A')}")

    # 先展示输入形状/dtype，再看标量损失和逐深度统计，帮助确认 dump 来自期望配置。
    # 这里只打印摘要，不会重新前向计算，也不证明保存的数值和模型状态彼此一致。
    if "batch" in data:
        print("\nBatch tensors:")
        for key, tensor in data["batch"].items():
            if torch.is_tensor(tensor):
                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    if "total_loss" in data:
        print(f"\nTotal loss: {data['total_loss'].item():.6f}")

    if "plosses" in data:
        print("\nPosition losses:")
        for i, pl in enumerate(data["plosses"]):
            print(f"  pos_{i}: {pl.item():.6f}")

    if "acces" in data:
        print("\nPosition accuracies:")
        for i, ac in enumerate(data["acces"]):
            print(f"  pos_{i}: {ac.item():.4f}")

    # 梯度最多显示按名字排序的前 20 个条目，输出范数便于定位大尺度异常。
    # 没有显示的参数仍可能存在于文件；要验证全部参数应调用 compare_eagle3_outputs。
    if show_gradients and "gradients" in data:
        print(f"\nGradients ({len(data['gradients'])} parameters):")
        for name, grad in sorted(data["gradients"].items())[:20]:
            norm = grad.float().norm().item()
            print(f"  {name}: shape={tuple(grad.shape)}, norm={norm:.6f}")
        if len(data["gradients"]) > 20:
            print(f"  ... and {len(data['gradients']) - 20} more parameters")

    if "model_weights" in data:
        print(f"\nModel weights ({len(data['model_weights'])} parameters):")
        for name, weight in sorted(data["model_weights"].items())[:20]:
            norm = weight.float().norm().item()
            print(f"  {name}: shape={tuple(weight.shape)}, norm={norm:.6f}")
        if len(data["model_weights"]) > 20:
            print(f"  ... and {len(data['model_weights']) - 20} more parameters")


# 只打印分别生成两组 dump 及比较的命令示例，并返回目录路径，不实际启动训练任务。
# 示例中包含旧版入口名，使用时应核对当前仓库入口和配置格式。
def run_comparison_test(
    config1_path: str,
    config2_path: str,
    num_steps: int = 1,
    output_dir: str = "./debug_dumps",
):
    """Run both configs and compare their outputs.

    This is a helper function for running comparison tests.
    Actual training execution must be done separately.

    Args:
        config1_path: Path to first config file.
        config2_path: Path to second config file.
        num_steps: Number of training steps to run.
        output_dir: Directory for debug dumps.

    Returns:
        Dict with paths to run for comparison.
    """
    # 这只是教学/诊断命令生成器：不会执行配置、创建模型、提交 Ray 任务或采集数据。
    # 需要用户在真正训练环境分别生成快照，再运行比较；num_steps 这里只插入打印的命令文本。
    dir1 = Path(output_dir) / "config1"
    dir2 = Path(output_dir) / "config2"

    print("To compare training outputs from two configs:")
    print()
    print("1. Run config1 with debug dumps enabled:")
    print(f"   python -m torchspec.train.train_eagle3 --config {config1_path} \\")
    print(
        f"       --save_debug_train_data '{dir1}/batch_{{step}}_{{batch_idx}}_rank{{rank}}.pt' \\"
    )
    print(f"       --max_num_steps {num_steps}")
    print()
    print("2. Run config2 with debug dumps enabled:")
    print(f"   python -m torchspec.train_entry --config {config2_path} \\")
    print(
        f"       debug.save_debug_train_data='{dir2}/batch_{{step}}_{{batch_idx}}_rank{{rank}}.pt' \\"
    )
    print(f"       training.num_train_steps={num_steps}")
    print()
    print("3. Compare results:")
    print("   from torchspec.utils.train_dump import compare_eagle3_batches")
    print(f"   results = compare_eagle3_batches('{dir1}', '{dir2}')")
    print()

    return {"dir1": str(dir1), "dir2": str(dir2)}


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compare Eagle3 training dumps")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    compare_parser = subparsers.add_parser("compare", help="Compare two dump files")
    compare_parser.add_argument("path1", help="Path to first dump file or directory")
    compare_parser.add_argument("path2", help="Path to second dump file or directory")
    compare_parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    compare_parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    compare_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    show_parser = subparsers.add_parser("show", help="Show contents of a dump file")
    show_parser.add_argument("path", help="Path to dump file")
    show_parser.add_argument("--gradients", action="store_true", help="Show gradient info")

    args = parser.parse_args()

    if args.command == "compare":
        p1, p2 = Path(args.path1), Path(args.path2)
        if p1.is_dir() and p2.is_dir():
            results = compare_eagle3_batches(
                args.path1, args.path2, rtol=args.rtol, atol=args.atol, verbose=args.verbose
            )
        else:
            results = compare_eagle3_outputs(
                args.path1, args.path2, rtol=args.rtol, atol=args.atol, verbose=args.verbose
            )
        if not results["match"]:
            print("\nDifferences found:")
            for key, diff in results.get("diffs", {}).items():
                print(f"  {key}: {diff}")
            sys.exit(1)
        else:
            print("All tensors match!")
            sys.exit(0)

    elif args.command == "show":
        load_and_print_eagle3_dump(args.path, show_gradients=args.gradients)

    else:
        parser.print_help()
        sys.exit(1)
