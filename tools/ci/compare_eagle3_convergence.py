#!/usr/bin/env python3
# 读取 TP 与 PP 两个 CI 任务输出的逐步 loss JSON，检查同一步的数值差异是否在阈值内。
# 这是本地 CPU 轨迹比较，不执行训练；通过不代表模型质量绝对正确，只说明两条轨迹在此指标上相近。

"""Compare per-step Eagle3 loss trajectories from TP and PP CI lanes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


# 拒绝空列表、长度/步号不同及非有限 loss，再计算对称相对差 abs(a-b)/max(abs(a),abs(b),1e-8)。
# 任一步超过阈值即报错，成功返回最差项和全部逐步记录供 CI 留档。
def compare_trajectories(
    tp_losses: list[dict[str, Any]],
    pp_losses: list[dict[str, Any]],
    max_relative_diff: float = 0.01,
) -> dict[str, Any]:
    if len(tp_losses) != len(pp_losses) or not tp_losses:
        raise ValueError(
            "TP/PP loss trajectories differ in length or are empty: "
            f"{len(tp_losses)} vs {len(pp_losses)}"
        )

    rows = []
    for tp_item, pp_item in zip(tp_losses, pp_losses):
        if tp_item["step"] != pp_item["step"]:
            raise ValueError(f"TP/PP step mismatch: {tp_item} vs {pp_item}")
        tp_loss = float(tp_item["loss"])
        pp_loss = float(pp_item["loss"])
        if not (math.isfinite(tp_loss) and math.isfinite(pp_loss)):
            raise ValueError(f"TP/PP losses must be finite: {tp_item} vs {pp_item}")
        # 例如 TP=2、PP=2.01，相对差为 0.01/2.01；对称分母避免只把某一条轨迹当绝对基准。
        denominator = max(abs(tp_loss), abs(pp_loss), 1.0e-8)
        rows.append(
            {
                "step": tp_item["step"],
                "tp_loss": tp_loss,
                "pp_loss": pp_loss,
                "relative_diff": abs(tp_loss - pp_loss) / denominator,
            }
        )

    # 不是只比较最终 loss 或平均差异：任一训练步偏离过大都会使 CI 失败。
    worst = max(rows, key=lambda row: row["relative_diff"])
    if worst["relative_diff"] > max_relative_diff:
        raise ValueError(
            f"PP Eagle3 convergence diverged from TP: worst={worst}, limit={max_relative_diff:.4f}"
        )

    return {
        "max_relative_diff": max_relative_diff,
        "worst": worst,
        "steps": rows,
    }


# 读取两个 JSON，运行比较后覆盖写报告，并输出机器可解析的最差差异和完整步列表。
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tp_losses", type=Path)
    parser.add_argument("pp_losses", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-relative-diff", type=float, default=0.01)
    args = parser.parse_args()

    tp_losses = json.loads(args.tp_losses.read_text(encoding="utf-8"))
    pp_losses = json.loads(args.pp_losses.read_text(encoding="utf-8"))
    result = compare_trajectories(tp_losses, pp_losses, args.max_relative_diff)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "CI_PP_CONVERGENCE "
        f"max_relative_diff={result['worst']['relative_diff']:.6f} "
        f"limit={args.max_relative_diff:.6f}"
    )
    print(f"CI_PP_CONVERGENCE_STEPS values={json.dumps(result['steps'], separators=(',', ':'))}")


if __name__ == "__main__":
    main()
