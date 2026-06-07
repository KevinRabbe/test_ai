import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .report import evaluation_summary


def load_config(run_dir: Path) -> Optional[Dict]:
    cfg_path = run_dir / "config_resolved.json"
    if not cfg_path.exists():
        return None
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_evaluation_summary(run_dir: Path) -> Optional[pd.DataFrame]:
    eval_path = run_dir / "evaluation.csv"
    if not eval_path.exists():
        return None
    evaluation = pd.read_csv(eval_path)
    if evaluation.empty:
        return None
    return evaluation_summary(evaluation)


def load_case_region_summary(run_dir: Path) -> Optional[pd.DataFrame]:
    path = run_dir / "case_region_separation.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def parse_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def first_metric(summary: pd.DataFrame, world: str, split: str, columns: list[str]) -> float:
    if summary is None or summary.empty:
        return float("nan")
    row = summary[(summary["world"] == world) & (summary["split"] == split)]
    if row.empty:
        return float("nan")
    for col in columns:
        if col in row.columns:
            return float(row.iloc[0][col])
    return float("nan")


def summarize_run(run_dir: Path) -> Dict:
    cfg = load_config(run_dir) or {}
    model_cfg = cfg.get("model", {})
    evaluation = load_evaluation_summary(run_dir)
    case_region = load_case_region_summary(run_dir)

    modulo_accuracy = first_metric(evaluation, "modulo_10", "modulo_all", ["classifier_accuracy", "accuracy"])
    train_accuracy = first_metric(evaluation, "number_line", "train_range", ["classifier_accuracy", "accuracy"])
    heldout_accuracy = first_metric(evaluation, "number_line", "heldout_range", ["classifier_accuracy", "accuracy"])
    heldout_mae = first_metric(evaluation, "number_line", "heldout_range", ["classifier_mae", "mean_absolute_error"])

    mean_activation_range = float(case_region["activation_range"].mean()) if case_region is not None else float("nan")
    max_activation_range = float(case_region["activation_range"].max()) if case_region is not None else float("nan")
    mean_top_case_gap = float(case_region["top_case_gap"].mean()) if case_region is not None else float("nan")

    passes_rule = (
        modulo_accuracy >= 0.70
        and train_accuracy >= 0.18
        and mean_activation_range > 0.05
        and mean_top_case_gap > 0.10
        and heldout_mae <= 60.74
    )

    return {
        "run_name": run_dir.name,
        "world_modulation_strength": parse_float(model_cfg.get("world_modulation_strength", float("nan"))),
        "modulo_accuracy": modulo_accuracy,
        "number_line_train_accuracy": train_accuracy,
        "number_line_heldout_accuracy": heldout_accuracy,
        "heldout_classifier_mae": heldout_mae,
        "mean_activation_range": mean_activation_range,
        "max_activation_range": max_activation_range,
        "mean_top_case_gap": mean_top_case_gap,
        "passes_rule": passes_rule,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")
    parser.add_argument("--output", type=str, default="runs/wb_microjepa_0e_worldmod_sweep_summary.csv")
    args = parser.parse_args()

    rows = [summarize_run(Path(p)) for p in args.run_dirs]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs found.")
        return

    df = df.sort_values(["world_modulation_strength", "run_name"], ascending=[True, True])
    cols = [
        "run_name",
        "world_modulation_strength",
        "modulo_accuracy",
        "number_line_train_accuracy",
        "number_line_heldout_accuracy",
        "heldout_classifier_mae",
        "mean_activation_range",
        "max_activation_range",
        "mean_top_case_gap",
        "passes_rule",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
