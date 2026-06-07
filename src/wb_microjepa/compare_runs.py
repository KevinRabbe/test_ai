import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .report import evaluation_summary


def load_metrics(run_dir: Path) -> Optional[pd.Series]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None
    row = metrics.iloc[-1].copy()
    row["run"] = run_dir.name
    row["run_dir"] = str(run_dir)
    return row


def load_evaluation(run_dir: Path) -> Optional[pd.DataFrame]:
    eval_path = run_dir / "evaluation.csv"
    if not eval_path.exists():
        return None
    evaluation = pd.read_csv(eval_path)
    if evaluation.empty:
        return None
    return evaluation


def load_microbrain_stats(run_dir: Path) -> Optional[pd.DataFrame]:
    stats_path = run_dir / "microbrain_stats.csv"
    if not stats_path.exists():
        return None
    stats = pd.read_csv(stats_path)
    if stats.empty:
        return None
    return stats


def load_action_delta_summary(run_dir: Path) -> Optional[pd.DataFrame]:
    path = run_dir / "action_delta_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def summarize_run(run_dir: Path) -> Dict:
    summary: Dict = {"run": run_dir.name, "run_dir": str(run_dir)}

    metrics = load_metrics(run_dir)
    if metrics is not None:
        summary["latest_epoch"] = int(metrics.get("epoch", 0))
        summary["latest_accuracy"] = float(metrics.get("accuracy", float("nan")))
        summary["latest_loss"] = float(metrics.get("loss", float("nan")))
        summary["latest_prediction_loss"] = float(metrics.get("prediction_loss", float("nan")))
        summary["latest_identity_loss"] = float(metrics.get("identity_loss", float("nan")))
        summary["latest_delta_consistency_loss"] = float(metrics.get("delta_consistency_loss", float("nan")))

    evaluation = load_evaluation(run_dir)
    if evaluation is not None:
        eval_summary = evaluation_summary(evaluation)
        for _, row in eval_summary.iterrows():
            world = str(row["world"])
            split = str(row["split"])
            prefix = f"eval_{world}_{split}"
            if "accuracy" in row:
                summary[f"{prefix}_accuracy"] = float(row["accuracy"])
            if "mean_absolute_error" in row:
                summary[f"{prefix}_mae"] = float(row["mean_absolute_error"])
            if "classifier_accuracy" in row:
                summary[f"{prefix}_classifier_accuracy"] = float(row["classifier_accuracy"])
            if "nearest_accuracy" in row:
                summary[f"{prefix}_nearest_accuracy"] = float(row["nearest_accuracy"])
            if "classifier_mae" in row:
                summary[f"{prefix}_classifier_mae"] = float(row["classifier_mae"])
            if "nearest_mae" in row:
                summary[f"{prefix}_nearest_mae"] = float(row["nearest_mae"])
            summary[f"{prefix}_samples"] = int(row["samples"])

    stats = load_microbrain_stats(run_dir)
    if stats is not None and "mean_probe_activation" in stats.columns:
        activations = stats["mean_probe_activation"].astype(float)
        summary["mean_probe_activation"] = float(activations.mean())
        summary["std_probe_activation"] = float(activations.std(ddof=0))
        summary["active_units_gt_0.5"] = int((activations > 0.5).sum())
        summary["active_units_gt_0.3"] = int((activations > 0.3).sum())
        summary["dead_units_lt_0.05"] = int((activations < 0.05).sum())
        summary["top10_activation_share"] = float(activations.nlargest(max(1, len(activations) // 10)).sum() / max(activations.sum(), 1e-8))

    action_delta = load_action_delta_summary(run_dir)
    if action_delta is not None:
        summary["mean_action_delta_norm"] = float(action_delta["action_delta_norm"].mean())
        summary["mean_braingraph_delta_norm"] = float(action_delta["braingraph_delta_norm"].mean())
        summary["mean_combined_delta_norm"] = float(action_delta["combined_delta_norm"].mean())
        summary["mean_delta_cosine"] = float(action_delta["delta_cosine"].mean())

    return summary


def discover_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    runs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "metrics.csv").exists():
            runs.append(child)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", help="Run directories to compare")
    parser.add_argument("--root", type=str, default="runs", help="Root directory to scan when no run dirs are given")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        run_dirs = discover_runs(Path(args.root))

    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    if not rows:
        print("No runs found.")
        return

    df = pd.DataFrame(rows)
    cols = [c for c in [
        "run",
        "latest_epoch",
        "latest_accuracy",
        "latest_loss",
        "eval_number_line_train_range_accuracy",
        "eval_number_line_heldout_range_accuracy",
        "eval_modulo_10_modulo_all_accuracy",
        "latest_delta_consistency_loss",
        "eval_number_line_train_range_mae",
        "eval_number_line_heldout_range_mae",
        "eval_modulo_10_modulo_all_mae",
        "std_probe_activation",
        "dead_units_lt_0.05",
        "top10_activation_share",
        "mean_action_delta_norm",
        "mean_braingraph_delta_norm",
        "mean_combined_delta_norm",
        "mean_delta_cosine",
    ] if c in df.columns]

    print(df[cols].sort_values(by=[c for c in ["eval_modulo_10_modulo_all_accuracy", "eval_number_line_train_range_accuracy", "latest_accuracy"] if c in df.columns], ascending=False).to_string(index=False))

    if args.output:
        out = Path(args.output)
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
