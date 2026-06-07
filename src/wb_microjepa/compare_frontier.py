import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def load_config(run_dir: Path) -> Dict:
    path = run_dir / "config_resolved.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def load_metrics(run_dir: Path) -> Optional[pd.Series]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[-1]


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "samples") -> float:
    if df.empty or value_col not in df.columns:
        return float("nan")
    if weight_col not in df.columns:
        return float(df[value_col].mean())
    weights = df[weight_col].astype(float)
    values = df[value_col].astype(float)
    denom = float(weights.sum())
    if denom <= 0:
        return float(values.mean())
    return float((values * weights).sum() / denom)


def eval_group(df: pd.DataFrame, world: str, split: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    subset = df[(df["world"] == world) & (df["split"] == split)]
    return subset if not subset.empty else None


def summarize_evaluation(run_dir: Path) -> Dict[str, float]:
    eval_path = run_dir / "evaluation.csv"
    df = load_csv(eval_path)
    if df is None:
        return {}

    summary: Dict[str, float] = {}

    if "classifier_correct" in df.columns:
        correct_col = "classifier_correct"
        mae_col = "classifier_absolute_error"
    else:
        correct_col = "correct"
        mae_col = "absolute_error" if "absolute_error" in df.columns else None

    def add(prefix: str, subset: pd.DataFrame) -> None:
        summary[f"{prefix}_accuracy"] = float(subset[correct_col].astype(float).mean())
        if mae_col and mae_col in subset.columns:
            summary[f"{prefix}_mae"] = float(subset[mae_col].astype(float).mean())
        summary[f"{prefix}_samples"] = int(len(subset))

    # Modulo is always a single split.
    modulo = eval_group(df, "modulo_10", "modulo_all")
    if modulo is not None:
        add("modulo_10_modulo_all", modulo)

    # Number-line uses either legacy or range-aware splits.
    train_parts = []
    heldout_parts = []
    for split_name in ["train_range", "range_train_low", "range_train_high"]:
        part = eval_group(df, "number_line", split_name)
        if part is not None:
            train_parts.append(part)
    for split_name in ["heldout_range", "range_heldout_gap", "range_heldout_extrap"]:
        part = eval_group(df, "number_line", split_name)
        if part is not None:
            heldout_parts.append(part)

    if train_parts:
        train_df = pd.concat(train_parts, ignore_index=True)
        add("number_line_train_range", train_df)
    if heldout_parts:
        heldout_df = pd.concat(heldout_parts, ignore_index=True)
        add("number_line_heldout_range", heldout_df)

    return summary


def summarize_case_separation(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "case_region_separation.csv"
    df = load_csv(path)
    if df is None:
        return {}
    return {
        "mean_activation_range": float(df["activation_range"].mean()),
        "max_activation_range": float(df["activation_range"].max()),
        "mean_top_case_gap": float(df["top_case_gap"].mean()),
        "max_top_case_gap": float(df["top_case_gap"].max()),
    }


def summarize_run(run_dir: Path) -> Dict[str, float]:
    cfg = load_config(run_dir)
    model_cfg = cfg.get("model", {})
    row: Dict[str, float] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "world_modulation_strength": float(model_cfg.get("world_modulation_strength", float("nan"))),
        "action_delta_basis_weight": float(model_cfg.get("action_delta_basis_weight", float("nan"))),
    }

    metrics = load_metrics(run_dir)
    if metrics is not None:
        row["latest_epoch"] = int(metrics.get("epoch", 0))
        row["latest_accuracy"] = float(metrics.get("accuracy", float("nan")))
        row["latest_loss"] = float(metrics.get("loss", float("nan")))
        row["latest_case_activation_contrast_loss"] = float(metrics.get("case_activation_contrast_loss", float("nan")))
        row["latest_case_activation_spread_mean"] = float(metrics.get("case_activation_spread_mean", float("nan")))

    row.update(summarize_evaluation(run_dir))
    row.update(summarize_case_separation(run_dir))

    row["modulo_accuracy"] = row.get("modulo_10_modulo_all_accuracy", float("nan"))
    row["number_line_train_accuracy"] = row.get("number_line_train_range_accuracy", float("nan"))
    row["number_line_heldout_accuracy"] = row.get("number_line_heldout_range_accuracy", float("nan"))
    row["heldout_classifier_mae"] = row.get("number_line_heldout_range_mae", float("nan"))
    row["train_classifier_mae"] = row.get("number_line_train_range_mae", float("nan"))
    row["modulo_classifier_mae"] = row.get("modulo_10_modulo_all_mae", float("nan"))

    pass_rule = (
        row["modulo_accuracy"] >= 0.75
        and row["number_line_train_accuracy"] >= 0.29
        and row["number_line_heldout_accuracy"] > 0.0
        and row["heldout_classifier_mae"] <= 46.39
        and row["mean_activation_range"] > 0.03
        and row["mean_top_case_gap"] >= 0.08
    )
    row["passes_rule"] = bool(pass_rule)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", help="Run directories to compare")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        run_dirs = [Path("runs") / name for name in [
            "wb_microjepa_0m_braingraph_targeted_contrast",
            "wb_microjepa_0l_braingraph_specialization_loss",
            "wb_microjepa_0k_braingraph_soft_curriculum",
            "wb_microjepa_0j_braingraph_staged_curriculum",
            "wb_microjepa_0c_braingraph_delta",
            "wb_microjepa_0e_braingraph_worldmod_s025",
            "wb_microjepa_0h_braingraph_action_basis",
            "wb_microjepa_0i_braingraph_range_curriculum",
        ]]

    rows = [summarize_run(run_dir) for run_dir in run_dirs if run_dir.exists()]
    if not rows:
        print("No runs found.")
        return

    df = pd.DataFrame(rows)
    cols = [
        "run_name",
        "world_modulation_strength",
        "action_delta_basis_weight",
        "latest_epoch",
        "latest_accuracy",
        "latest_loss",
        "modulo_accuracy",
        "number_line_train_accuracy",
        "number_line_heldout_accuracy",
        "heldout_classifier_mae",
        "mean_activation_range",
        "max_activation_range",
        "mean_top_case_gap",
        "latest_case_activation_contrast_loss",
        "latest_case_activation_spread_mean",
        "passes_rule",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
