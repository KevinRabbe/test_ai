from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px

from .report import evaluation_summary, case_type_summary


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def region_separation_table(region_case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (world, split, region), group in region_case.groupby(["world", "split", "region"]):
        ordered = group.sort_values("mean_activation", ascending=False).reset_index(drop=True)
        dominant = ordered.iloc[0]
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None
        bottom = ordered.iloc[-1]
        rows.append({
            "world": world,
            "split": split,
            "region": region,
            "case_type_count": int(group["case_type"].nunique()),
            "mean_activation": float(group["mean_activation"].mean()),
            "activation_std": float(group["mean_activation"].std(ddof=0) if len(group) > 1 else 0.0),
            "activation_range": float(group["mean_activation"].max() - group["mean_activation"].min()),
            "dominant_case_type": str(dominant["case_type"]),
            "dominant_case_activation": float(dominant["mean_activation"]),
            "runner_up_case_type": str(runner_up["case_type"]) if runner_up is not None else "",
            "runner_up_case_activation": float(runner_up["mean_activation"]) if runner_up is not None else float("nan"),
            "top_case_gap": float(dominant["mean_activation"] - runner_up["mean_activation"]) if runner_up is not None else float("nan"),
            "bottom_case_type": str(bottom["case_type"]),
            "bottom_case_activation": float(bottom["mean_activation"]),
            "dominance_ratio": float(dominant["mean_activation"] / max(float(group["mean_activation"].mean()), 1e-8)),
            "active_units_gt_0_5": int(group["active_units_gt_0_5"].sum()),
            "samples": int(group["samples"].sum()),
        })
    return pd.DataFrame(rows)


def case_dominance_table(region_case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (world, split, case_type), group in region_case.groupby(["world", "split", "case_type"]):
        ordered = group.sort_values("mean_activation", ascending=False).reset_index(drop=True)
        dominant = ordered.iloc[0]
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None
        rows.append({
            "world": world,
            "split": split,
            "case_type": case_type,
            "region_count": int(group["region"].nunique()),
            "dominant_region": str(dominant["region"]),
            "dominant_region_activation": float(dominant["mean_activation"]),
            "runner_up_region": str(runner_up["region"]) if runner_up is not None else "",
            "runner_up_region_activation": float(runner_up["mean_activation"]) if runner_up is not None else float("nan"),
            "region_gap": float(dominant["mean_activation"] - runner_up["mean_activation"]) if runner_up is not None else float("nan"),
            "mean_region_activation": float(group["mean_activation"].mean()),
            "activation_std": float(group["mean_activation"].std(ddof=0) if len(group) > 1 else 0.0),
            "samples": int(group["samples"].sum()),
        })
    return pd.DataFrame(rows)


def make_heatmap(region_case: pd.DataFrame, title: str):
    pivot = region_case.pivot_table(index="region", columns="case_type", values="mean_activation", aggfunc="mean").fillna(0.0)
    fig = px.imshow(
        pivot,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Viridis",
        title=title,
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    return df.head(max_rows).to_html(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    region_case_path = run_dir / "region_case_activation_summary.csv"
    evaluation_path = run_dir / "evaluation.csv"
    stats_path = run_dir / "microbrain_stats.csv"

    region_case = load_csv(region_case_path)
    if region_case is None:
        raise FileNotFoundError(f"Missing region_case_activation_summary.csv in {run_dir}")

    separation = region_separation_table(region_case).sort_values(
        ["activation_range", "top_case_gap", "dominance_ratio"],
        ascending=False,
    )
    case_dominance = case_dominance_table(region_case).sort_values(
        ["region_gap", "dominant_region_activation"],
        ascending=False,
    )

    separation_path = output_dir / "case_region_separation.csv"
    separation.to_csv(separation_path, index=False)

    evaluation = load_csv(evaluation_path)
    stats = load_csv(stats_path)

    sections = [f"<h1>Case Separation Report - {run_dir.name}</h1>"]
    sections.append("<h2>Region separation ranking</h2>")
    sections.append(html_table(separation))

    sections.append("<h2>Dominant region by case type</h2>")
    sections.append(html_table(case_dominance))

    if evaluation is not None:
        sections.append("<h2>Evaluation summary</h2>")
        sections.append(html_table(evaluation_summary(evaluation)))
        csum = case_type_summary(evaluation)
        if csum is not None:
            sections.append("<h2>Case type summary</h2>")
            sections.append(html_table(csum))

    if stats is not None:
        sections.append("<h2>Top active micro-brains</h2>")
        sections.append(html_table(stats.sort_values("mean_probe_activation", ascending=False).head(30)))

    for world in sorted(region_case["world"].unique()):
        world_df = region_case[region_case["world"] == world]
        for split in sorted(world_df["split"].unique()):
            subset = world_df[world_df["split"] == split]
            sections.append(f"<h2>Heatmap: {world} / {split}</h2>")
            sections.append(make_heatmap(subset, f"{world} / {split} mean activation by case type and region").to_html(full_html=False, include_plotlyjs="cdn"))

    document = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Case Separation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }}
    .plot {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  {'\n'.join(sections)}
</body>
</html>
"""

    out = output_dir / "case_region_report.html"
    out.write_text(document, encoding="utf-8")
    print(f"Saved: {separation_path}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
