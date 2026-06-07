import argparse
from pathlib import Path
import html
import pandas as pd


def img_tag(path: Path, run_dir: Path) -> str:
    rel = path.relative_to(run_dir).as_posix()
    return f'<div class="plot"><h3>{html.escape(path.name)}</h3><img src="{html.escape(rel)}" /></div>'


def evaluation_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    # New evaluation format: classifier and nearest-embedding decoding.
    if "classifier_correct" in evaluation.columns and "nearest_correct" in evaluation.columns:
        return evaluation.groupby(["world", "split"]).agg(
            classifier_accuracy=("classifier_correct", "mean"),
            nearest_accuracy=("nearest_correct", "mean"),
            classifier_mae=("classifier_absolute_error", "mean"),
            nearest_mae=("nearest_absolute_error", "mean"),
            samples=("classifier_correct", "count"),
        ).reset_index()

    # Legacy evaluation format.
    if "correct" in evaluation.columns:
        return evaluation.groupby(["world", "split"]).agg(
            accuracy=("correct", "mean"),
            mean_absolute_error=("absolute_error", "mean") if "absolute_error" in evaluation.columns else ("correct", "count"),
            samples=("correct", "count"),
        ).reset_index()

    raise ValueError("Unknown evaluation.csv schema. Expected classifier_correct/nearest_correct or correct column.")


def case_type_summary(evaluation: pd.DataFrame) -> pd.DataFrame | None:
    if "case_type" not in evaluation.columns:
        return None

    if "classifier_correct" in evaluation.columns and "nearest_correct" in evaluation.columns:
        return evaluation.groupby(["world", "split", "case_type"]).agg(
            classifier_accuracy=("classifier_correct", "mean"),
            nearest_accuracy=("nearest_correct", "mean"),
            classifier_mae=("classifier_absolute_error", "mean"),
            nearest_mae=("nearest_absolute_error", "mean"),
            samples=("classifier_correct", "count"),
        ).reset_index()

    if "correct" in evaluation.columns:
        return evaluation.groupby(["world", "split", "case_type"]).agg(
            accuracy=("correct", "mean"),
            mean_absolute_error=("absolute_error", "mean") if "absolute_error" in evaluation.columns else ("correct", "count"),
            samples=("correct", "count"),
        ).reset_index()

    return None


def action_delta_summary(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "action_delta_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    stats_path = run_dir / "microbrain_stats.csv"
    eval_path = run_dir / "evaluation.csv"

    sections = []

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        sections.append("<h2>Latest metrics</h2>")
        sections.append(metrics.tail(15).to_html(index=False))

    if eval_path.exists():
        evaluation = pd.read_csv(eval_path)
        summary = evaluation_summary(evaluation)
        sections.append("<h2>Evaluation summary</h2>")
        sections.append(summary.to_html(index=False))
        csum = case_type_summary(evaluation)
        if csum is not None:
            sections.append("<h2>Case type summary</h2>")
            sections.append(csum.to_html(index=False))

    region_case_path = run_dir / "region_case_activation_summary.csv"
    if region_case_path.exists():
        region_case = pd.read_csv(region_case_path)
        sections.append("<h2>Region by case type</h2>")
        sections.append(region_case.to_html(index=False))

    if stats_path.exists():
        stats = pd.read_csv(stats_path).sort_values("mean_probe_activation", ascending=False)
        sections.append("<h2>Top active micro-brains</h2>")
        sections.append(stats.head(30).to_html(index=False))

    action_delta_path = run_dir / "action_delta_summary.csv"
    if action_delta_path.exists():
        action_delta = pd.read_csv(action_delta_path)
        sections.append("<h2>Action delta summary</h2>")
        sections.append(action_delta.to_html(index=False))

    region_path = run_dir / "region_activation_summary.csv"
    if region_path.exists():
        regions = pd.read_csv(region_path)
        sections.append("<h2>Latest region activation summary</h2>")
        latest_epoch = regions["epoch"].max()
        sections.append(regions[regions["epoch"] == latest_epoch].to_html(index=False))

    plot_paths = sorted((run_dir / "plots").glob("*.png")) if (run_dir / "plots").exists() else []
    if plot_paths:
        sections.append("<h2>Plots</h2>")
        for p in plot_paths[-20:]:
            sections.append(img_tag(p, run_dir))

    body = "\n".join(sections)
    document = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>WB-MicroJEPA Run Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
    img {{ max-width: 1100px; width: 100%; border: 1px solid #ddd; }}
    .plot {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>WB-MicroJEPA Run Report</h1>
  {body}
</body>
</html>
"""
    out = run_dir / "report.html"
    out.write_text(document, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
