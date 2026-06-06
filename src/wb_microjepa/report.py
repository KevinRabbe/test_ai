import argparse
from pathlib import Path
import html
import pandas as pd


def img_tag(path: Path, run_dir: Path) -> str:
    rel = path.relative_to(run_dir).as_posix()
    return f'<div class="plot"><h3>{html.escape(path.name)}</h3><img src="{html.escape(rel)}" /></div>'


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
        summary = evaluation.groupby(["world", "split"])["correct"].mean().reset_index()
        sections.append("<h2>Evaluation summary</h2>")
        sections.append(summary.to_html(index=False))

    if stats_path.exists():
        stats = pd.read_csv(stats_path).sort_values("mean_probe_activation", ascending=False)
        sections.append("<h2>Top active micro-brains</h2>")
        sections.append(stats.head(30).to_html(index=False))

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
