from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .brain_topology import build_brain_graph, ID_TO_REGION
from .report import evaluation_summary


RUNS_ROOT = Path("runs")


def list_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    runs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "metrics.csv").exists():
            runs.append(child)
    return runs


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def load_jsonl(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows if rows else None


def load_config(run_dir: Path) -> Optional[Dict]:
    cfg_path = run_dir / "config_resolved.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return None


def latest_row(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1]


def parse_microbrain_index(label: str) -> int:
    match = re.search(r"(\d+)$", str(label))
    return int(match.group(1)) if match else -1


def build_brain_figure(run_dir: Path, cfg: Dict, stats: pd.DataFrame) -> go.Figure:
    graph = build_brain_graph(cfg)
    stats = stats.copy()
    if "microbrain_id" in stats.columns:
        stats["unit_index"] = stats["microbrain_id"].map(parse_microbrain_index)
        stats = stats.sort_values("unit_index")
    coords = graph.coords.cpu().numpy()
    regions = [ID_TO_REGION[int(r)] for r in graph.region_ids.cpu().numpy().tolist()]
    activations = stats["mean_probe_activation"].astype(float).tolist() if "mean_probe_activation" in stats.columns else [0.0] * len(coords)
    frame = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "activation": activations,
            "region": regions,
        }
    )

    fig = px.scatter_3d(
        frame,
        x="x",
        y="y",
        z="z",
        color="activation",
        size="activation",
        size_max=16,
        hover_name="region",
        hover_data={"x": ":.3f", "y": ":.3f", "z": ":.3f", "activation": ":.3f"},
        color_continuous_scale="Viridis",
        title=f"BrainGraph 3D - {run_dir.name}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=700)
    return fig


def render_overview(run_dir: Path, metrics: Optional[pd.DataFrame], evaluation: Optional[pd.DataFrame], stats: Optional[pd.DataFrame], region: Optional[pd.DataFrame]) -> None:
    cols = st.columns(4)
    if metrics is not None:
        row = latest_row(metrics)
        cols[0].metric("Epoch", int(row.get("epoch", 0)))
        cols[1].metric("Accuracy", f"{float(row.get('accuracy', 0.0)):.3f}")
        cols[2].metric("Loss", f"{float(row.get('loss', 0.0)):.3f}")
        cols[3].metric("Pred loss", f"{float(row.get('prediction_loss', 0.0)):.3f}")
    else:
        cols[0].metric("Epoch", "n/a")
        cols[1].metric("Accuracy", "n/a")
        cols[2].metric("Loss", "n/a")
        cols[3].metric("Pred loss", "n/a")

    if metrics is not None:
        st.subheader("Training metrics")
        st.line_chart(metrics.set_index("epoch")[["accuracy", "loss", "prediction_loss", "identity_loss"]])

    if evaluation is not None:
        st.subheader("Evaluation summary")
        st.dataframe(evaluation_summary(evaluation), use_container_width=True)

    if region is not None:
        st.subheader("Latest region activation summary")
        st.dataframe(region, use_container_width=True)

    if stats is not None:
        st.subheader("Top active micro-brains")
        st.dataframe(stats.sort_values("mean_probe_activation", ascending=False).head(30), use_container_width=True)


def render_evaluation(evaluation: Optional[pd.DataFrame]) -> None:
    if evaluation is None:
        st.info("No evaluation.csv found for this run.")
        return

    summary = evaluation_summary(evaluation)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Evaluation detail")
    if "case_type" in evaluation.columns:
        if "classifier_correct" in evaluation.columns and "nearest_correct" in evaluation.columns:
            detail = evaluation.groupby(["world", "split", "case_type"]).agg(
                classifier_accuracy=("classifier_correct", "mean"),
                nearest_accuracy=("nearest_correct", "mean"),
                classifier_mae=("classifier_absolute_error", "mean"),
                nearest_mae=("nearest_absolute_error", "mean"),
                samples=("classifier_correct", "count"),
            ).reset_index()
        else:
            detail = evaluation.groupby(["world", "split", "case_type"]).agg(
                accuracy=("correct", "mean") if "correct" in evaluation.columns else ("world", "count"),
                mean_absolute_error=("absolute_error", "mean") if "absolute_error" in evaluation.columns else ("world", "count"),
                samples=("world", "count"),
            ).reset_index()
        st.dataframe(detail, use_container_width=True)
    else:
        st.dataframe(evaluation.head(200), use_container_width=True)


def render_region_case(region_case: Optional[pd.DataFrame]) -> None:
    if region_case is None:
        st.info("No region_case_activation_summary.csv found for this run.")
        return

    st.dataframe(region_case, use_container_width=True)
    if {"case_type", "region", "mean_activation"}.issubset(region_case.columns):
        pivot = region_case.pivot_table(index="region", columns="case_type", values="mean_activation", aggfunc="mean").fillna(0.0)
        st.subheader("Mean activation by region and case type")
        st.dataframe(pivot, use_container_width=True)


def render_brain(cfg: Optional[Dict], stats: Optional[pd.DataFrame], run_dir: Path) -> None:
    if cfg is None or stats is None or "topology" not in cfg:
        st.info("BrainGraph viewer needs config_resolved.json and microbrain_stats.csv.")
        return

    graph = build_brain_graph(cfg)
    stats = stats.copy()
    stats["unit_index"] = stats["microbrain_id"].map(parse_microbrain_index) if "microbrain_id" in stats.columns else range(len(stats))
    stats = stats.sort_values("unit_index").reset_index(drop=True)
    activations = stats["mean_probe_activation"].astype(float).to_list()

    st.plotly_chart(build_brain_figure(run_dir, cfg, stats), use_container_width=True)

    st.caption("Region order and 3D coordinates come from the saved topology. Color and size reflect mean probe activation.")
    region_df = pd.DataFrame({
        "region": [ID_TO_REGION[int(r)] for r in graph.region_ids.cpu().numpy().tolist()],
        "activation": activations,
    })
    st.dataframe(region_df.groupby("region").agg(mean_activation=("activation", "mean"), max_activation=("activation", "max"), units=("activation", "count")).reset_index(), use_container_width=True)


def render_probes(traces: Optional[List[Dict]]) -> None:
    if not traces:
        st.info("No probe_traces.jsonl found for this run.")
        return

    df = pd.DataFrame(traces)
    st.dataframe(df.head(200), use_container_width=True)

    if "world" in df.columns:
        selected_world = st.selectbox("Probe world", sorted(df["world"].dropna().unique().tolist()))
        filtered = df[df["world"] == selected_world]
    else:
        filtered = df

    if not filtered.empty:
        idx = st.slider("Probe row", 0, len(filtered) - 1, 0)
        row = filtered.iloc[idx]
        st.json(row.to_dict())


def render_compare(root: Path) -> None:
    run_dirs = list_runs(root)
    if not run_dirs:
        st.info("No completed runs found.")
        return

    from .compare_runs import summarize_run

    df = pd.DataFrame([summarize_run(run_dir) for run_dir in run_dirs])
    preferred_cols = [
        "run",
        "latest_epoch",
        "latest_accuracy",
        "latest_loss",
        "eval_number_line_train_range_accuracy",
        "eval_number_line_heldout_range_accuracy",
        "eval_modulo_10_modulo_all_accuracy",
        "std_probe_activation",
        "dead_units_lt_0.05",
        "top10_activation_share",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    st.dataframe(df[cols].sort_values(by=[c for c in ["eval_modulo_10_modulo_all_accuracy", "eval_number_line_train_range_accuracy", "latest_accuracy"] if c in df.columns], ascending=False), use_container_width=True)

    numeric_cols = [c for c in ["latest_accuracy", "latest_loss", "eval_number_line_train_range_accuracy", "eval_number_line_heldout_range_accuracy", "eval_modulo_10_modulo_all_accuracy"] if c in df.columns]
    if numeric_cols:
        chart_df = df[["run"] + numeric_cols].set_index("run")
        st.bar_chart(chart_df)


def main() -> None:
    st.set_page_config(page_title="WB-MicroJEPA Monitor", layout="wide")
    st.title("WB-MicroJEPA Monitor")

    root_text = st.sidebar.text_input("Runs root", value=str(RUNS_ROOT))
    root = Path(root_text)
    run_dirs = list_runs(root)

    if not run_dirs:
        st.warning("No runs found yet. Train a run first, then refresh the dashboard.")
        return

    run_map = {run.name: run for run in run_dirs}
    selected_run_name = st.sidebar.selectbox("Run", list(run_map.keys()))
    run_dir = run_map[selected_run_name]

    cfg = load_config(run_dir)
    metrics = load_csv(run_dir / "metrics.csv")
    evaluation = load_csv(run_dir / "evaluation.csv")
    stats = load_csv(run_dir / "microbrain_stats.csv")
    region = load_csv(run_dir / "region_activation_summary.csv")
    region_case = load_csv(run_dir / "region_case_activation_summary.csv")
    traces = load_jsonl(run_dir / "probe_traces.jsonl")

    tabs = st.tabs(["Overview", "Evaluation", "Region Cases", "BrainGraph", "Probes", "Compare Runs", "Plots"])
    with tabs[0]:
        render_overview(run_dir, metrics, evaluation, stats, region)
    with tabs[1]:
        render_evaluation(evaluation)
    with tabs[2]:
        render_region_case(region_case)
    with tabs[3]:
        render_brain(cfg, stats, run_dir)
    with tabs[4]:
        render_probes(traces)
    with tabs[5]:
        render_compare(root)
    with tabs[6]:
        plot_dir = run_dir / "plots"
        if plot_dir.exists():
            for path in sorted(plot_dir.glob("*.png")):
                st.image(str(path), caption=path.name, use_container_width=True)
        else:
            st.info("No plots found.")


if __name__ == "__main__":
    main()
