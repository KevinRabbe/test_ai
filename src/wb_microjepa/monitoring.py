from pathlib import Path
from typing import Dict, List
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


class DiagnosticsLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        self.trace_path = self.run_dir / "probe_traces.jsonl"
        self.metrics_path = self.run_dir / "metrics.csv"

    def log_metrics(self, row: Dict) -> None:
        df = pd.DataFrame([row])
        header = not self.metrics_path.exists()
        df.to_csv(self.metrics_path, mode="a", header=header, index=False)

    def log_probe_traces(self, epoch: int, traces: List[Dict]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as f:
            for tr in traces:
                tr["epoch"] = epoch
                f.write(json.dumps(tr) + "\n")

    def save_activation_heatmap(self, epoch: int, activation_matrix: np.ndarray, labels: List[str]) -> None:
        plt.figure(figsize=(12, max(4, len(labels) * 0.35)))
        plt.imshow(activation_matrix, aspect="auto")
        plt.colorbar(label="activation")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("micro-brain id")
        plt.ylabel("probe")
        plt.title(f"Micro-brain activation heatmap - epoch {epoch}")
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / f"activation_heatmap_epoch_{epoch:04d}.png", dpi=160)
        plt.close()

    def save_embedding_plot(self, epoch: int, embeddings: np.ndarray, title: str = "State embeddings") -> None:
        # Simple deterministic 2D projection using first two principal directions via SVD.
        x = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        coords = x @ vt[:2].T

        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], s=14)
        for idx in range(len(coords)):
            if idx % 5 == 0:
                plt.text(coords[idx, 0], coords[idx, 1], str(idx), fontsize=7)
        plt.title(f"{title} - epoch {epoch}")
        plt.tight_layout()
        plt.savefig(self.run_dir / "plots" / f"embeddings_epoch_{epoch:04d}.png", dpi=160)
        plt.close()

    def save_microbrain_stats(self, stats: pd.DataFrame) -> None:
        stats.to_csv(self.run_dir / "microbrain_stats.csv", index=False)


def tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x
