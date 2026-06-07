import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
import yaml
import torch
import torch.nn.functional as F
import pandas as pd

from .worlds import ACTION_TO_ID, WORLD_ID
from .train import choose_device, build_model


@torch.no_grad()
def evaluate_transition_grid(model, cfg: Dict, device: torch.device) -> pd.DataFrame:
    rows: List[Dict] = []
    actions = cfg["worlds"]["actions"]
    max_number = int(cfg["worlds"]["max_number"])
    train_max = int(cfg["worlds"]["train_max_number"])
    modulo_n = int(cfg["worlds"]["modulo_n"])

    model.eval()
    state_bank = build_state_bank(model, cfg, device)

    for state in range(0, max_number + 1):
        for action in actions:
            target = state + action
            if 0 <= target <= max_number:
                classifier_pred, nearest_pred = predict_both(model, device, state_bank, WORLD_ID["number_line"], state, ACTION_TO_ID[action])
                rows.append({
                    "world": "number_line",
                    "state": state,
                    "action": action,
                    "target": target,
                    "classifier_prediction": classifier_pred,
                    "nearest_prediction": nearest_pred,
                    "classifier_correct": classifier_pred == target,
                    "nearest_correct": nearest_pred == target,
                    "classifier_absolute_error": abs(classifier_pred - target),
                    "nearest_absolute_error": abs(nearest_pred - target),
                    "split": "train_range" if state <= train_max else "heldout_range",
                })

    for state in range(0, modulo_n):
        for action in actions:
            target = (state + action) % modulo_n
            classifier_pred, nearest_pred = predict_both(model, device, state_bank, WORLD_ID["modulo_10"], state, ACTION_TO_ID[action])
            rows.append({
                "world": "modulo_10",
                "state": state,
                "action": action,
                "target": target,
                "classifier_prediction": classifier_pred,
                "nearest_prediction": nearest_pred,
                "classifier_correct": classifier_pred == target,
                "nearest_correct": nearest_pred == target,
                "classifier_absolute_error": abs(classifier_pred - target),
                "nearest_absolute_error": abs(nearest_pred - target),
                "split": "modulo_all",
            })

    return pd.DataFrame(rows)


@torch.no_grad()
def build_state_bank(model, cfg: Dict, device: torch.device) -> torch.Tensor:
    states = torch.arange(int(cfg["model"]["num_numbers"]), dtype=torch.long, device=device)
    bank = model.encode_state(states)
    return F.normalize(bank, dim=-1)


@torch.no_grad()
def predict_both(model, device, state_bank: torch.Tensor, world_id: int, state: int, action_id: int) -> Tuple[int, int]:
    s = torch.tensor([state], dtype=torch.long, device=device)
    a = torch.tensor([action_id], dtype=torch.long, device=device)
    w = torch.tensor([world_id], dtype=torch.long, device=device)
    logits, trace = model(s, a, w)
    classifier_pred = int(logits.argmax(dim=-1).item())

    pred_emb = F.normalize(trace["predicted_target_embedding"], dim=-1)
    similarity = pred_emb @ state_bank.T
    nearest_pred = int(similarity.argmax(dim=-1).item())
    return classifier_pred, nearest_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = Path(args.config) if args.config else run_dir / "config_resolved.json"
    if cfg_path.suffix == ".json":
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    device = choose_device(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "final.pt", map_location=device))

    df = evaluate_transition_grid(model, cfg, device)
    out = run_dir / "evaluation.csv"
    df.to_csv(out, index=False)

    summary = df.groupby(["world", "split"]).agg(
        classifier_accuracy=("classifier_correct", "mean"),
        nearest_accuracy=("nearest_correct", "mean"),
        classifier_mae=("classifier_absolute_error", "mean"),
        nearest_mae=("nearest_absolute_error", "mean"),
        samples=("classifier_correct", "count"),
    ).reset_index()
    print(summary.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
