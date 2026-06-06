import argparse
from pathlib import Path
from typing import Dict, List
import json
import yaml
import torch
import pandas as pd

from .models import WBMICROJEPA
from .worlds import ACTION_TO_ID, WORLD_ID
from .train import choose_device


@torch.no_grad()
def evaluate_transition_grid(model, cfg: Dict, device: torch.device) -> pd.DataFrame:
    rows: List[Dict] = []
    actions = cfg["worlds"]["actions"]
    max_number = int(cfg["worlds"]["max_number"])
    train_max = int(cfg["worlds"]["train_max_number"])
    modulo_n = int(cfg["worlds"]["modulo_n"])

    model.eval()

    # Number line: evaluate full configured range.
    for state in range(0, max_number + 1):
        for action in actions:
            target = state + action
            if 0 <= target <= max_number:
                pred = predict_one(model, device, WORLD_ID["number_line"], state, ACTION_TO_ID[action])
                rows.append({
                    "world": "number_line",
                    "state": state,
                    "action": action,
                    "target": target,
                    "prediction": pred,
                    "correct": pred == target,
                    "split": "train_range" if state <= train_max else "heldout_range",
                })

    # Modulo world.
    for state in range(0, modulo_n):
        for action in actions:
            target = (state + action) % modulo_n
            pred = predict_one(model, device, WORLD_ID["modulo_10"], state, ACTION_TO_ID[action])
            rows.append({
                "world": "modulo_10",
                "state": state,
                "action": action,
                "target": target,
                "prediction": pred,
                "correct": pred == target,
                "split": "modulo_all",
            })

    return pd.DataFrame(rows)


@torch.no_grad()
def predict_one(model, device, world_id: int, state: int, action_id: int) -> int:
    s = torch.tensor([state], dtype=torch.long, device=device)
    a = torch.tensor([action_id], dtype=torch.long, device=device)
    w = torch.tensor([world_id], dtype=torch.long, device=device)
    logits, _ = model(s, a, w)
    return int(logits.argmax(dim=-1).item())


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
    model = WBMICROJEPA(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "final.pt", map_location=device))

    df = evaluate_transition_grid(model, cfg, device)
    out = run_dir / "evaluation.csv"
    df.to_csv(out, index=False)

    summary = df.groupby(["world", "split"])["correct"].mean().reset_index()
    print(summary.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
