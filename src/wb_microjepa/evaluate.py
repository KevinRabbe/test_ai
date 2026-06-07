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
from .brain_topology import ID_TO_REGION


def load_compatible_state_dict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()

    if "state_embedding.weight" in state_dict and "state_encoder.id_embedding.weight" in model_state:
        state_dict["state_encoder.id_embedding.weight"] = state_dict.pop("state_embedding.weight")

    if "state_encoder.id_embedding.weight" in state_dict and "state_embedding.weight" in model_state:
        state_dict["state_embedding.weight"] = state_dict.pop("state_encoder.id_embedding.weight")

    model.load_state_dict(state_dict, strict=False)


def case_type_for(world: str, state: int, action: int, target: int, cfg: Dict) -> str:
    if world == "number_line":
        worlds = cfg.get("worlds", {})
        train_ranges = worlds.get("train_ranges")
        heldout_ranges = worlds.get("heldout_ranges")
        if train_ranges and heldout_ranges:
            train_ranges = [tuple(r) for r in train_ranges]
            heldout_ranges = [tuple(r) for r in heldout_ranges]
            if train_ranges[0][0] <= state <= train_ranges[0][1]:
                return "range_train_low"
            if len(train_ranges) > 1 and train_ranges[1][0] <= state <= train_ranges[1][1]:
                return "range_train_high"
            if heldout_ranges[0][0] <= state <= heldout_ranges[0][1]:
                return "range_heldout_gap"
            if len(heldout_ranges) > 1 and heldout_ranges[1][0] <= state <= heldout_ranges[1][1]:
                return "range_heldout_extrap"
            return "range_unknown"
        train_max = int(worlds["train_max_number"])
        return "normal_small" if state <= train_max else "normal_heldout"

    if world == "modulo_10":
        modulo_n = int(cfg["worlds"]["modulo_n"])
        raw = state + action
        if raw < 0:
            return "modulo_negative_wrap"
        if raw >= modulo_n:
            return "modulo_positive_wrap"
        return "modulo_no_wrap"

    return "unknown"


def region_case_rows(world: str, split: str, case_type: str, activations: torch.Tensor, region_ids: torch.Tensor) -> List[Dict]:
    rows: List[Dict] = []
    activation_values = activations.detach().cpu().numpy()
    region_values = region_ids.detach().cpu().numpy()
    for region_id in sorted(set(region_values.tolist())):
        mask = region_values == region_id
        if not mask.any():
            continue
        region_name = ID_TO_REGION[int(region_id)]
        region_acts = activation_values[mask]
        rows.append({
            "world": world,
            "split": split,
            "case_type": case_type,
            "region": region_name,
            "unit_count": int(mask.sum()),
            "mean_activation": float(region_acts.mean()),
            "max_activation": float(region_acts.max()),
            "active_units_gt_0_5": int((region_acts > 0.5).sum()),
        })
    return rows


@torch.no_grad()
def evaluate_transition_grid(model, cfg: Dict, device: torch.device) -> Tuple[pd.DataFrame, List[Dict]]:
    rows: List[Dict] = []
    region_rows: List[Dict] = []
    actions = cfg["worlds"]["actions"]
    max_number = int(cfg["worlds"]["max_number"])
    modulo_n = int(cfg["worlds"]["modulo_n"])
    train_ranges = cfg["worlds"].get("train_ranges")
    heldout_ranges = cfg["worlds"].get("heldout_ranges")
    range_mode = bool(train_ranges and heldout_ranges)
    if range_mode:
        train_ranges = [tuple(r) for r in train_ranges]
        heldout_ranges = [tuple(r) for r in heldout_ranges]

    model.eval()
    state_bank = build_state_bank(model, cfg, device)

    for state in range(0, max_number + 1):
        for action in actions:
            target = state + action
            if 0 <= target <= max_number:
                classifier_pred, nearest_pred = predict_both(model, device, state_bank, WORLD_ID["number_line"], state, ACTION_TO_ID[action])
                logits, trace = forward_with_trace(model, device, "number_line", state, ACTION_TO_ID[action])
                if range_mode:
                    if train_ranges[0][0] <= state <= train_ranges[0][1]:
                        split = "range_train_low"
                    elif len(train_ranges) > 1 and train_ranges[1][0] <= state <= train_ranges[1][1]:
                        split = "range_train_high"
                    elif heldout_ranges[0][0] <= state <= heldout_ranges[0][1]:
                        split = "range_heldout_gap"
                    elif len(heldout_ranges) > 1 and heldout_ranges[1][0] <= state <= heldout_ranges[1][1]:
                        split = "range_heldout_extrap"
                    else:
                        split = "range_unknown"
                else:
                    split = "train_range" if state <= int(cfg["worlds"]["train_max_number"]) else "heldout_range"
                region_rows.extend(region_case_rows(
                    "number_line",
                    split,
                    case_type_for("number_line", state, action, target, cfg),
                    trace["activations"][0],
                    trace["region_ids"],
                ))
                rows.append({
                    "world": "number_line",
                    "state": state,
                    "action": action,
                    "target": target,
                    "case_type": case_type_for("number_line", state, action, target, cfg),
                    "classifier_prediction": classifier_pred,
                    "nearest_prediction": nearest_pred,
                    "classifier_correct": classifier_pred == target,
                    "nearest_correct": nearest_pred == target,
                    "classifier_absolute_error": abs(classifier_pred - target),
                    "nearest_absolute_error": abs(nearest_pred - target),
                    "split": split,
                })

    for state in range(0, modulo_n):
        for action in actions:
            target = (state + action) % modulo_n
            classifier_pred, nearest_pred = predict_both(model, device, state_bank, WORLD_ID["modulo_10"], state, ACTION_TO_ID[action])
            logits, trace = forward_with_trace(model, device, "modulo_10", state, ACTION_TO_ID[action])
            region_rows.extend(region_case_rows(
                "modulo_10",
                "modulo_all",
                case_type_for("modulo_10", state, action, target, cfg),
                trace["activations"][0],
                trace["region_ids"],
            ))
            rows.append({
                "world": "modulo_10",
                "state": state,
                "action": action,
                "target": target,
                "case_type": case_type_for("modulo_10", state, action, target, cfg),
                "classifier_prediction": classifier_pred,
                "nearest_prediction": nearest_pred,
                "classifier_correct": classifier_pred == target,
                "nearest_correct": nearest_pred == target,
                "classifier_absolute_error": abs(classifier_pred - target),
                "nearest_absolute_error": abs(nearest_pred - target),
                "split": "modulo_all",
            })

    df = pd.DataFrame(rows)
    return df, region_rows


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


@torch.no_grad()
def forward_with_trace(model, device, world_name: str, state: int, action_id: int):
    s = torch.tensor([state], dtype=torch.long, device=device)
    a = torch.tensor([action_id], dtype=torch.long, device=device)
    w = torch.tensor([WORLD_ID[world_name]], dtype=torch.long, device=device)
    return model(s, a, w)


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
    load_compatible_state_dict(model, run_dir / "checkpoints" / "final.pt")

    df, region_case_rows_out = evaluate_transition_grid(model, cfg, device)
    out = run_dir / "evaluation.csv"
    df.to_csv(out, index=False)

    if region_case_rows_out:
        region_case_df = pd.DataFrame(region_case_rows_out)
        region_case_summary = region_case_df.groupby(["world", "split", "case_type", "region"]).agg(
            mean_activation=("mean_activation", "mean"),
            max_activation=("max_activation", "max"),
            mean_unit_count=("unit_count", "mean"),
            active_units_gt_0_5=("active_units_gt_0_5", "sum"),
            samples=("region", "count"),
        ).reset_index()
        region_case_summary.to_csv(run_dir / "region_case_activation_summary.csv", index=False)

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
