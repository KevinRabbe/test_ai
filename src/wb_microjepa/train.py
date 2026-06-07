import argparse
import json
import shutil
from pathlib import Path
from typing import Dict
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .determinism import set_deterministic
from .worlds import MixedWorldSampler, fixed_probe_set
from .models import WBMICROJEPA
from .brain_models import BrainGraphMicroJEPA
from .monitoring import DiagnosticsLogger
from .brain_topology import ID_TO_REGION, region_count_summary


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(cfg: Dict) -> torch.device:
    requested = cfg["training"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(cfg: Dict):
    model_type = cfg.get("model", {}).get("type", "flat_microjepa")
    if model_type == "flat_microjepa":
        return WBMICROJEPA(cfg)
    if model_type == "brain_graph_microjepa":
        return BrainGraphMicroJEPA(cfg)
    raise ValueError(f"Unknown model.type: {model_type}")


def batch_to_tensors(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "world_id": torch.tensor(batch["world_id"], dtype=torch.long, device=device),
        "state": torch.tensor(batch["state"], dtype=torch.long, device=device),
        "action_id": torch.tensor(batch["action_id"], dtype=torch.long, device=device),
        "target": torch.tensor(batch["target"], dtype=torch.long, device=device),
    }


def _decoder_autoencode_loss(model, cfg: Dict, device: torch.device) -> torch.Tensor:
    lw = cfg["loss"]
    weight = float(lw.get("decoder_autoencode_weight", 0.0))
    if weight <= 0.0:
        return torch.tensor(0.0, device=device)

    num_numbers = int(cfg["model"]["num_numbers"])
    max_samples = int(lw.get("decoder_autoencode_samples", num_numbers))
    if max_samples >= num_numbers:
        states = torch.arange(num_numbers, dtype=torch.long, device=device)
    else:
        states = torch.randint(0, num_numbers, (max_samples,), dtype=torch.long, device=device)

    encoded = model.encode_state(states)
    logits = model.decoder(encoded)
    return F.cross_entropy(logits, states)


def _state_smoothness_loss(model, cfg: Dict, device: torch.device) -> torch.Tensor:
    lw = cfg["loss"]
    weight = float(lw.get("state_smoothness_weight", 0.0))
    if weight <= 0.0:
        return torch.tensor(0.0, device=device)

    num_numbers = int(cfg["model"]["num_numbers"])
    states = torch.arange(num_numbers, dtype=torch.long, device=device)
    emb = model.encode_state(states)
    if emb.shape[0] < 3:
        return torch.tensor(0.0, device=device)
    first_diff = emb[1:] - emb[:-1]
    second_diff = first_diff[1:] - first_diff[:-1]
    return (second_diff ** 2).mean()


def _delta_consistency_loss(model, trace, state: torch.Tensor, target: torch.Tensor, cfg: Dict) -> torch.Tensor:
    lw = cfg["loss"]
    weight = float(lw.get("delta_consistency_weight", 0.0))
    if weight <= 0.0:
        return torch.tensor(0.0, device=target.device)

    state_emb = model.encode_state(state)
    target_emb = model.encode_state(target).detach()
    delta_target = (target_emb - state_emb).detach()
    predicted_delta = trace.get("predicted_delta")
    if predicted_delta is None:
        predicted_delta = trace["predicted_target_embedding"] - state_emb
    return F.mse_loss(predicted_delta, delta_target)


def compute_losses(model, logits, trace, state, target, cfg):
    device = logits.device
    target_emb = model.encode_state(target).detach()
    pred_emb = trace["predicted_target_embedding"]
    delta_loss = _delta_consistency_loss(model, trace, state, target, cfg)

    prediction_loss = F.mse_loss(pred_emb, target_emb)
    identity_loss = F.cross_entropy(logits, target)

    activations = trace["activations"]
    sparsity_loss = activations.mean()

    votes = trace["votes"]
    if votes.shape[1] > 1:
        normalized = F.normalize(votes, dim=-1)
        sim = torch.matmul(normalized, normalized.transpose(1, 2))
        eye = torch.eye(sim.shape[1], device=sim.device).unsqueeze(0)
        diversity_loss = ((sim * (1 - eye)) ** 2).mean()
    else:
        diversity_loss = torch.tensor(0.0, device=device)

    decoder_autoencode_loss = _decoder_autoencode_loss(model, cfg, device)
    state_smoothness_loss = _state_smoothness_loss(model, cfg, device)

    braingraph_delta = trace.get("braingraph_delta", trace.get("predicted_delta"))
    action_delta_basis = trace.get("action_delta_basis")
    combined_delta = trace.get("combined_delta", trace.get("predicted_delta"))
    if braingraph_delta is not None:
        braingraph_delta_norm = braingraph_delta.norm(dim=-1).mean()
    else:
        braingraph_delta_norm = torch.tensor(0.0, device=device)
    if action_delta_basis is not None:
        action_delta_norm = action_delta_basis.norm(dim=-1).mean()
        delta_cosine = F.cosine_similarity(braingraph_delta, action_delta_basis, dim=-1).mean() if braingraph_delta is not None else torch.tensor(0.0, device=device)
    else:
        action_delta_norm = torch.tensor(0.0, device=device)
        delta_cosine = torch.tensor(0.0, device=device)
    if combined_delta is not None:
        combined_delta_norm = combined_delta.norm(dim=-1).mean()
    else:
        combined_delta_norm = torch.tensor(0.0, device=device)

    lw = cfg["loss"]
    loss = (
        lw["prediction_weight"] * prediction_loss
        + lw["identity_weight"] * identity_loss
        + lw["sparsity_weight"] * sparsity_loss
        + lw["diversity_weight"] * diversity_loss
        + float(lw.get("delta_consistency_weight", 0.0)) * delta_loss
        + float(lw.get("decoder_autoencode_weight", 0.0)) * decoder_autoencode_loss
        + float(lw.get("state_smoothness_weight", 0.0)) * state_smoothness_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "identity_loss": float(identity_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "diversity_loss": float(diversity_loss.detach().cpu()),
        "delta_consistency_loss": float(delta_loss.detach().cpu()),
        "decoder_autoencode_loss": float(decoder_autoencode_loss.detach().cpu()),
        "state_smoothness_loss": float(state_smoothness_loss.detach().cpu()),
        "action_delta_norm": float(action_delta_norm.detach().cpu()),
        "braingraph_delta_norm": float(braingraph_delta_norm.detach().cpu()),
        "combined_delta_norm": float(combined_delta_norm.detach().cpu()),
        "delta_cosine_mean": float(delta_cosine.detach().cpu()),
    }


@torch.no_grad()
def run_probes(model, device, logger: DiagnosticsLogger, epoch: int):
    probes = fixed_probe_set()
    labels = []
    activation_rows = []
    traces_json = []
    action_rows = []

    model.eval()
    for p in probes:
        state = torch.tensor([p.state], dtype=torch.long, device=device)
        action_id = torch.tensor([p.action_id], dtype=torch.long, device=device)
        world_id = torch.tensor([p.world_id], dtype=torch.long, device=device)

        logits, trace = model(state, action_id, world_id)
        pred = int(logits.argmax(dim=-1).item())
        correct = pred == p.target

        braingraph_delta = trace.get("braingraph_delta", trace.get("predicted_delta"))
        action_delta_basis = trace.get("action_delta_basis")
        combined_delta = trace.get("combined_delta", trace.get("predicted_delta"))
        action_delta_norm = float(action_delta_basis.norm(dim=-1).mean().item()) if action_delta_basis is not None else 0.0
        braingraph_delta_norm = float(braingraph_delta.norm(dim=-1).mean().item()) if braingraph_delta is not None else 0.0
        combined_delta_norm = float(combined_delta.norm(dim=-1).mean().item()) if combined_delta is not None else 0.0
        delta_cosine = float(F.cosine_similarity(braingraph_delta, action_delta_basis, dim=-1).mean().item()) if braingraph_delta is not None and action_delta_basis is not None else 0.0

        activations = trace["activations"][0].cpu().numpy()
        activation_rows.append(activations)
        labels.append(f"{p.world_name}:{p.state}{p.action:+d}->{p.target}")

        topk = trace["topk_indices"][0].cpu().numpy().tolist()
        weights = trace["weights"][0].cpu().numpy().tolist()
        confs = trace["confidences"][0].cpu().numpy().tolist()
        region_ids = trace.get("region_ids")

        top_microbrains = []
        for i in range(min(10, len(topk))):
            entry = {
                "id": f"MB_{topk[i]:04d}",
                "index": int(topk[i]),
                "weight": float(weights[i]),
                "confidence": float(confs[i]),
            }
            if region_ids is not None:
                entry["region"] = ID_TO_REGION[int(region_ids[topk[i]])]
            top_microbrains.append(entry)

        traces_json.append({
            "world": p.world_name,
            "state": p.state,
            "action": p.action,
            "target": p.target,
            "prediction": pred,
            "correct": correct,
            "top_microbrains": top_microbrains,
            "action_delta_norm": action_delta_norm,
            "braingraph_delta_norm": braingraph_delta_norm,
            "combined_delta_norm": combined_delta_norm,
            "delta_cosine": delta_cosine,
        })

        action_rows.append({
            "world": p.world_name,
            "action": p.action,
            "action_delta_norm": action_delta_norm,
            "braingraph_delta_norm": braingraph_delta_norm,
            "combined_delta_norm": combined_delta_norm,
            "delta_cosine": delta_cosine,
        })

    logger.save_activation_heatmap(epoch, np.stack(activation_rows), labels)
    logger.log_probe_traces(epoch, traces_json)

    states = torch.arange(model.num_numbers, dtype=torch.long, device=device)
    embeddings = model.encode_state(states).detach().cpu().numpy()
    logger.save_embedding_plot(epoch, embeddings, title="State embeddings")

    activation_matrix = np.stack(activation_rows)
    stats = []
    region_ids = None
    if hasattr(model, "graph"):
        region_ids = model.graph.region_ids.cpu().numpy()
    for mb in range(activation_matrix.shape[1]):
        row = {
            "microbrain_id": f"MB_{mb:04d}",
            "mean_probe_activation": float(activation_matrix[:, mb].mean()),
            "max_probe_activation": float(activation_matrix[:, mb].max()),
            "active_on_probe_count_gt_0_5": int((activation_matrix[:, mb] > 0.5).sum()),
        }
        if region_ids is not None:
            row["region"] = ID_TO_REGION[int(region_ids[mb])]
        stats.append(row)
    logger.save_microbrain_stats(pd.DataFrame(stats))

    if hasattr(model, "graph"):
        logger.save_brain_projection(epoch, model.graph, activation_matrix.mean(axis=0))
        logger.save_region_activation_summary(epoch, model.graph, activation_matrix.mean(axis=0))
    logger.save_action_delta_summary(action_rows)

    model.train()


def prepare_run_dir(cfg: Dict) -> Path:
    run_dir = Path(cfg["run"]["output_dir"]) / cfg["run"]["name"]
    clear_output = bool(cfg.get("run", {}).get("clear_output", True))
    if clear_output and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def train(config_path: str):
    cfg = load_config(config_path)
    set_deterministic(cfg["run"]["seed"], cfg["run"]["deterministic"])
    device = choose_device(cfg)

    run_dir = prepare_run_dir(cfg)

    with (run_dir / "config_resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    logger = DiagnosticsLogger(run_dir)
    sampler = MixedWorldSampler(
        world_names=cfg["worlds"]["train_worlds"],
        max_number=cfg["worlds"]["max_number"],
        train_max_number=cfg["worlds"]["train_max_number"],
        actions=cfg["worlds"]["actions"],
        modulo_n=cfg["worlds"]["modulo_n"],
    )

    model = build_model(cfg).to(device)
    if hasattr(model, "graph"):
        print("BrainGraph region counts:", region_count_summary(model.graph))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])

    for epoch in range(1, epochs + 1):
        batch = sampler.sample_batch(batch_size)
        tb = batch_to_tensors(batch, device)

        opt.zero_grad(set_to_none=True)
        logits, trace = model(tb["state"], tb["action_id"], tb["world_id"])
        loss, loss_parts = compute_losses(model, logits, trace, tb["state"], tb["target"], cfg)
        loss.backward()
        opt.step()

        pred = logits.argmax(dim=-1)
        acc = (pred == tb["target"]).float().mean().item()

        if epoch % cfg["training"]["log_every_epochs"] == 0 or epoch == 1:
            row = {"epoch": epoch, "accuracy": acc, **loss_parts}
            logger.log_metrics(row)
            print(f"epoch={epoch:04d} loss={loss_parts['loss']:.4f} acc={acc:.3f}")

        if epoch % cfg["training"]["probe_every_epochs"] == 0 or epoch == 1:
            run_probes(model, device, logger, epoch)

    torch.save(model.state_dict(), run_dir / "checkpoints" / "final.pt")
    run_probes(model, device, logger, epochs)
    print(f"Done. Run saved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wb_microjepa_0a.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
