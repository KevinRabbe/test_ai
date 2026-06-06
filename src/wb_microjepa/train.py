import argparse
import json
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
from .monitoring import DiagnosticsLogger


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(cfg: Dict) -> torch.device:
    requested = cfg["training"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def batch_to_tensors(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "world_id": torch.tensor(batch["world_id"], dtype=torch.long, device=device),
        "state": torch.tensor(batch["state"], dtype=torch.long, device=device),
        "action_id": torch.tensor(batch["action_id"], dtype=torch.long, device=device),
        "target": torch.tensor(batch["target"], dtype=torch.long, device=device),
    }


def compute_losses(model, logits, trace, target, cfg):
    target_emb = model.encode_state(target).detach()
    pred_emb = trace["predicted_target_embedding"]

    prediction_loss = F.mse_loss(pred_emb, target_emb)
    identity_loss = F.cross_entropy(logits, target)

    # Sparsity encourages not all micro-brains activating.
    activations = trace["activations"]
    sparsity_loss = activations.mean()

    # Diversity proxy on selected votes: discourage identical votes.
    votes = trace["votes"]
    if votes.shape[1] > 1:
        normalized = F.normalize(votes, dim=-1)
        sim = torch.matmul(normalized, normalized.transpose(1, 2))
        eye = torch.eye(sim.shape[1], device=sim.device).unsqueeze(0)
        diversity_loss = ((sim * (1 - eye)) ** 2).mean()
    else:
        diversity_loss = torch.tensor(0.0, device=logits.device)

    lw = cfg["loss"]
    loss = (
        lw["prediction_weight"] * prediction_loss
        + lw["identity_weight"] * identity_loss
        + lw["sparsity_weight"] * sparsity_loss
        + lw["diversity_weight"] * diversity_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "identity_loss": float(identity_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "diversity_loss": float(diversity_loss.detach().cpu()),
    }


@torch.no_grad()
def run_probes(model, device, logger: DiagnosticsLogger, epoch: int):
    probes = fixed_probe_set()
    labels = []
    activation_rows = []
    traces_json = []

    model.eval()
    for p in probes:
        state = torch.tensor([p.state], dtype=torch.long, device=device)
        action_id = torch.tensor([p.action_id], dtype=torch.long, device=device)
        world_id = torch.tensor([p.world_id], dtype=torch.long, device=device)

        logits, trace = model(state, action_id, world_id)
        pred = int(logits.argmax(dim=-1).item())
        correct = pred == p.target

        activations = trace["activations"][0].cpu().numpy()
        activation_rows.append(activations)
        labels.append(f"{p.world_name}:{p.state}{p.action:+d}->{p.target}")

        topk = trace["topk_indices"][0].cpu().numpy().tolist()
        weights = trace["weights"][0].cpu().numpy().tolist()
        confs = trace["confidences"][0].cpu().numpy().tolist()

        top_microbrains = []
        for i in range(min(10, len(topk))):
            top_microbrains.append({
                "id": f"MB_{topk[i]:04d}",
                "index": int(topk[i]),
                "weight": float(weights[i]),
                "confidence": float(confs[i]),
            })

        traces_json.append({
            "world": p.world_name,
            "state": p.state,
            "action": p.action,
            "target": p.target,
            "prediction": pred,
            "correct": correct,
            "top_microbrains": top_microbrains,
        })

    logger.save_activation_heatmap(epoch, np.stack(activation_rows), labels)
    logger.log_probe_traces(epoch, traces_json)

    embeddings = model.state_embedding.weight.detach().cpu().numpy()
    logger.save_embedding_plot(epoch, embeddings, title="State embeddings")

    activation_matrix = np.stack(activation_rows)
    stats = []
    for mb in range(activation_matrix.shape[1]):
        stats.append({
            "microbrain_id": f"MB_{mb:04d}",
            "mean_probe_activation": float(activation_matrix[:, mb].mean()),
            "max_probe_activation": float(activation_matrix[:, mb].max()),
            "active_on_probe_count_gt_0_5": int((activation_matrix[:, mb] > 0.5).sum()),
        })
    logger.save_microbrain_stats(pd.DataFrame(stats))

    model.train()


def train(config_path: str):
    cfg = load_config(config_path)
    set_deterministic(cfg["run"]["seed"], cfg["run"]["deterministic"])
    device = choose_device(cfg)

    run_dir = Path(cfg["run"]["output_dir"]) / cfg["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

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

    model = WBMICROJEPA(cfg).to(device)
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
        loss, loss_parts = compute_losses(model, logits, trace, tb["target"], cfg)
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
