from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MicroBrainTrace:
    activations: torch.Tensor
    confidences: torch.Tensor
    weights: torch.Tensor
    votes: torch.Tensor
    topk_indices: torch.Tensor
    decoded_prediction: torch.Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2):
        super().__init__()
        mods = []
        dim = in_dim
        for _ in range(max(0, layers - 1)):
            mods.append(nn.Linear(dim, hidden_dim))
            mods.append(nn.GELU())
            dim = hidden_dim
        mods.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CortexMicroBrainSwarm(nn.Module):
    def __init__(
        self,
        num_microbrains: int,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        adapter_dim: int,
        top_k: int,
    ):
        super().__init__()
        self.num_microbrains = num_microbrains
        self.embedding_dim = embedding_dim
        self.top_k = top_k

        self.micro_id = nn.Embedding(num_microbrains, adapter_dim)
        self.shared_core = MLP(input_dim + adapter_dim, hidden_dim, hidden_dim, layers=2)
        self.activation_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.vote_head = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = x.shape[0]
        ids = torch.arange(self.num_microbrains, device=x.device)
        id_emb = self.micro_id(ids)

        x_exp = x.unsqueeze(1).expand(batch, self.num_microbrains, x.shape[-1])
        id_exp = id_emb.unsqueeze(0).expand(batch, self.num_microbrains, id_emb.shape[-1])
        unit_input = torch.cat([x_exp, id_exp], dim=-1)

        h = self.shared_core(unit_input)
        raw_activation = self.activation_head(h).squeeze(-1)
        activations = torch.sigmoid(raw_activation)

        top_k = min(self.top_k, self.num_microbrains)
        top_values, top_indices = torch.topk(activations, k=top_k, dim=1)

        gathered_h = torch.gather(
            h,
            1,
            top_indices.unsqueeze(-1).expand(batch, top_k, h.shape[-1])
        )

        confidences = torch.sigmoid(self.confidence_head(gathered_h).squeeze(-1))
        votes = self.vote_head(gathered_h)

        weights = top_values * confidences
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        aggregated_delta = (votes * weights.unsqueeze(-1)).sum(dim=1)

        trace = {
            "activations": activations.detach(),
            "topk_indices": top_indices.detach(),
            "topk_activations": top_values.detach(),
            "confidences": confidences.detach(),
            "weights": weights.detach(),
            "votes": votes.detach(),
        }
        return aggregated_delta, trace


class WBMICROJEPA(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        m = cfg["model"]
        self.num_numbers = int(m["num_numbers"])
        self.embedding_dim = int(m["embedding_dim"])

        self.state_embedding = nn.Embedding(m["num_numbers"], m["embedding_dim"])
        self.action_embedding = nn.Embedding(m["num_actions"], m["action_dim"])
        self.world_embedding = nn.Embedding(m["num_worlds"], m["world_dim"])

        combined_dim = m["embedding_dim"] + m["action_dim"] + m["world_dim"]

        self.input_norm = nn.LayerNorm(combined_dim)
        self.cortex = CortexMicroBrainSwarm(
            num_microbrains=m["num_microbrains"],
            input_dim=combined_dim,
            embedding_dim=m["embedding_dim"],
            hidden_dim=m["hidden_dim"],
            adapter_dim=m["adapter_dim"],
            top_k=m["top_k_microbrains"],
        )

        self.decoder = MLP(m["embedding_dim"], m["hidden_dim"], m["num_numbers"], layers=2)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_embedding(state)

    def forward(self, state: torch.Tensor, action_id: torch.Tensor, world_id: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state_emb = self.state_embedding(state)
        action_emb = self.action_embedding(action_id)
        world_emb = self.world_embedding(world_id)

        x = torch.cat([state_emb, action_emb, world_emb], dim=-1)
        x = self.input_norm(x)

        delta, trace = self.cortex(x)
        predicted_target_emb = state_emb + delta
        logits = self.decoder(predicted_target_emb)
        decoded = logits.argmax(dim=-1)

        trace["state_embedding"] = state_emb.detach()
        trace["predicted_target_embedding"] = predicted_target_emb.detach()
        trace["decoded_prediction"] = decoded.detach()
        return logits, trace
