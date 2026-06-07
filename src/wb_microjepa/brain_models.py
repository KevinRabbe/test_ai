from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn

from .brain_topology import BrainGraph, build_brain_graph, REGION_TO_ID
from .models import MLP


class NumericStateEncoder(nn.Module):
    """State encoder with optional continuous number features.

    This is not an answer shortcut. It gives the model a continuous input
    representation so held-out numeric states are not pure unseen ID rows.
    """

    def __init__(self, num_numbers: int, embedding_dim: int, hidden_dim: int, use_numeric_features: bool, id_weight: float):
        super().__init__()
        self.num_numbers = int(num_numbers)
        self.embedding_dim = int(embedding_dim)
        self.use_numeric_features = bool(use_numeric_features)
        self.id_weight = float(id_weight)
        self.id_embedding = nn.Embedding(num_numbers, embedding_dim)

        if self.use_numeric_features:
            # n/max, centered n, parity, sin/cos low frequency.
            self.numeric_mlp = MLP(5, hidden_dim, embedding_dim, layers=2)
        else:
            self.numeric_mlp = None

    def _features(self, state: torch.Tensor) -> torch.Tensor:
        s = state.float()
        denom = float(max(1, self.num_numbers - 1))
        n01 = s / denom
        centered = n01 * 2.0 - 1.0
        parity = (state % 2).float()
        phase = n01 * math.tau
        return torch.stack([n01, centered, parity, torch.sin(phase), torch.cos(phase)], dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        id_emb = self.id_embedding(state)
        if not self.use_numeric_features:
            return id_emb
        numeric_emb = self.numeric_mlp(self._features(state))
        return numeric_emb + self.id_weight * id_emb


class BrainGraphCortexSwarm(nn.Module):
    """Topology-aware micro-brain swarm.

    The topology affects computation through local neighbor smoothing over a
    deterministic 3D BrainGraph. It does not encode task answers.
    """

    def __init__(
        self,
        graph: BrainGraph,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        adapter_dim: int,
        top_k: int,
        message_passing_steps: int,
        world_dim: int,
        use_world_modulation: bool,
        world_modulation_strength: float,
        world_modulation_regions: Optional[List[str]],
        global_world_modulation_strength: Optional[float] = None,
        targeted_world_modulation_strength: Optional[float] = None,
    ):
        super().__init__()
        self.graph = graph
        self.num_units = graph.num_units
        self.embedding_dim = embedding_dim
        self.top_k = min(top_k, self.num_units)
        self.message_passing_steps = int(message_passing_steps)
        self.use_world_modulation = bool(use_world_modulation)
        self.legacy_world_modulation_strength = float(world_modulation_strength)
        self.world_modulation_regions = list(world_modulation_regions or [])
        allowed_region_ids = {
            REGION_TO_ID[region_name]
            for region_name in self.world_modulation_regions
            if region_name in REGION_TO_ID
        }
        has_target_regions = len(allowed_region_ids) > 0
        if global_world_modulation_strength is None and targeted_world_modulation_strength is None:
            self.global_world_modulation_strength = float(
                self.legacy_world_modulation_strength if not has_target_regions else 0.0
            )
            self.targeted_world_modulation_strength = float(
                self.legacy_world_modulation_strength if has_target_regions else 0.0
            )
        else:
            self.global_world_modulation_strength = float(global_world_modulation_strength or 0.0)
            self.targeted_world_modulation_strength = float(
                targeted_world_modulation_strength if targeted_world_modulation_strength is not None else 0.0
            )

        self.unit_id = nn.Embedding(self.num_units, adapter_dim)
        self.region_embedding = nn.Embedding(len(REGION_TO_ID), adapter_dim)
        self.coord_projection = nn.Linear(3, adapter_dim)

        self.shared_core = MLP(input_dim + adapter_dim * 3, hidden_dim, hidden_dim, layers=2)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.message_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.vote_head = nn.Linear(hidden_dim, embedding_dim)

        if self.use_world_modulation:
            self.world_scale = nn.Linear(world_dim, hidden_dim)
            self.world_shift = nn.Linear(world_dim, hidden_dim)
            nn.init.zeros_(self.world_scale.weight)
            nn.init.zeros_(self.world_scale.bias)
            nn.init.zeros_(self.world_shift.weight)
            nn.init.zeros_(self.world_shift.bias)
        else:
            self.world_scale = None
            self.world_shift = None

        self.register_buffer("neighbor_indices", graph.neighbor_indices)
        self.register_buffer("neighbor_mask", graph.neighbor_mask)
        self.register_buffer("coords", graph.coords)
        self.register_buffer("region_ids", graph.region_ids)
        modulation_mask = torch.tensor(
            [1.0 if REGION_TO_ID[u.region] in allowed_region_ids else 0.0 for u in graph.units],
            dtype=torch.float32,
        ).unsqueeze(-1)
        self.register_buffer("world_modulation_mask", modulation_mask)

    def _local_message_pass(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, U, H]
        if self.message_passing_steps <= 0:
            return h

        batch, units, hidden = h.shape
        gather_idx = self.neighbor_indices.unsqueeze(0).unsqueeze(-1).expand(batch, units, self.neighbor_indices.shape[1], hidden)
        mask = self.neighbor_mask.unsqueeze(0).unsqueeze(-1)

        for _ in range(self.message_passing_steps):
            expanded = h.unsqueeze(1).expand(batch, units, units, hidden)
            neighbor_h = torch.gather(expanded, 2, gather_idx)
            denom = mask.sum(dim=2).clamp_min(1.0)
            neighbor_mean = (neighbor_h * mask).sum(dim=2) / denom
            h = self.message_norm(h + self.message_projection(neighbor_mean))
        return h

    def forward(self, x: torch.Tensor, world_emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = x.shape[0]
        ids = torch.arange(self.num_units, device=x.device)
        unit_emb = self.unit_id(ids)
        region_emb = self.region_embedding(self.region_ids.to(x.device))
        coord_emb = self.coord_projection(self.coords.to(x.device))

        topology_emb = torch.cat([unit_emb, region_emb, coord_emb], dim=-1)
        x_exp = x.unsqueeze(1).expand(batch, self.num_units, x.shape[-1])
        topo_exp = topology_emb.unsqueeze(0).expand(batch, self.num_units, topology_emb.shape[-1])

        h = self.shared_core(torch.cat([x_exp, topo_exp], dim=-1))
        if self.use_world_modulation and world_emb is not None:
            scale = self.world_scale(world_emb)
            shift = self.world_shift(world_emb)
            if self.global_world_modulation_strength > 0.0:
                h = h * (1.0 + self.global_world_modulation_strength * scale.unsqueeze(1)) + self.global_world_modulation_strength * shift.unsqueeze(1)
            if self.targeted_world_modulation_strength > 0.0:
                mask = self.world_modulation_mask.to(h.device).unsqueeze(0)
                h = h * (1.0 + mask * self.targeted_world_modulation_strength * scale.unsqueeze(1)) + mask * self.targeted_world_modulation_strength * shift.unsqueeze(1)
        h = self._local_message_pass(h)

        raw_activation = self.activation_head(h).squeeze(-1)
        activations = torch.sigmoid(raw_activation)
        top_values, top_indices = torch.topk(activations, k=self.top_k, dim=1)

        gathered_h = torch.gather(
            h,
            1,
            top_indices.unsqueeze(-1).expand(batch, self.top_k, h.shape[-1]),
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
            "coords": self.coords.detach().cpu(),
            "region_ids": self.region_ids.detach().cpu(),
        }
        if self.use_world_modulation and world_emb is not None:
            trace["world_scale"] = scale.detach()
            trace["world_shift"] = shift.detach()
            trace["world_modulation_mask"] = self.world_modulation_mask.detach().cpu()
            trace["global_world_modulation_strength"] = torch.tensor(self.global_world_modulation_strength)
            trace["targeted_world_modulation_strength"] = torch.tensor(self.targeted_world_modulation_strength)
        return aggregated_delta, trace


class BrainGraphMicroJEPA(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        m = cfg["model"]
        self.graph = build_brain_graph(cfg)
        self.num_numbers = int(m["num_numbers"])
        self.embedding_dim = int(m["embedding_dim"])
        self.use_action_delta_basis = bool(m.get("use_action_delta_basis", False))
        self.action_delta_basis_weight = float(m.get("action_delta_basis_weight", 0.0))

        self.state_encoder = NumericStateEncoder(
            num_numbers=m["num_numbers"],
            embedding_dim=m["embedding_dim"],
            hidden_dim=m["hidden_dim"],
            use_numeric_features=bool(m.get("use_numeric_state_features", False)),
            id_weight=float(m.get("state_id_weight", 1.0)),
        )
        self.action_embedding = nn.Embedding(m["num_actions"], m["action_dim"])
        self.world_embedding = nn.Embedding(m["num_worlds"], m["world_dim"])
        if self.use_action_delta_basis:
            self.action_delta_basis = MLP(m["action_dim"] + m["world_dim"], m["hidden_dim"], m["embedding_dim"], layers=2)
        else:
            self.action_delta_basis = None

        combined_dim = m["embedding_dim"] + m["action_dim"] + m["world_dim"]
        self.input_norm = nn.LayerNorm(combined_dim)
        self.cortex = BrainGraphCortexSwarm(
            graph=self.graph,
            input_dim=combined_dim,
            embedding_dim=m["embedding_dim"],
            hidden_dim=m["hidden_dim"],
            adapter_dim=m["adapter_dim"],
            top_k=m["top_k_microbrains"],
            message_passing_steps=int(cfg.get("topology", {}).get("message_passing_steps", 1)),
            world_dim=m["world_dim"],
            use_world_modulation=bool(m.get("use_world_modulation", False)),
            world_modulation_strength=float(m.get("world_modulation_strength", 1.0)),
            world_modulation_regions=list(m.get("world_modulation_regions", [])),
            global_world_modulation_strength=(
                float(m["global_world_modulation_strength"]) if "global_world_modulation_strength" in m else None
            ),
            targeted_world_modulation_strength=(
                float(m["targeted_world_modulation_strength"]) if "targeted_world_modulation_strength" in m else None
            ),
        )
        self.decoder = MLP(m["embedding_dim"], m["hidden_dim"], m["num_numbers"], layers=2)

    @property
    def state_embedding(self):
        # Compatibility for older monitoring code that expects an embedding table.
        return self.state_encoder.id_embedding

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state)

    def forward(self, state: torch.Tensor, action_id: torch.Tensor, world_id: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state_emb = self.encode_state(state)
        action_emb = self.action_embedding(action_id)
        world_emb = self.world_embedding(world_id)
        x = self.input_norm(torch.cat([state_emb, action_emb, world_emb], dim=-1))

        braingraph_delta, trace = self.cortex(x, world_emb=world_emb)
        if self.use_action_delta_basis:
            action_basis_input = torch.cat([action_emb, world_emb], dim=-1)
            action_delta_basis = self.action_delta_basis(action_basis_input)
        else:
            action_delta_basis = torch.zeros_like(braingraph_delta)

        combined_delta = braingraph_delta + self.action_delta_basis_weight * action_delta_basis
        predicted_target_emb = state_emb + combined_delta
        logits = self.decoder(predicted_target_emb)
        decoded = logits.argmax(dim=-1)

        trace["state_embedding"] = state_emb.detach()
        trace["braingraph_delta"] = braingraph_delta
        trace["action_delta_basis"] = action_delta_basis
        trace["predicted_delta"] = combined_delta
        trace["combined_delta"] = combined_delta
        trace["predicted_target_embedding"] = predicted_target_emb.detach()
        trace["decoded_prediction"] = decoded.detach()
        trace["model_type"] = "brain_graph_microjepa"
        return logits, trace
