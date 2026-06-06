from typing import Dict, Tuple
import torch
import torch.nn as nn

from .models import MLP


class DenseJEPA(nn.Module):
    """Dense baseline with the same state/action/world transition task.

    This baseline is intentionally not whitebox like the micro-brain swarm.
    It exists to test whether the micro-brain parameter arrangement gives
    better or more interpretable behavior than a normal dense model.
    """

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
        self.predictor = MLP(combined_dim, m["hidden_dim"], m["embedding_dim"], layers=3)
        self.decoder = MLP(m["embedding_dim"], m["hidden_dim"], m["num_numbers"], layers=2)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_embedding(state)

    def forward(self, state: torch.Tensor, action_id: torch.Tensor, world_id: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state_emb = self.state_embedding(state)
        action_emb = self.action_embedding(action_id)
        world_emb = self.world_embedding(world_id)
        x = torch.cat([state_emb, action_emb, world_emb], dim=-1)
        x = self.input_norm(x)

        delta = self.predictor(x)
        predicted_target_emb = state_emb + delta
        logits = self.decoder(predicted_target_emb)
        decoded = logits.argmax(dim=-1)

        trace = {
            "state_embedding": state_emb.detach(),
            "predicted_target_embedding": predicted_target_emb.detach(),
            "decoded_prediction": decoded.detach(),
            "baseline": "dense_jepa",
        }
        return logits, trace
