"""PyTorch architectures for the BS pricing function: MLP, residual and highway."""

from __future__ import annotations

import torch
from torch import nn

N_INPUTS = 4
HIDDEN = 50
DROPOUT = 0.1


class MLP(nn.Module):
    """Plain feed-forward net: 4 -> 50 -> 50 -> 1 with ReLU and dropout."""

    def __init__(self, hidden: int = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUTS, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (n,)."""
        return self.net(x).squeeze(-1)


class ResidualMLP(nn.Module):
    """Two hidden layers where the second adds a skip connection: ``h2 = h1 + f(h1)``."""

    def __init__(self, hidden: int = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_INPUTS, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (n,)."""
        h1 = self.drop(torch.relu(self.fc1(x)))
        h2 = self.drop(h1 + torch.relu(self.fc2(h1)))
        return self.out(h2).squeeze(-1)


class HighwayMLP(nn.Module):
    """Two hidden layers where the second is a highway layer with a learnable gate.

    ``h2 = g * f(h1) + (1 - g) * h1`` with ``g = sigmoid(W_g h1 + b_g)``.
    """

    def __init__(self, hidden: int = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_INPUTS, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (n,)."""
        h1 = self.drop(torch.relu(self.fc1(x)))
        g = torch.sigmoid(self.gate(h1))
        h2 = self.drop(g * torch.relu(self.fc2(h1)) + (1 - g) * h1)
        return self.out(h2).squeeze(-1)


MODELS: dict[str, type[nn.Module]] = {
    "mlp": MLP,
    "residual": ResidualMLP,
    "highway": HighwayMLP,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name (`mlp`, `residual` or `highway`)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}.")
    return MODELS[name](**kwargs)
