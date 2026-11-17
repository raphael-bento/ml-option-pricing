"""Reproducible training loop: MSE loss, SGD or Adam, best-on-validation checkpointing.

Inputs are standardised with a scaler fit on the train split only; the statistics live in the
returned model (buffers), so ``model(raw_X)`` and ``predict`` work on raw features.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from option_pricing.data import Dataset, make_dataset
from option_pricing.models import MODELS, build_model
from option_pricing.preprocessing import StandardScaler

OPTIMIZERS = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)


class ScaledModel(nn.Module):
    """Standardise raw inputs with train-set statistics, then apply ``net``."""

    def __init__(self, net: nn.Module, scaler: StandardScaler) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.as_tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(scaler.std_, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions of shape (n,) from raw (unscaled) features."""
        return self.net((x - self.mean) / self.std)


@dataclass
class TrainResult:
    """Trained model (restored to its best-validation weights) and loss history."""

    model: nn.Module
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (global seed for third-party code)
    torch.manual_seed(seed)


def _tensors(split) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.as_tensor(split.X, dtype=torch.float32), torch.as_tensor(
        split.y, dtype=torch.float32
    )


def train(
    dataset: Dataset,
    model_name: str = "mlp",
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 64,
    seed: int = 42,
    optimizer: str = "sgd",
) -> TrainResult:
    """Train a model with SGD (or Adam if ``optimizer="adam"``) on the train split.

    Inputs are standardised with a scaler fit on the train split. The validation loss is
    tracked each epoch.

    The test split is never touched. The returned model holds the weights of the epoch
    with the lowest validation MSE.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer!r}; choose from {sorted(OPTIMIZERS)}.")
    set_seed(seed)
    scaler = StandardScaler().fit(dataset.train.X)  # fit on train only
    model = ScaledModel(build_model(model_name), scaler)
    x_tr, y_tr = _tensors(dataset.train)
    x_val, y_val = _tensors(dataset.val)
    opt = OPTIMIZERS[optimizer](model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    result = TrainResult(model=model)
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr), generator=generator)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_val), y_val).item()
        result.train_loss.append(total / len(x_tr))
        result.val_loss.append(val)
        if val < best_val:
            best_val, result.best_epoch = val, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m option_pricing.train --model mlp``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="mlp")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="sgd")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = p.parse_args(argv)

    result = train(
        make_dataset(seed=args.seed),
        args.model,
        args.epochs,
        args.lr,
        args.batch_size,
        args.seed,
        args.optimizer,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), args.out_dir / f"{args.model}.pt")
    history = {
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "best_epoch": result.best_epoch,
    }
    (args.out_dir / f"{args.model}_history.json").write_text(json.dumps(history), encoding="utf-8")
    print(
        f"{args.model}: best epoch {result.best_epoch}, "
        f"val MSE {result.val_loss[result.best_epoch]:.3e}"
    )


if __name__ == "__main__":
    main()
