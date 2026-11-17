"""Deep ensembles: N models trained with different seeds, mean prediction and std as uncertainty.

All members share the same dataset (and hence the same train-only scaler); only the seed
(weight init and batch order) changes: member ``i`` uses ``seed + i``.
The CLI reports on the validation split by default. Note that each member's weights are
selected by its best validation MSE, so validation metrics are optimistic (not a test leak);
use ``--split test`` for the final evaluation only.
Uncertainty quality is reported with ``mean_std``, the Spearman correlation between ``std`` and
``|mean - y|`` and the coverage of ``mean ± 2·std``. That coverage is not a 95 % interval: N is
small, ``std`` uses ddof=0 and the BS target is deterministic (``std`` only captures training
variance), so a low coverage is expected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from torch import nn

from option_pricing.data import Dataset, Split, make_dataset
from option_pricing.evaluate import metrics, predict
from option_pricing.models import MODELS
from option_pricing.train import train


@dataclass
class Ensemble:
    """Trained ensemble members."""

    models: list[nn.Module]

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) over members; std uses ddof=0 (0 for a single member)."""
        preds = np.stack([predict(m, X) for m in self.models])
        return preds.mean(axis=0), preds.std(axis=0)


def train_ensemble(
    dataset: Dataset,
    n_models: int = 5,
    model_name: str = "mlp",
    seed: int = 42,
    **train_kwargs,
) -> Ensemble:
    """Train ``n_models`` networks with seeds ``seed, seed+1, ...`` on the same dataset."""
    if n_models < 1:
        raise ValueError("n_models must be >= 1.")
    models = [
        train(dataset, model_name, seed=seed + i, **train_kwargs).model for i in range(n_models)
    ]
    return Ensemble(models)


def uncertainty_metrics(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, float]:
    """Mean std, Spearman(std, |error|) (nan if undefined) and coverage of mean ± 2·std."""
    err = np.abs(mean - y)
    finite = np.isfinite(std).all() and np.isfinite(err).all()
    if not finite or np.ptp(std) == 0 or np.ptp(err) == 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(std, err).statistic)
    return {
        "mean_std": float(std.mean()),
        "spearman": rho,
        "coverage_2std": float((err <= 2 * std).mean()),
    }


def compare(ensemble: Ensemble, split: Split) -> dict[str, dict[str, float]]:
    """Metrics of each member, their average, and the ensemble mean prediction."""
    table = {
        f"member {i}": metrics(split.y, predict(m, split.X)) for i, m in enumerate(ensemble.models)
    }
    keys = next(iter(table.values())).keys()
    table["single model (mean of members)"] = {
        k: float(np.mean([row[k] for row in table.values()])) for k in keys
    }
    mean, std = ensemble.predict(split.X)
    table[f"ensemble (N={len(ensemble.models)})"] = metrics(split.y, mean) | uncertainty_metrics(
        split.y, mean, std
    )
    return table


def table_markdown(table: dict[str, dict[str, float]]) -> str:
    """Markdown table of RMSE / MAE / median rel. error, plus uncertainty columns when present."""
    lines = [
        "| model | RMSE | MAE | median rel. err | mean std | Spearman(std, err) | cov. 2std |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in table.items():
        std = f"{m['mean_std']:.3e}" if "mean_std" in m else ""
        rho = f"{m['spearman']:.3f}" if "spearman" in m else ""
        cov = f"{m['coverage_2std']:.3f}" if "coverage_2std" in m else ""
        lines.append(
            f"| {name} | {m['rmse']:.3e} | {m['mae']:.3e} | {m['median_rel_err']:.3e} "
            f"| {std} | {rho} | {cov} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m option_pricing.uncertainty``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="mlp")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    a = p.parse_args(argv)
    ds = make_dataset(seed=a.seed)
    ens = train_ensemble(
        ds, a.n_models, a.model, a.seed, epochs=a.epochs, lr=a.lr, optimizer=a.optimizer
    )
    table = compare(ens, getattr(ds, a.split))
    md = table_markdown(table)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"ensemble_{a.split}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
