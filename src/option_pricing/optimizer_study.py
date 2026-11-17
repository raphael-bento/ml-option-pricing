"""Optimizer study: SGD vs Adam, crossed with two learning rates, same seed and epoch budget.

The report config is SGD lr=1e-4; Adam lr=1e-3 differs in both optimizer and lr, so the two
crossed configs (SGD lr=1e-3, Adam lr=1e-4) are added to separate the two effects.
Inputs are standardised (scaler fit on the train split), so SGD is not handicapped by raw
feature scales.
Caveats: each figure comes from a single seed (no variance estimate); metrics are on the
validation split, which also selects the best epoch, so they are slightly optimistic.

The comparison uses the validation split (the optimizer is a hyperparameter choice, so the
test set stays untouched). A constant baseline (train-mean prediction) shows whether a
configuration learned anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from option_pricing.data import make_dataset
from option_pricing.evaluate import metrics, predict
from option_pricing.models import MODELS
from option_pricing.train import TrainResult, train

CONFIGS = {
    "SGD lr=1e-4 (report)": ("sgd", 1e-4),
    "Adam lr=1e-3": ("adam", 1e-3),
    "SGD lr=1e-3 (crossed)": ("sgd", 1e-3),
    "Adam lr=1e-4 (crossed)": ("adam", 1e-4),
}


def run_study(
    model_name: str = "mlp", epochs: int = 100, seed: int = 42
) -> tuple[dict[str, TrainResult], dict[str, dict[str, float]]]:
    """Train each config; return results and validation metrics (plus a mean baseline)."""
    ds = make_dataset(seed=seed)
    results, table = {}, {}
    for name, (opt, lr) in CONFIGS.items():
        res = train(ds, model_name, epochs=epochs, lr=lr, seed=seed, optimizer=opt)
        results[name] = res
        table[name] = metrics(ds.val.y, predict(res.model, ds.val.X)) | {
            "best_epoch": res.best_epoch,
            "final_train_mse": res.train_loss[-1],
        }
    base = np.full_like(ds.val.y, ds.train.y.mean())
    table["Baseline (train mean)"] = metrics(ds.val.y, base)
    return results, table


def table_markdown(table: dict[str, dict[str, float]]) -> str:
    """Markdown table of validation metrics, one row per configuration."""
    lines = [
        "| config | val RMSE | val MAE | val median rel. err | best epoch | final train MSE |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in table.items():
        be = m.get("best_epoch", "-")
        ft = f"{m['final_train_mse']:.3e}" if "final_train_mse" in m else "-"
        lines.append(
            f"| {name} | {m['rmse']:.3e} | {m['mae']:.3e} | {m['median_rel_err']:.3e} "
            f"| {be} | {ft} |"
        )
    return "\n".join(lines) + "\n"


def plot_curves(results: dict[str, TrainResult], path: Path) -> Path:
    """Train (solid) and validation (dashed) loss curves, log scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in results.items():
        (line,) = ax.plot(res.train_loss, label=f"{name} train")
        ax.plot(res.val_loss, "--", color=line.get_color(), label=f"{name} val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m option_pricing.optimizer_study``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="mlp")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    a = p.parse_args(argv)
    results, table = run_study(a.model, a.epochs, a.seed)
    md = table_markdown(table)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "optimizer_study.md").write_text(md, encoding="utf-8")
    plot_curves(results, a.out_dir / "optimizer_study.png")
    print(md)


if __name__ == "__main__":
    main()
