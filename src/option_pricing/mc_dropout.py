"""Monte-Carlo dropout: T stochastic forward passes with dropout active at inference.

Only dropout layers (``nn.Dropout``, ``Dropout1d/2d/3d``, ``AlphaDropout``) are switched to
train mode; the rest of the model stays in eval mode.
The mean of the passes is the prediction and their std (ddof=0) the uncertainty. The 95 %
interval is ``mean ± 1.96·std``, but it is NOT a calibrated interval: the BS target is
deterministic and the std is only the spread of dropout passes, so coverage can be over- or
under-estimated (it is judged against a constant-width baseline and Spearman(std, |error|)).
The CLI reports on the validation split by default; use ``--split test`` for the final evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn.modules.dropout import _DropoutNd

from option_pricing.data import Dataset, Split, make_dataset
from option_pricing.evaluate import metrics, moneyness_masks, predict
from option_pricing.models import MODELS
from option_pricing.train import train

Z95 = 1.959963984540054
Z_LEVELS = {"0p5": 0.5, "1": 1.0, "1p96": Z95}  # key tag -> z


def mc_dropout_predict(
    model: nn.Module, X: np.ndarray, n_passes: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over ``n_passes`` dropout-active forward passes (model mode restored)."""
    if n_passes < 2:
        raise ValueError("n_passes must be >= 2 (std is 0 with a single pass).")
    was_training = model.training
    model.eval()
    for m in model.modules():
        if isinstance(m, _DropoutNd):
            m.train()
    x = torch.as_tensor(X, dtype=torch.float32)
    # Save/restore the global RNG so the call is reproducible and has no side effect.
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            preds = np.stack([model(x).numpy().astype(float) for _ in range(n_passes)])
    finally:
        torch.set_rng_state(state)
        model.train(was_training)
    return preds.mean(axis=0), preds.std(axis=0)


def interval_coverage(y: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = Z95) -> float:
    """Fraction of targets inside ``mean ± z·std``."""
    return float((np.abs(mean - y) <= z * std).mean())


def _spearman(std: np.ndarray, err: np.ndarray) -> float:
    """Spearman(std, |error|), nan if undefined (constant input)."""
    if np.ptp(std) == 0 or np.ptp(err) == 0:
        return float("nan")
    return float(spearmanr(std, err).statistic)


def evaluate_mc(
    model: nn.Module,
    split: Split,
    n_passes: int = 100,
    seed: int = 42,
    *,
    calib: Split,
) -> dict[str, float]:
    """Deterministic vs MC-mean metrics and uncertainty quality on ``split``.

    Reports mean std, Spearman(std, |error|), coverage/width of ``mean ± z·std`` for
    z in (0.5, 1, 1.96), coverage per moneyness bucket, and a constant-width baseline interval
    whose half-width is the 95 % quantile of the deterministic ``|error|`` on ``calib``
    (required; if ``calib is split`` the baseline is in-sample and flagged as such).
    """
    mean, std = mc_dropout_predict(model, split.X, n_passes, seed)
    pred_det = predict(model, split.X)
    det = metrics(split.y, pred_det)
    mc = metrics(split.y, mean)
    half = float(np.quantile(np.abs(predict(model, calib.X) - calib.y), 0.95))
    err = np.abs(mean - split.y)
    res = {
        "rmse_det": det["rmse"],
        "rmse_mc": mc["rmse"],
        "mae_mc": mc["mae"],
        "median_rel_err_mc": mc["median_rel_err"],
        "mean_std": float(std.mean()),
        "spearman": _spearman(std, err),
        "baseline_half_width": half,
        "baseline_in_sample": float(calib is split),
        "baseline_coverage": float((np.abs(pred_det - split.y) <= half).mean()),
    }
    for tag, z in Z_LEVELS.items():
        res[f"coverage_{tag}std"] = interval_coverage(split.y, mean, std, z)
        res[f"width_{tag}std"] = float(2 * z * std.mean())
    for name, mask in moneyness_masks(split.X[:, 0]).items():
        if mask.any():
            res[f"coverage_1p96std_{name}"] = interval_coverage(
                split.y[mask], mean[mask], std[mask]
            )
    return res


def mc_table_markdown(res: dict[str, float], n_passes: int) -> str:
    """Markdown tables of ``evaluate_mc`` output (accuracy, then uncertainty)."""
    lines = [
        "| T | RMSE (det.) | RMSE (MC mean) | MAE (MC) | median rel. err (MC) |",
        "|---|---|---|---|---|",
        f"| {n_passes} | {res['rmse_det']:.3e} | {res['rmse_mc']:.3e} | {res['mae_mc']:.3e} "
        f"| {res['median_rel_err_mc']:.3e} |",
        "",
        "| interval | coverage | width |",
        "|---|---|---|",
    ]
    for tag, z in Z_LEVELS.items():
        lines.append(
            f"| mean ± {z:.2f}·std (mean width) | {res[f'coverage_{tag}std']:.3f} "
            f"| {res[f'width_{tag}std']:.3e} |"
        )
    in_sample = ", in-sample" if res.get("baseline_in_sample") else ""
    lines.append(
        f"| constant width (95 % abs. error quantile, calibrated on val{in_sample}) "
        f"| {res['baseline_coverage']:.3f} "
        f"| {2 * res['baseline_half_width']:.3e} |"
    )
    buckets = ", ".join(
        f"{n} {res[f'coverage_1p96std_{n}']:.3f}"
        for n in ("OTM", "ATM", "ITM")
        if f"coverage_1p96std_{n}" in res
    )
    lines += [
        "",
        f"mean std {res['mean_std']:.3e}; Spearman(std, |err|) {res['spearman']:.3f}; "
        f"coverage 1.96·std per bucket: {buckets}.",
        "",
        "The deterministic prediction remains the pricer; MC dropout is an uncertainty "
        "indicator, not a calibrated interval.",
    ]
    return "\n".join(lines) + "\n"


def run(
    dataset: Dataset, split_name: str = "val", n_passes: int = 100, seed: int = 42, **train_kwargs
) -> dict[str, float]:
    """Train one model then evaluate it with MC dropout on ``val`` or ``test``."""
    if split_name not in ("val", "test"):
        raise ValueError("split_name must be 'val' or 'test'.")
    model = train(dataset, seed=seed, **train_kwargs).model
    return evaluate_mc(model, getattr(dataset, split_name), n_passes, seed, calib=dataset.val)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m option_pricing.mc_dropout``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="mlp")
    p.add_argument("--passes", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    a = p.parse_args(argv)
    ds = make_dataset(seed=a.seed)
    res = run(
        ds,
        a.split,
        a.passes,
        a.seed,
        model_name=a.model,
        epochs=a.epochs,
        lr=a.lr,
        optimizer=a.optimizer,
    )
    md = mc_table_markdown(res, a.passes)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / f"mc_dropout_{a.split}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
