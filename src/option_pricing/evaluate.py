"""Evaluation of pricing models against the exact Black-Scholes price."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

from option_pricing.data import Split

ATM_BAND = (0.95, 1.05)  # moneyness m = S0/K; below: OTM, above: ITM (for a call)
REL_EPS = 1e-8  # floor on the true price in the relative error (deep OTM prices ~ 0)
BUCKETS = ("OTM", "ATM", "ITM", "all")
METRICS = ("rmse", "mae", "median_rel_err")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and median relative error ``|pred - true| / max(true, REL_EPS)``."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same shape.")
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "median_rel_err": float(np.median(np.abs(err) / np.maximum(y_true, REL_EPS))),
    }


def moneyness_masks(m: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks for the OTM / ATM / ITM buckets (they partition the input)."""
    lo, hi = ATM_BAND
    return {"OTM": m < lo, "ATM": (m >= lo) & (m <= hi), "ITM": m > hi}


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Deterministic predictions (eval mode, no gradient) as a numpy array."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(float)


def evaluate_model(model: nn.Module, split: Split) -> dict[str, dict[str, float]]:
    """Metrics overall and per moneyness bucket; an empty bucket is omitted."""
    y_pred = predict(model, split.X)
    out = {"all": metrics(split.y, y_pred)}
    for name, mask in moneyness_masks(split.X[:, 0]).items():
        if mask.any():
            out[name] = metrics(split.y[mask], y_pred[mask])
    return out


def results_markdown(results: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """Comparison table (one row per model and bucket) from `evaluate_model` outputs."""
    lines = [
        "| model | bucket | RMSE | MAE | median rel. err |",
        "|---|---|---|---|---|",
    ]
    for model_name, by_bucket in results.items():
        for bucket in BUCKETS:
            if bucket in by_bucket:
                m = by_bucket[bucket]
                lines.append(
                    f"| {model_name} | {bucket} | {m['rmse']:.3e} | {m['mae']:.3e} "
                    f"| {m['median_rel_err']:.3e} |"
                )
    return "\n".join(lines) + "\n"


def write_results(results, path: str | Path = "outputs/results.md") -> Path:
    """Write the Markdown comparison table, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_markdown(results), encoding="utf-8")
    return path
