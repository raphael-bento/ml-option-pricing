"""Synthetic dataset generation: uniform sampling of the BS domain, labelled with the true price."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from option_pricing.black_scholes import scaled_call_price

FEATURE_NAMES = ("m", "tau", "r", "sigma")
BOUNDS = {
    "m": (0.4, 1.6),
    "tau": (0.2, 1.1),
    "r": (0.02, 0.10),
    "sigma": (0.01, 1.00),
}


@dataclass(frozen=True)
class Split:
    """Features ``X`` of shape (n, 4) in `FEATURE_NAMES` order and target ``y`` of shape (n,)."""

    X: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class Dataset:
    """Train / validation / test splits, drawn once."""

    train: Split
    val: Split
    test: Split


def sample_inputs(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` points uniformly in the domain; columns follow `FEATURE_NAMES`."""
    lows = np.array([BOUNDS[f][0] for f in FEATURE_NAMES])
    highs = np.array([BOUNDS[f][1] for f in FEATURE_NAMES])
    return rng.uniform(lows, highs, size=(n, len(FEATURE_NAMES)))


def make_split(n: int, rng: np.random.Generator) -> Split:
    """Sample ``n`` points and label them with the exact scaled BS call price ``C / K``."""
    if n < 1:
        raise ValueError(f"Split size must be >= 1, got {n}.")
    X = sample_inputs(n, rng)
    y = scaled_call_price(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
    return Split(X=X, y=y)


def make_dataset(
    n_train: int = 5000, n_val: int = 1000, n_test: int = 1000, seed: int = 42
) -> Dataset:
    """Generate the train/val/test splits from a single seeded generator."""
    rng = np.random.default_rng(seed)
    return Dataset(
        train=make_split(n_train, rng),
        val=make_split(n_val, rng),
        test=make_split(n_test, rng),
    )
