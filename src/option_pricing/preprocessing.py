"""Feature standardisation, fit on the training set only (no leakage into val/test)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_MIN_STD = 1e-12  # below this a column is treated as constant


class StandardScaler:
    """Per-column standardisation ``(x - mean) / std``.

    Call `fit` on the training split only, then reuse `transform` on val/test.
    The standard deviation uses ``ddof=0`` (same convention as scikit-learn).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        """Compute column means and standard deviations from ``x`` (shape (n, d))."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("Expected a non-empty 2D array.")
        if not np.isfinite(x).all():
            raise ValueError("Input contains NaN or infinite values.")
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.std_ = np.where(std > _MIN_STD, std, 1.0)  # constant column: avoid division by zero
        return self

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted.")

    def _as_array(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 0 or x.shape[-1] != self.mean_.shape[0]:
            raise ValueError(f"Expected last dimension {self.mean_.shape[0]}, got shape {x.shape}.")
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardise ``x`` with the statistics learned in `fit`."""
        return (self._as_array(x) - self.mean_) / self.std_

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Map standardised values back to the original scale."""
        return self._as_array(z) * self.std_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit on ``x`` then transform it."""
        return self.fit(x).transform(x)

    def save(self, path: str | Path) -> None:
        """Write the statistics to a JSON file."""
        self._check_fitted()
        payload = {"mean": self.mean_.tolist(), "std": self.std_.tolist()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> StandardScaler:
        """Load a scaler saved with `save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not {"mean", "std"} <= payload.keys():
            raise ValueError("Scaler file must contain 'mean' and 'std'.")
        mean = np.asarray(payload["mean"], dtype=float)
        std = np.asarray(payload["std"], dtype=float)
        if mean.ndim != 1 or mean.shape != std.shape or mean.size == 0:
            raise ValueError("'mean' and 'std' must be 1D arrays of the same length.")
        scaler = cls()
        scaler.mean_, scaler.std_ = mean, std
        return scaler
