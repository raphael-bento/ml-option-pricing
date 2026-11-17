"""Error and uncertainty heatmaps over (m, tau) with sigma and r fixed.

Two uncertainty estimates are mapped next to the true absolute error ``|mean - BS|``:
the deep-ensemble std and the MC-dropout std. The grid is analytic (no dataset needed for
the truth). Whether uncertainty grows toward the domain edges is checked quantitatively:
mean value in each 5% edge band (m low/high, tau low/high) vs the core, over several seeds,
plus Spearman(std, error). A ratio of 1 is the no-edge-effect reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from option_pricing.black_scholes import scaled_call_price
from option_pricing.data import BOUNDS, make_dataset
from option_pricing.mc_dropout import mc_dropout_predict
from option_pricing.uncertainty import train_ensemble

EDGE_FRACTION = 0.05  # width of each edge band, as a fraction of the axis range
REL_MIN_PRICE = 0.01  # relative error only where the true price (K = 1) is at least this
EDGES = ("m_low", "m_high", "tau_low", "tau_high")
KEYS = ("std", "err", "relerr")


def make_grid(
    sigma: float = 0.3, r: float = 0.06, n_m: int = 40, n_tau: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (m_axis, tau_axis, X) with X of shape (n_tau * n_m, 4), row-major in (tau, m)."""
    m = np.linspace(*BOUNDS["m"], n_m)
    tau = np.linspace(*BOUNDS["tau"], n_tau)
    mm, tt = np.meshgrid(m, tau)
    X = np.column_stack([mm.ravel(), tt.ravel(), np.full(mm.size, r), np.full(mm.size, sigma)])
    return m, tau, X


def edge_masks(X: np.ndarray, fraction: float = EDGE_FRACTION) -> dict[str, np.ndarray]:
    """Boolean masks for each of the four edges plus the ``core`` (outside every edge band).

    Corner points belong to two edge bands at once (e.g. m_low and tau_low), so they are counted
    in both bands; they are never in the core.
    """
    masks = {}
    for col, name in ((0, "m"), (1, "tau")):
        lo, hi = BOUNDS[name]
        w = fraction * (hi - lo)
        masks[f"{name}_low"] = X[:, col] < lo + w
        masks[f"{name}_high"] = X[:, col] > hi - w
    masks["core"] = ~np.any([masks[e] for e in EDGES], axis=0)
    if not masks["core"].any() or not all(masks[e].any() for e in EDGES):
        raise ValueError("grid too coarse: an edge band or the core is empty")
    return masks


def _nanmean(v: np.ndarray) -> float:
    """Mean ignoring NaN; NaN (no warning) if no valid point (e.g. relerr, price < 0.01)."""
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def edge_stats(
    X: np.ndarray, err: np.ndarray, std: np.ndarray, truth: np.ndarray | None = None
) -> dict[str, float]:
    """Per-edge / core ratios of std and error (and relative error if ``truth``), plus Spearman.

    A ratio of 1 means no edge effect. Edges are reported separately (m low/high, tau low/high).
    """
    masks = edge_masks(X)
    core = masks["core"]
    fields = {"std": std, "err": err}
    if truth is not None:
        # Undefined (NaN, ignored by the means below) where the price is ~0, so the ratio is
        # not dominated by a floor in the deep-OTM / short-maturity corner.
        fields["relerr"] = np.where(truth >= REL_MIN_PRICE, err / np.maximum(truth, 1e-12), np.nan)
    rho = float("nan")
    if np.ptp(std) > 0 and np.ptp(err) > 0:
        rho = float(spearmanr(std, err).statistic)
    out = {"spearman": rho}
    for key, v in fields.items():
        core_mean = _nanmean(v[core])
        out[f"{key}_core"] = core_mean
        for e in EDGES:
            out[f"{key}_ratio_{e}"] = _nanmean(v[masks[e]]) / core_mean
    return out


def plot_maps(
    m: np.ndarray, tau: np.ndarray, fields: dict[str, np.ndarray], title: str, path: Path
) -> None:
    """Save one heatmap per field (each reshaped to (len(tau), len(m))) side by side."""
    n = len(fields)
    plt.switch_backend("Agg")  # headless; only switched when actually plotting
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4), constrained_layout=True)
    for ax, (name, z) in zip(np.atleast_1d(axes), fields.items(), strict=True):
        im = ax.pcolormesh(m, tau, z.reshape(len(tau), len(m)), shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("m = S0/K")
        ax.set_ylabel("tau")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt(mean: float, std: float, spec: str) -> str:
    """``mean ± std`` or ``n/a`` when undefined."""
    return "n/a" if np.isnan(mean) else f"{mean:{spec}} ± {std:{spec}}"


def stats_markdown(stats: dict[str, dict[str, tuple[float, float]]]) -> str:
    """Markdown table of mean ± std (over seeds) of the edge / core ratios per method.

    1.00 is the no-edge-effect reference. ``relerr`` is error / true price (price >= REL_MIN_PRICE);
    a band with no such point (typically ``m_low`` at sigma=0.3: deep OTM, price < 0.01) is
    undefined and shown as "n/a".
    """
    lines = [
        "| method | metric | " + " | ".join(EDGES) + " | Spearman(std, err) |",
        "|---|---|" + "---|" * (len(EDGES) + 1),
    ]
    for name, s in stats.items():
        for k in KEYS:
            cells = " | ".join(_fmt(*s[f"{k}_ratio_{e}"], ".2f") for e in EDGES)
            rho = _fmt(*s["spearman"], ".3f") if k == "std" else ""
            lines.append(f"| {name} | {k} ratio | {cells} | {rho} |")
    return "\n".join(lines) + "\n"


def run(
    sigma: float = 0.3,
    r: float = 0.06,
    n_models: int = 5,
    passes: int = 100,
    seed: int = 42,
    grid: int = 40,
    out_dir: Path = Path("docs/figures"),
    n_seeds: int = 3,
    **train_kwargs,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Train for ``n_seeds`` seeds, map error / std, save figures (first seed only).

    Returns, per method, the (mean, sample std with ddof=1 over seeds; 0 if one seed) of every
    edge statistic.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")
    m, tau, X = make_grid(sigma, r, grid, grid)
    truth = scaled_call_price(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
    tag = f"sigma{sigma:g}_r{r:g}"
    runs: dict[str, list[dict[str, float]]] = {}
    for i in range(n_seeds):
        s = seed + 100 * i
        ds = make_dataset(seed=s)
        ens = train_ensemble(ds, n_models, seed=s, **train_kwargs)
        e_mean, e_std = ens.predict(X)
        # MC dropout on ensemble member 0 (same network): no redundant training.
        d_mean, d_std = mc_dropout_predict(ens.models[0], X, passes, s)
        for slug, name, mean, std in (
            ("ensemble", f"deep ensemble (N={n_models})", e_mean, e_std),
            ("mc_dropout", f"MC dropout (T={passes})", d_mean, d_std),
        ):
            err = np.abs(mean - truth)
            runs.setdefault(name, []).append(edge_stats(X, err, std, truth))
            if i == 0:
                plot_maps(
                    m,
                    tau,
                    {"|prediction - BS|": err, "uncertainty (std)": std},
                    f"{name}, sigma={sigma:g}, r={r:g}",
                    out_dir / f"uncertainty_map_{slug}_{tag}.png",
                )
    return {name: {k: _mean_std([x[k] for x in rs]) for k in rs[0]} for name, rs in runs.items()}


def _mean_std(values: list[float]) -> tuple[float, float]:
    """NaN-aware mean and sample std (ddof=1); NaN mean if all values are NaN."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m option_pricing.uncertainty_maps``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--r", type=float, default=0.06)
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--passes", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    p.add_argument("--md-dir", type=Path, default=Path("outputs"))
    a = p.parse_args(argv)
    res = run(
        a.sigma,
        a.r,
        a.n_models,
        a.passes,
        a.seed,
        a.grid,
        a.out_dir,
        a.n_seeds,
        epochs=a.epochs,
        lr=a.lr,
        optimizer=a.optimizer,
    )
    md = stats_markdown(res)
    a.md_dir.mkdir(parents=True, exist_ok=True)
    (a.md_dir / "uncertainty_maps.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
