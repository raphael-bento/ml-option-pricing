"""Rolling annualised realised volatility of a price series (TSMC) vs the synthetic sigma domain.

Usage: python -m option_pricing.historical_vol [--csv data/tsmc_daily.csv]
Run ``python scripts/download_tsmc.py`` first to create the CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WINDOWS = (21, 63)
SIGMA_MIN, SIGMA_MAX = 0.01, 1.00  # synthetic training domain


def realized_vol(close: pd.Series, window: int, trading_days: int = TRADING_DAYS) -> pd.Series:
    """Rolling std of daily log returns (ddof=1), annualised by sqrt(trading_days)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    if not (close > 0).all():  # also False for NaN
        raise ValueError("close must contain only positive, non-NaN prices")
    if len(close) <= window:
        raise ValueError(f"need more than {window} prices, got {len(close)}")
    log_ret = np.log(close).diff()
    vol = log_ret.rolling(window).std() * np.sqrt(trading_days)
    return vol.dropna().rename(f"vol_{window}d")


def summarize(vol: pd.Series, lo: float = SIGMA_MIN, hi: float = SIGMA_MAX) -> dict:
    """Distribution summary and share of observations inside [lo, hi]."""
    return {
        "n": int(len(vol)),
        "mean": float(vol.mean()),
        "std": float(vol.std()),
        "min": float(vol.min()),
        "q05": float(vol.quantile(0.05)),
        "median": float(vol.median()),
        "q95": float(vol.quantile(0.95)),
        "max": float(vol.max()),
        "share_in_domain": float(((vol >= lo) & (vol <= hi)).mean()),
    }


def plot_vol(vols: dict[int, pd.Series], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for w, v in vols.items():
        ax1.plot(v.index, v.values, lw=1, label=f"{w}d")
        ax2.hist(v.values, bins=50, alpha=0.5, label=f"{w}d")
    ax1.axhspan(SIGMA_MIN, SIGMA_MAX, color="grey", alpha=0.1)
    ax2.axvspan(SIGMA_MIN, SIGMA_MAX, color="grey", alpha=0.15, label="domain [0.01, 1.00]")
    ax1.set_title("TSMC rolling realised volatility (annualised)")
    ax1.set_ylabel("sigma")
    ax2.set_title("Distribution vs synthetic domain")
    ax2.set_xlabel("sigma")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("data/tsmc_daily.csv"))
    p.add_argument("--out", type=Path, default=Path("docs/figures/tsmc_realized_vol.png"))
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    missing = {"date", "close"} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.csv}: missing column(s) {sorted(missing)}")
    close = df.set_index(pd.to_datetime(df["date"]))["close"]
    vols = {w: realized_vol(close, w) for w in WINDOWS}
    table = pd.DataFrame({f"{w}d": summarize(v) for w, v in vols.items()}).T.round(4)
    print(table.to_string())
    plot_vol(vols, args.out)
    print(f"figure saved to {args.out}")


if __name__ == "__main__":
    main()
