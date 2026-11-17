"""Download TSMC daily prices with yfinance into ``data/`` (git-ignored).

Usage: pip install -e ".[market]" && python scripts/download_tsmc.py
Reproducible: fixed ticker and explicit start/end dates (end is exclusive).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_TICKER = "2330.TW"
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-01-01"


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with a sorted ``date`` index and a ``close`` column."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "close" not in df.columns:
        raise ValueError("downloaded data has no 'Close' column")
    df = df[["close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df[~df.index.duplicated()].sort_index()


def download(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit('yfinance missing: pip install -e ".[market]"') from exc
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise SystemExit(f"no data returned for {ticker}")
    return clean_prices(raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker", default=DEFAULT_TICKER)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--out", type=Path, default=Path("data/tsmc_daily.csv"))
    args = p.parse_args()
    df = download(args.ticker, args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"{len(df)} rows {df.index.min().date()} -> {df.index.max().date()} saved to {args.out}")


if __name__ == "__main__":
    main()
