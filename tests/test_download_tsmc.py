import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "download_tsmc", Path(__file__).parents[1] / "scripts" / "download_tsmc.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_clean_prices_multiindex_and_dupes():
    idx = pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02"]).tz_localize("UTC")
    cols = pd.MultiIndex.from_tuples([("Close", "2330.TW"), ("Volume", "2330.TW")])
    raw = pd.DataFrame([[2.0, 1], [1.0, 1], [1.0, 1]], index=idx, columns=cols)
    out = mod.clean_prices(raw)
    assert list(out.columns) == ["close"]
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing
    assert len(out) == 2


def test_clean_prices_requires_close():
    with pytest.raises(ValueError):
        mod.clean_prices(pd.DataFrame({"Open": [1.0]}))
