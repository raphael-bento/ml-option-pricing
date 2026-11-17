import numpy as np
import pandas as pd
import pytest

from option_pricing.historical_vol import main, realized_vol, summarize


def _gbm(sigma, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, sigma / np.sqrt(252), n)
    return pd.Series(np.exp(np.cumsum(r)), index=pd.bdate_range("2020-01-01", periods=n))


def test_realized_vol_recovers_sigma():
    v = realized_vol(_gbm(0.3), 63)
    assert abs(v.mean() - 0.3) < 0.03
    assert len(v) == 2000 - 63


def test_constant_prices_give_zero_vol():
    close = pd.Series(np.ones(50))
    assert (realized_vol(close, 21) == 0).all()


def test_bad_window():
    with pytest.raises(ValueError):
        realized_vol(_gbm(0.2, 10), 1)


def test_too_short_series():
    with pytest.raises(ValueError, match="need more than"):
        realized_vol(_gbm(0.2, 21), 21)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_invalid_prices(bad):
    close = _gbm(0.2, 50)
    close.iloc[10] = bad
    with pytest.raises(ValueError, match="positive"):
        realized_vol(close, 21)


def test_main_creates_figure(tmp_path, monkeypatch, capsys):
    csv = tmp_path / "p.csv"
    s = _gbm(0.3, 200)
    pd.DataFrame({"date": s.index, "close": s.values}).to_csv(csv, index=False)
    out = tmp_path / "sub" / "fig.png"
    monkeypatch.setattr("sys.argv", ["hv", "--csv", str(csv), "--out", str(out)])
    main()
    assert out.exists() and out.stat().st_size > 0
    assert "21d" in capsys.readouterr().out


def test_main_missing_column(tmp_path, monkeypatch):
    csv = tmp_path / "p.csv"
    pd.DataFrame({"date": ["2020-01-01"], "px": [1.0]}).to_csv(csv, index=False)
    monkeypatch.setattr("sys.argv", ["hv", "--csv", str(csv), "--out", str(tmp_path / "f.png")])
    with pytest.raises(SystemExit, match="close"):
        main()


def test_summarize_share_in_domain():
    s = summarize(pd.Series([0.005, 0.2, 0.5, 1.5]))
    assert s["share_in_domain"] == 0.5
    assert s["n"] == 4
