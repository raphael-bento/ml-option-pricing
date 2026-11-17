import math

import numpy as np
import pytest

from option_pricing.uncertainty_maps import (
    EDGES,
    edge_masks,
    edge_stats,
    main,
    make_grid,
    stats_markdown,
)


def test_grid_shape_and_fixed_params():
    m, tau, X = make_grid(0.3, 0.05, 6, 4)
    assert X.shape == (24, 4)
    assert len(m) == 6 and len(tau) == 4
    assert (X[:, 2] == 0.05).all() and (X[:, 3] == 0.3).all()


def test_edge_masks_partition_and_core():
    _, _, X = make_grid(n_m=20, n_tau=20)
    masks = edge_masks(X)
    assert masks["m_low"][0] and masks["m_high"][19] and masks["tau_low"][0]
    assert masks["core"][len(X) // 2 + 10]
    assert not masks["core"][0] and 0 < masks["core"].mean() < 1


def test_edge_masks_empty_core_raises():
    _, _, X = make_grid(n_m=2, n_tau=2)
    with pytest.raises(ValueError, match="too coarse"):
        edge_masks(X)


def test_edge_stats_per_edge_ratios_independent_error():
    _, _, X = make_grid(n_m=30, n_tau=30)
    masks = edge_masks(X)
    std = np.where(masks["m_high"], 2.0, 1.0)  # uncertainty grows only at m high
    err = np.where(masks["m_low"], 3.0, 1.0)  # error grows only at m low
    s = edge_stats(X, err, std, truth=np.ones(len(X)))
    assert s["std_ratio_m_high"] == 2.0 and s["std_ratio_m_low"] == 1.0
    assert s["err_ratio_m_low"] == 3.0 and s["err_ratio_m_high"] == 1.0
    assert s["relerr_ratio_m_low"] == 3.0


def test_edge_stats_spearman_nan_when_std_constant():
    _, _, X = make_grid(n_m=20, n_tau=20)
    err = np.linspace(0, 1, len(X))
    s = edge_stats(X, err, np.full(len(X), 0.5))
    assert math.isnan(s["spearman"])


def test_stats_markdown_content():
    cell = {f"{k}_ratio_{e}": (1.5, 0.1) for k in ("std", "err", "relerr") for e in EDGES}
    md = stats_markdown({"method A": cell | {"spearman": (0.5, 0.05)}})
    assert "method A" in md and "1.50 ± 0.10" in md and "0.500 ± 0.050" in md
    assert md.count("\n") == 2 + 3  # header + separator + one row per metric


def test_cli_writes_figures_and_markdown(tmp_path):
    argv = ["--n-models", "2", "--passes", "5", "--epochs", "1", "--grid", "30", "--n-seeds", "2"]
    main([*argv, "--out-dir", str(tmp_path / "figs"), "--md-dir", str(tmp_path / "md")])
    assert len(list((tmp_path / "figs").glob("*.png"))) == 2
    md = (tmp_path / "md" / "uncertainty_maps.md").read_text(encoding="utf-8")
    assert "deep ensemble" in md and "MC dropout" in md and "m_low" in md


def test_run_rejects_zero_seeds():
    import pytest

    from option_pricing.uncertainty_maps import run

    with pytest.raises(ValueError):
        run(n_seeds=0)


def test_relerr_ignores_near_zero_prices():
    import numpy as np

    from option_pricing.uncertainty_maps import edge_stats, make_grid

    _, _, X = make_grid(n_m=20, n_tau=20)
    truth = np.ones(len(X))
    truth[X[:, 0] < 0.6] = 0.0  # price ~0 on the low-m side: relerr must stay finite
    err = np.full(len(X), 0.01)
    s = edge_stats(X, err, err, truth)
    assert np.isfinite(s["relerr_core"])
    assert abs(s["relerr_core"] - 0.01) < 1e-12
