import numpy as np
import pytest
import torch
from torch import nn

from option_pricing.data import make_dataset
from option_pricing.evaluate import predict
from option_pricing.uncertainty import (
    Ensemble,
    compare,
    table_markdown,
    train_ensemble,
    uncertainty_metrics,
)


@pytest.fixture(scope="module")
def setup():
    ds = make_dataset(200, 50, 50)
    return ds, train_ensemble(ds, n_models=3, epochs=2, lr=1e-2)


def test_ensemble_shapes_and_std(setup):
    ds, ens = setup
    mean, std = ens.predict(ds.val.X)
    assert mean.shape == std.shape == (len(ds.val.X),)
    assert (std >= 0).all() and std.max() > 0  # different seeds -> different members
    preds = np.stack([predict(m, ds.val.X) for m in ens.models])
    np.testing.assert_allclose(mean, preds.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(std, preds.std(axis=0), rtol=1e-5)


class _Const(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0],), self.value)


def test_known_mean_std():
    X = np.zeros((4, 4), dtype=np.float32)
    mean, std = Ensemble([_Const(0.0), _Const(2.0)]).predict(X)
    np.testing.assert_allclose(mean, 1.0)
    np.testing.assert_allclose(std, 1.0)


def test_uncertainty_metrics_values():
    y = np.zeros(4)
    mean = np.array([0.1, 0.2, 0.3, 5.0])
    std = np.array([0.1, 0.2, 0.3, 0.4])
    m = uncertainty_metrics(y, mean, std)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["coverage_2std"] == pytest.approx(0.75)
    assert m["mean_std"] == pytest.approx(0.25)


@pytest.mark.parametrize("constant", ["std", "err"])
def test_uncertainty_metrics_nan_spearman(constant, recwarn):
    y = np.zeros(4)
    if constant == "std":
        mean, std = np.array([0.1, 0.2, 0.3, 0.4]), np.zeros(4)
    else:
        mean, std = np.full(4, 0.5), np.array([0.1, 0.2, 0.3, 0.4])
    m = uncertainty_metrics(y, mean, std)
    assert np.isnan(m["spearman"])
    assert np.isfinite(m["mean_std"]) and np.isfinite(m["coverage_2std"])
    assert len(recwarn) == 0
    row = {"rmse": 1.0, "mae": 1.0, "median_rel_err": 1.0} | m
    assert "nan" in table_markdown({"ens": row})


def test_uncertainty_metrics_non_finite():
    m = uncertainty_metrics(np.zeros(3), np.array([0.1, np.nan, 0.3]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(m["spearman"])


def test_reproducible(setup):
    ds, ens = setup
    again = train_ensemble(ds, n_models=3, epochs=2, lr=1e-2)
    np.testing.assert_allclose(ens.predict(ds.val.X)[0], again.predict(ds.val.X)[0], rtol=1e-5)


def test_single_member_has_zero_std(setup):
    ds, _ = setup
    _, std = train_ensemble(ds, n_models=1, epochs=1, lr=1e-2).predict(ds.val.X)
    assert (std == 0).all()


def test_invalid_n_models(setup):
    with pytest.raises(ValueError):
        train_ensemble(setup[0], n_models=0)


def test_compare_table(setup):
    ds, ens = setup
    table = compare(ens, ds.val)
    assert "ensemble (N=3)" in table and "single model (mean of members)" in table
    md = table_markdown(table)
    assert "| ensemble (N=3) |" in md and "mean std" in md and "Spearman" in md
    assert "mean_std" in table["ensemble (N=3)"]
