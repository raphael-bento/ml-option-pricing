import numpy as np
import pytest

from option_pricing.black_scholes import bs_call_price, scaled_call_price
from option_pricing.data import BOUNDS, FEATURE_NAMES, make_dataset


def test_split_sizes_default():
    ds = make_dataset()
    assert ds.train.X.shape == (5000, 4)
    assert ds.val.X.shape == (1000, 4)
    assert ds.test.X.shape == (1000, 4)
    assert ds.train.y.shape == (5000,)


def test_inputs_within_bounds():
    ds = make_dataset(500, 100, 100)
    for split in (ds.train, ds.val, ds.test):
        for i, name in enumerate(FEATURE_NAMES):
            lo, hi = BOUNDS[name]
            assert split.X[:, i].min() >= lo
            assert split.X[:, i].max() <= hi


def test_same_seed_is_reproducible():
    a, b = make_dataset(seed=7), make_dataset(seed=7)
    np.testing.assert_array_equal(a.train.X, b.train.X)
    np.testing.assert_array_equal(a.test.y, b.test.y)


def test_different_seed_differs():
    assert not np.array_equal(make_dataset(seed=1).train.X, make_dataset(seed=2).train.X)


def test_target_is_true_bs_price():
    ds = make_dataset(200, 50, 50)
    m, tau, r, sigma = ds.train.X.T
    np.testing.assert_allclose(ds.train.y, bs_call_price(m, 1.0, tau, r, sigma))


def test_splits_share_no_rows():
    ds = make_dataset(100, 100, 100)
    rows = [set(map(tuple, s.X)) for s in (ds.train, ds.val, ds.test)]
    assert not rows[0] & rows[1]
    assert not rows[0] & rows[2]
    assert not rows[1] & rows[2]


def test_sampling_covers_domain():
    X = make_dataset(5000, 10, 10).train.X
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = BOUNDS[name]
        width = hi - lo
        assert X[:, i].min() < lo + 0.02 * width
        assert X[:, i].max() > hi - 0.02 * width
        assert abs(X[:, i].mean() - (lo + hi) / 2) < 0.03 * width


@pytest.mark.parametrize("n", [0, -5])
def test_invalid_size_raises(n):
    with pytest.raises(ValueError):
        make_dataset(n_train=n)


def test_target_reference_value_and_no_arbitrage_bounds():
    assert scaled_call_price(1.0, 1.0, 0.05, 0.2) == pytest.approx(0.10451, abs=1e-4)
    ds = make_dataset(1000, 10, 10)
    m, tau, r, _ = ds.train.X.T
    lower = np.maximum(m - np.exp(-r * tau), 0.0)
    assert np.all(ds.train.y >= lower - 1e-12)
    assert np.all(ds.train.y <= m + 1e-12)
