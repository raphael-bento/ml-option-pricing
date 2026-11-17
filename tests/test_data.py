import numpy as np

from option_pricing.black_scholes import bs_call_price
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


def test_splits_do_not_overlap():
    ds = make_dataset(100, 100, 100)
    assert not np.array_equal(ds.train.X, ds.val.X)
    assert not np.array_equal(ds.val.X, ds.test.X)
