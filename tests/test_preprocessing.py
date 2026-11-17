import numpy as np
import pytest

from option_pricing.preprocessing import StandardScaler


def _x(n, seed=0):
    return np.random.default_rng(seed).uniform(0.0, 5.0, size=(n, 4))


def test_train_is_zero_mean_unit_std():
    z = StandardScaler().fit_transform(_x(500))
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), 1, rtol=1e-12)


def test_inverse_transform_roundtrip():
    x = _x(200)
    s = StandardScaler().fit(x)
    np.testing.assert_allclose(s.inverse_transform(s.transform(x)), x)


def test_statistics_come_from_train_only():
    train, val = _x(300, 1), _x(50, 2) + 100.0  # val far from train
    s = StandardScaler().fit(train)
    mean, std = s.mean_.copy(), s.std_.copy()
    np.testing.assert_allclose(mean, train.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(std, train.std(axis=0), rtol=1e-12)
    s.transform(val)  # must not mutate state
    np.testing.assert_array_equal(s.mean_, mean)
    np.testing.assert_array_equal(s.std_, std)
    assert s.transform(val).mean() > 10  # val is not re-centred


def test_save_load_roundtrip(tmp_path):
    x = _x(100)
    s = StandardScaler().fit(x)
    s.save(tmp_path / "scaler.json")
    loaded = StandardScaler.load(tmp_path / "scaler.json")
    np.testing.assert_array_equal(loaded.transform(x), s.transform(x))


@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_unfitted_raises(method):
    with pytest.raises(RuntimeError):
        getattr(StandardScaler(), method)(np.zeros((2, 4)))


def test_unfitted_save_raises(tmp_path):
    with pytest.raises(RuntimeError):
        StandardScaler().save(tmp_path / "s.json")


def test_constant_column_is_zero_and_other_standardised():
    z = StandardScaler().fit_transform(np.array([[1.0, 2.0], [1.0, 3.0]]))
    np.testing.assert_array_equal(z[:, 0], 0.0)
    np.testing.assert_allclose(z[:, 1], [-1.0, 1.0])


def test_near_constant_column_does_not_explode():
    x = np.column_stack([np.full(100, 0.1) + 1e-17 * np.arange(100), _x(100)[:, 0]])
    assert np.abs(StandardScaler().fit_transform(x)).max() < 10


@pytest.mark.parametrize("bad", [np.array([[1.0, np.nan]]), np.array([[np.inf, 1.0]])])
def test_fit_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        StandardScaler().fit(bad)


@pytest.mark.parametrize("bad", [np.zeros((0, 4)), np.zeros(5)])
def test_fit_rejects_empty_or_1d(bad):
    with pytest.raises(ValueError):
        StandardScaler().fit(bad)


def test_wrong_column_count_raises():
    s = StandardScaler().fit(_x(10))
    with pytest.raises(ValueError):
        s.transform(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        s.inverse_transform(np.zeros((3, 5)))


def test_fit_transform_equals_fit_then_transform():
    x = _x(50)
    np.testing.assert_array_equal(
        StandardScaler().fit_transform(x), StandardScaler().fit(x).transform(x)
    )


def test_save_creates_parent_dir(tmp_path):
    path = tmp_path / "a" / "b" / "scaler.json"
    StandardScaler().fit(_x(10)).save(path)
    assert path.exists()


@pytest.mark.parametrize(
    "payload", ['{"mean": [0.0]}', '{"mean": [0.0, 1.0], "std": [1.0]}', "[1, 2]"]
)
def test_load_rejects_invalid_file(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        StandardScaler.load(path)
