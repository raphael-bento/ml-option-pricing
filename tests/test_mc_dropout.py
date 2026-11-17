import numpy as np
import pytest
import torch
from torch import nn

from option_pricing.data import make_dataset
from option_pricing.evaluate import predict
from option_pricing.mc_dropout import (
    _spearman,
    evaluate_mc,
    interval_coverage,
    main,
    mc_dropout_predict,
    mc_table_markdown,
    run,
)
from option_pricing.train import train


@pytest.fixture(scope="module")
def setup():
    ds = make_dataset(200, 50, 50)
    return ds, train(ds, epochs=2, lr=1e-2).model


def test_shapes_and_positive_std(setup):
    ds, model = setup
    mean, std = mc_dropout_predict(model, ds.val.X, n_passes=20)
    assert mean.shape == std.shape == (len(ds.val.X),)
    assert std.max() > 0


def test_reproducible_and_restores_mode(setup):
    ds, model = setup
    model.eval()
    a = mc_dropout_predict(model, ds.val.X, 10, seed=1)
    b = mc_dropout_predict(model, ds.val.X, 10, seed=1)
    np.testing.assert_allclose(a[0], b[0])
    assert not model.training
    assert not any(m.training for m in model.modules() if isinstance(m, nn.Dropout))


def test_no_dropout_gives_zero_std():
    model = nn.Sequential(nn.Linear(4, 1), nn.Flatten(0))
    _, std = mc_dropout_predict(model, np.ones((5, 4), dtype=np.float32), 5)
    assert (std == 0).all()


def test_invalid_passes(setup):
    for n in (0, 1):
        with pytest.raises(ValueError):
            mc_dropout_predict(setup[1], setup[0].val.X, n)


def test_different_seed_differs_and_train_mode_restored(setup):
    ds, model = setup
    model.train()
    a = mc_dropout_predict(model, ds.val.X, 10, seed=1)
    b = mc_dropout_predict(model, ds.val.X, 10, seed=2)
    assert model.training
    assert not np.allclose(a[1], b[1])


def test_only_dropout_in_train_mode():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.Linear(4, 1))
    seen = {}
    model[1].register_forward_hook(lambda m, i, o: seen.update(bn=m.training))
    model[2].register_forward_hook(lambda m, i, o: seen.update(do=m.training))
    mc_dropout_predict(model, np.ones((8, 4), dtype=np.float32), 2)
    assert seen == {"bn": False, "do": True}


def test_zero_dropout_mean_equals_deterministic():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.0), nn.Linear(8, 1), nn.Flatten(0)
    )
    X = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    mean, std = mc_dropout_predict(model, X, 3)
    np.testing.assert_allclose(mean, predict(model, X), atol=1e-6)
    assert (std == 0).all()


def test_interval_coverage():
    y = np.zeros(4)
    assert interval_coverage(y, np.array([0.1, 0.1, 0.1, 5.0]), np.full(4, 0.1)) == 0.75


def test_evaluate_and_markdown(setup):
    ds, model = setup
    res = evaluate_mc(model, ds.val, n_passes=10, calib=ds.val)
    assert res["rmse_mc"] > 0 and res["mean_std"] > 0
    # coverage is monotone in z and a constant-width baseline is reported
    assert res["coverage_0p5std"] <= res["coverage_1std"] <= res["coverage_1p96std"]
    assert res["width_1p96std"] > res["width_1std"] > res["width_0p5std"]
    assert 0.9 <= res["baseline_coverage"] <= 1 and res["baseline_half_width"] > 0
    buckets = {k for k in res if k.startswith("coverage_1p96std_")}
    assert buckets and buckets <= {f"coverage_1p96std_{n}" for n in ("OTM", "ATM", "ITM")}
    assert all(0 <= res[k] <= 1 for k in buckets)
    md = mc_table_markdown(res, 10)
    assert "| 10 |" in md and "constant width" in md and "in-sample" in md


def test_baseline_uses_calib_split(setup):
    ds, model = setup
    # calib with exact targets -> zero error -> zero half-width
    exact = type(ds.val)(ds.val.X, predict(model, ds.val.X))
    res = evaluate_mc(model, ds.test, n_passes=5, calib=exact)
    assert res["baseline_half_width"] == 0.0 and res["baseline_in_sample"] == 0.0
    assert "in-sample" not in mc_table_markdown(res, 5)


def test_calib_required(setup):
    ds, model = setup
    with pytest.raises(TypeError):
        evaluate_mc(model, ds.val, n_passes=5)


def test_spearman_value_and_nan():
    assert _spearman(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4])) == pytest.approx(1.0)
    assert _spearman(np.array([4.0, 3, 2, 1]), np.array([1.0, 2, 3, 4])) == pytest.approx(-1.0)
    assert np.isnan(_spearman(np.zeros(4), np.arange(4.0)))


def test_run_rejects_bad_split():
    with pytest.raises(ValueError):
        run(make_dataset(50, 20, 20), "train", epochs=1)


def test_main_writes_markdown(tmp_path):
    main(["--epochs", "1", "--passes", "3", "--out-dir", str(tmp_path)])
    assert "constant width" in (tmp_path / "mc_dropout_val.md").read_text(encoding="utf-8")


def test_global_rng_untouched(setup):
    ds, model = setup
    torch.manual_seed(0)
    expected = torch.rand(1)
    torch.manual_seed(0)
    mc_dropout_predict(model, ds.val.X, 3)
    assert torch.rand(1) == expected
