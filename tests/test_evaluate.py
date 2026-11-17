import numpy as np
import pytest
import torch
from torch import nn

from option_pricing.data import Split, make_dataset
from option_pricing.evaluate import (
    evaluate_model,
    metrics,
    moneyness_masks,
    results_markdown,
    write_results,
)
from option_pricing.models import MODELS, build_model


class _Oracle(nn.Module):
    """Returns the exact BS price (used to check that a perfect model has zero error)."""

    def forward(self, x):
        from option_pricing import scaled_call_price

        a = x.numpy().astype(float)
        return torch.as_tensor(scaled_call_price(a[:, 0], a[:, 1], a[:, 2], a[:, 3]))


def test_metrics_known_values():
    out = metrics(np.array([1.0, 2.0, 4.0]), np.array([1.0, 3.0, 2.0]))
    assert out["rmse"] == pytest.approx(np.sqrt((0 + 1 + 4) / 3))
    assert out["mae"] == pytest.approx(1.0)
    assert out["median_rel_err"] == pytest.approx(0.5)  # errors 0, 0.5, 0.5


def test_metrics_zero_true_price_is_finite():
    assert np.isfinite(metrics(np.array([0.0, 1.0]), np.array([1e-3, 1.0]))["median_rel_err"])


def test_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics(np.zeros(3), np.zeros(2))


def test_buckets_partition_inputs():
    m = np.linspace(0.4, 1.6, 1000)
    masks = moneyness_masks(m)
    assert np.all(sum(v.astype(int) for v in masks.values()) == 1)
    assert masks["OTM"][0] and masks["ITM"][-1]


def test_oracle_model_has_near_zero_error():
    ds = make_dataset(10, 10, 300)
    res = evaluate_model(_Oracle(), ds.test)
    assert set(res) == {"all", "OTM", "ATM", "ITM"}
    assert res["all"]["rmse"] < 1e-6  # float32 round-off only


def test_evaluate_model_is_deterministic_despite_dropout():
    ds = make_dataset(10, 10, 100)
    model = build_model("mlp").train()  # eval() must be forced by evaluate_model
    assert evaluate_model(model, ds.test) == evaluate_model(model, ds.test)


def test_empty_bucket_is_omitted():
    split = Split(X=np.array([[1.5, 0.5, 0.05, 0.2]] * 3), y=np.array([0.5, 0.5, 0.5]))
    assert set(evaluate_model(build_model("mlp"), split)) == {"all", "ITM"}


def test_results_markdown_and_file(tmp_path):
    ds = make_dataset(10, 10, 100)
    results = {n: evaluate_model(build_model(n), ds.test) for n in MODELS}
    md = results_markdown(results)
    for n in MODELS:
        assert f"| {n} | all |" in md
    path = write_results(results, tmp_path / "sub" / "results.md")
    assert path.read_text(encoding="utf-8") == md
