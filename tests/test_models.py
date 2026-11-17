import pytest
import torch

from option_pricing.models import MODELS, build_model

H = 50
# Parameter counts: Linear(a, b) has a*b + b parameters.
L1, L2, OUT = 4 * H + H, H * H + H, H + 1
EXPECTED = {"mlp": L1 + L2 + OUT, "residual": L1 + L2 + OUT, "highway": L1 + 2 * L2 + OUT}


@pytest.mark.parametrize("name", MODELS)
def test_output_shape(name):
    assert build_model(name)(torch.randn(8, 4)).shape == (8,)


@pytest.mark.parametrize("name", MODELS)
def test_parameter_count(name):
    assert sum(p.numel() for p in build_model(name).parameters()) == EXPECTED[name]


@pytest.mark.parametrize("name", MODELS)
def test_dropout_off_in_eval_on_in_train(name):
    torch.manual_seed(0)
    model, x = build_model(name), torch.randn(16, 4)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("nope")
