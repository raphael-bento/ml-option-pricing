import json

import pytest
import torch

from option_pricing.data import make_dataset
from option_pricing.models import MODELS
from option_pricing.train import main, train


@pytest.fixture(scope="module")
def small_ds():
    return make_dataset(200, 50, 50)


@pytest.mark.parametrize("name", MODELS)
def test_training_loss_decreases(small_ds, name):
    r = train(small_ds, name, epochs=2, lr=1e-2)
    assert len(r.train_loss) == len(r.val_loss) == 2
    assert r.train_loss[-1] < r.train_loss[0]


def test_reproducible_with_same_seed(small_ds):
    a = train(small_ds, epochs=2, lr=1e-2, seed=3)
    b = train(small_ds, epochs=2, lr=1e-2, seed=3)
    assert a.train_loss == b.train_loss
    assert a.val_loss == b.val_loss


def test_best_model_matches_best_val_epoch(small_ds):
    r = train(small_ds, epochs=5, lr=1e-2)
    assert r.best_epoch == r.val_loss.index(min(r.val_loss))
    x = torch.as_tensor(small_ds.val.X, dtype=torch.float32)
    y = torch.as_tensor(small_ds.val.y, dtype=torch.float32)
    r.model.eval()
    with torch.no_grad():
        val = torch.nn.functional.mse_loss(r.model(x), y).item()
    assert val == pytest.approx(min(r.val_loss), rel=1e-5)


def test_cli_writes_outputs(tmp_path):
    main(["--model", "mlp", "--epochs", "1", "--out-dir", str(tmp_path)])
    assert (tmp_path / "mlp.pt").exists()
    assert len(json.loads((tmp_path / "mlp_history.json").read_text())["val_loss"]) == 1
