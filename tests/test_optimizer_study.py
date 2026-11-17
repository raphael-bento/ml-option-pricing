import numpy as np
import pytest

from option_pricing.data import make_dataset
from option_pricing.optimizer_study import CONFIGS, main, run_study, table_markdown
from option_pricing.train import OPTIMIZERS, train


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError):
        train(make_dataset(50, 20, 20), epochs=1, optimizer="rmsprop")


def test_optimizers_registry():
    assert OPTIMIZERS["adam"].__name__ == "Adam"
    assert OPTIMIZERS["sgd"].__name__ == "SGD"


def test_sgd_and_adam_differ():
    ds = make_dataset(100, 20, 20)
    sgd = train(ds, epochs=2, seed=0, optimizer="sgd")
    adam = train(ds, epochs=2, seed=0, optimizer="adam")
    assert sgd.train_loss != adam.train_loss


def test_run_study_table(tmp_path):
    _, table = run_study(epochs=2)
    assert len(table) == len(CONFIGS) + 1
    for name, m in table.items():
        assert all(np.isfinite(v) for v in m.values()), name
    for name in CONFIGS:
        assert {"best_epoch", "final_train_mse"} <= table[name].keys()
    assert table["Baseline (train mean)"]["rmse"] > 0
    md = table_markdown(table)
    assert "Adam" in md and "Baseline" in md
    main(["--epochs", "1", "--out-dir", str(tmp_path)])
    written = (tmp_path / "optimizer_study.md").read_text(encoding="utf-8")
    assert "Baseline" in written and "| config |" in written
    assert (tmp_path / "optimizer_study.png").exists()
