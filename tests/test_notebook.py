"""The demo notebook must be valid, committed without outputs, and syntactically correct."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_valid_nbformat4():
    nb = _load()
    assert nb["nbformat"] == 4
    kinds = {c["cell_type"] for c in nb["cells"]}
    assert kinds == {"markdown", "code"}


def test_notebook_outputs_are_cleared():
    for cell in _load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_code_cells_compile():
    for i, cell in enumerate(_load()["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"demo.ipynb[cell {i}]", "exec")


def test_notebook_smoke_run(monkeypatch):
    """Execute every code cell with tiny sizes so API renames break this test."""
    import matplotlib

    matplotlib.use("Agg")
    for key, value in {
        "DEMO_EPOCHS": "2",
        "DEMO_N_TRAIN": "200",
        "DEMO_N_EVAL": "100",
        "DEMO_N_MODELS": "2",
        "DEMO_N_PASSES": "5",
    }.items():
        monkeypatch.setenv(key, value)
    namespace: dict = {"__name__": "__notebook__"}
    for i, cell in enumerate(_load()["cells"]):
        if cell["cell_type"] == "code":
            code = compile("".join(cell["source"]), f"demo.ipynb[cell {i}]", "exec")
            exec(code, namespace)
