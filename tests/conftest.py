import json
import pytest


# sample files

@pytest.fixture
def sample_csv(tmp_path):
    f = tmp_path / "sample.csv"
    f.write_text("col_a,col_b,col_c\n1,2,3\n4,5,6\n7,,9\n")
    return f


@pytest.fixture
def sample_csv_with_pipe_columns(tmp_path):
    # NSE market data uses | in column names — this is what broke
    # the original f-string SQL approach in get_dataset_stats
    f = tmp_path / "nse_data.csv"
    f.write_text("NSE_EQ|INE585B01010,NSE_EQ|INE467B01029\n100,200\n150,250\n")
    return f


@pytest.fixture
def sample_tsv(tmp_path):
    f = tmp_path / "sample.tsv"
    f.write_text("name\tscore\nalpha\t0.91\nbeta\t0.87\n")
    return f


@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("training loss: 0.02\nvalidation loss: 0.05\nepoch complete\n")
    return f


@pytest.fixture
def sample_py(tmp_path):
    f = tmp_path / "train.py"
    f.write_text("import torch\n\ndef train():\n    pass\n")
    return f


@pytest.fixture
def fake_model_pt(tmp_path):
    # not real weights, just needs the right extension for framework detection
    f = tmp_path / "model.pt"
    f.write_bytes(b"fake pytorch weights")
    return f


@pytest.fixture
def fake_model_onnx(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"fake onnx model")
    return f


@pytest.fixture
def fake_notebook(tmp_path):
    # minimal nbformat v4 — enough for get_notebook_summary to parse
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# EDA Notebook",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": "import pandas as pd",
                "outputs": [],
                "execution_count": 1,
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": "df = pd.read_csv('data.csv')",
                "outputs": [],
                "execution_count": 2,
            },
        ],
    }
    f = tmp_path / "analysis.ipynb"
    f.write_text(json.dumps(nb))
    return f


# directory layouts

@pytest.fixture
def ml_project_dir(tmp_path):
    # checkpoint_copy.pt is intentionally identical to checkpoint.pt
    # so duplicate detection tests have something to find
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.csv").write_text("x,y\n1,2\n3,4\n")
    (tmp_path / "data" / "val.csv").write_text("x,y\n5,6\n")

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "checkpoint.pt").write_bytes(b"weights")
    (tmp_path / "models" / "checkpoint_copy.pt").write_bytes(b"weights")

    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "eda.ipynb").write_text("{}")

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("# training script\n")
    (tmp_path / "src" / "utils.py").write_text("# utilities\n")

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("lr: 0.001\nbatch_size: 32\n")

    return tmp_path


@pytest.fixture
def duplicate_files_dir(tmp_path):
    content = b"identical content across files" * 5
    (tmp_path / "original.bin").write_bytes(content)
    (tmp_path / "duplicate.bin").write_bytes(content)
    (tmp_path / "unique.bin").write_bytes(b"completely different")
    return tmp_path


# router state — not autouse because most tests don't touch the router
# learned this the hard way when autouse was slowing down the whole suite

@pytest.fixture
def clean_router():
    import mlops_mcp.router as r

    def _clear():
        r._ACTIVE_MODULES.clear()
        r._ACTIVE_MODULE_TOOLS.clear()
        r._RUNTIME_REGISTERED_TOOLS.clear()
        r._APP = None

    _clear()
    yield
    _clear()  # cleanup even if test fails


@pytest.fixture
def mock_fastmcp_app():
    from unittest.mock import MagicMock
    app = MagicMock()
    app.tool = MagicMock(return_value=lambda fn: fn)
    app.name = "mlops-mcp-server"
    return app


# factory for when you need a few files fast and no fixture exists for them

@pytest.fixture
def make_files(tmp_path):
    def _make(spec: dict):
        out = {}
        for name, content in spec.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                p.write_bytes(content)
            else:
                p.write_text(content)
            out[name] = p
        return out
    return _make