import ast
import os
import pathlib

import pytest
import torch
from lightning import Fabric
from torch import nn

import sheeprl
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE


def test_every_model_is_logged_by_name_and_pickled():
    # MLflow 3 logs the models by `name` (`artifact_path` is deprecated) and by default serializes them as traced
    # programs, which need an input example: the agents take dictionaries of observations
    calls = []
    for path in pathlib.Path(sheeprl.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "log_model":
                calls.append((f"{path.name}:{node.lineno}", {k.arg: k.value for k in node.keywords}))
    assert len(calls) > 0
    for where, kwargs in calls:
        assert "artifact_path" not in kwargs and "name" in kwargs, where
        serialization_format = kwargs.get("serialization_format")
        assert isinstance(serialization_format, ast.Constant) and serialization_format.value == "pickle", where


@pytest.mark.skipif(not _IS_MLFLOW_AVAILABLE, reason="MLflow is not installed")
def test_the_model_manager_registers_the_best_model_and_downloads_it(tmp_path, monkeypatch):
    import mlflow

    from sheeprl.utils.mlflow import MlflowModelManager

    # The artifacts of the runs are stored in the working directory. MLflow sets the tracking URI and the experiment
    # in the environment: they are restored at the end of the test
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "0")
    manager = MlflowModelManager(Fabric(accelerator="cpu", devices=1), tracking_uri)
    mlflow.set_experiment("test_model_manager")
    for reward in (1.0, 3.0, 2.0):
        with mlflow.start_run():
            model = nn.Linear(2, 1)
            nn.init.constant_(model.weight, reward)
            mlflow.pytorch.log_model(model, name="agent", serialization_format="pickle")
            mlflow.log_metric("Test/cumulative_reward", reward)

    models_info = {"agent": {"path": "agent", "name": "test_agent", "tags": {"k": "v"}, "description": "The best"}}
    versions = manager.register_best_models("test_model_manager", models_info)
    assert versions is not None and str(versions["agent"].version) == "1"
    # A second registration of the same model: its description is added to the one of the first
    versions = manager.register_best_models("test_model_manager", models_info)
    assert str(versions["agent"].version) == "2"
    assert str(manager.get_latest_version("test_agent").version) == "2"
    description = manager.client.get_registered_model("test_agent").description
    assert description.startswith("# MODEL CHANGELOG\n## **Version 1**\n")
    assert description.count("# MODEL CHANGELOG") == 1 and "## **Version 2**\n" in description
    assert manager.transition_model("test_agent", 2, "Staging").current_stage == "Staging"

    manager.download_model("test_agent", 2, str(tmp_path / "download"))
    downloaded = torch.load(os.path.join(tmp_path, "download", "data", "model.pth"), weights_only=False)
    assert torch.equal(downloaded.weight, torch.full((1, 2), 3.0))
