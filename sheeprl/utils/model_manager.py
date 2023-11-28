"""Thank you @lorenzomammana: https://github.com/orobix/quadra/blob/main/quadra/utils/model_manager.py"""

from __future__ import annotations

import getpass
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Literal, Set

from git import Sequence
from lightning import Fabric

try:
    import mlflow  # noqa
    from mlflow.entities import Run  # noqa
    from mlflow.entities.model_registry import ModelVersion  # noqa
    from mlflow.exceptions import RestException  # noqa
    from mlflow.tracking import MlflowClient  # noqa

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


VERSION_MD_TEMPLATE = "## **Version {}**\n"
DESCRIPTION_MD_TEMPLATE = "### Description: \n{}\n"


class AbstractModelManager(ABC):
    """Abstract class for model managers."""

    @abstractmethod
    def __init__(self, fabric: Fabric) -> None:
        self.fabric = fabric

    @abstractmethod
    def register_model(
        self, model_location: str, model_name: str, description: str, tags: Dict[str, Any] | None = None
    ) -> Any:
        """Register a model in the model registry."""

    @abstractmethod
    def get_latest_version(self, model_name: str) -> Any:
        """Get the latest version of a model for all the possible stages or filtered by stage."""

    @abstractmethod
    def transition_model(self, model_name: str, version: int, stage: str, description: str | None = None) -> Any:
        """Transition the model with the given version to a new stage."""

    @abstractmethod
    def delete_model(self, model_name: str, version: int, description: str | None = None) -> None:
        """Delete a model with the given version."""

    @abstractmethod
    def register_best_models(
        self,
        experiment_name: str,
        models_info: Dict[str, Dict[str, Any]],
        metric: str = "Test/cumulative_reward",
        mode: Literal["max", "min"] = "max",
    ) -> Any:
        """Register the best models from an experiment."""

    @abstractmethod
    def download_model(self, model_name: str, version: int, output_path: str) -> None:
        """Download the model with the given version to the given output path."""


class MlflowModelManager(AbstractModelManager):
    """Model manager for Mlflow."""

    def __init__(self, fabric: Fabric, tracking_uri: str):
        if not MLFLOW_AVAILABLE:
            raise ImportError("Mlflow is not available, please install it with pip install mlflow.")

        super().__init__(fabric)
        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self, model_location: str, model_name: str, description: str | None = None, tags: Dict[str, Any] | None = None
    ) -> ModelVersion:
        """Register a model in the model registry.

        Args:
            model_location (str): The model uri.
            model_name (str): The name of the model after it is registered.
            description (str, optional): A description of the model, this will be added to the model changelog.
                Default to None.
            tags (Dict[str, Any], optional): A dictionary of tags to add to the model.
                Default to None.

        Returns:
            The model version.
        """
        model_version = mlflow.register_model(model_uri=model_location, name=model_name, tags=tags)
        self.fabric.print(f"Registered model {model_name} with version {model_version.version}")
        registered_model_description = self.client.get_registered_model(model_name).description

        if model_version.version == "1":
            header = "# MODEL CHANGELOG\n"
        else:
            header = ""

        new_model_description = VERSION_MD_TEMPLATE.format(model_version.version)
        new_model_description += self._get_author_and_date()
        new_model_description += self._generate_description(description)

        self.client.update_registered_model(model_name, header + registered_model_description + new_model_description)

        self.client.update_model_version(
            model_name, model_version.version, "# MODEL CHANGELOG\n" + new_model_description
        )

        return model_version

    def get_latest_version(self, model_name: str) -> ModelVersion:
        """Get the latest version of a model.

        Args:
            model_name (str): The name of the model.

        Returns:
            The model version.
        """
        latest_version = max(int(x.version) for x in self.client.get_latest_versions(model_name))
        model_version = self.client.get_model_version(model_name, latest_version)

        return model_version

    def transition_model(
        self, model_name: str, version: int, stage: str, description: str | None = None
    ) -> ModelVersion | None:
        """Transition a model to a new stage.

        Args:
            model_name (str): The name of the model.
            version (int): The version of the model
            stage (str): The stage of the model.
            description (str, optional): A description of the transition, this will be added to the model changelog.
                Default to None.
        """
        previous_stage = self._safe_get_stage(model_name, version)

        if previous_stage is None:
            return None

        if previous_stage.lower() == stage.lower():
            warnings.warn(f"Model {model_name} version {version} is already in stage {stage}")
            return self.client.get_model_version(model_name, version)

        self.fabric.print(f"Transitioning model {model_name} version {version} from {previous_stage} to {stage}")
        model_version = self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
        new_stage = model_version.current_stage
        registered_model_description = self.client.get_registered_model(model_name).description
        single_model_description = self.client.get_model_version(model_name, version).description

        new_model_description = "## **Transition:**\n"
        new_model_description += f"### Version {model_version.version} from {previous_stage} to {new_stage}\n"
        new_model_description += self._get_author_and_date()
        new_model_description += self._generate_description(description)

        self.client.update_registered_model(model_name, registered_model_description + new_model_description)
        self.client.update_model_version(
            model_name, model_version.version, single_model_description + new_model_description
        )

        return model_version

    def delete_model(self, model_name: str, version: int, description: str | None = None) -> None:
        """Delete a model.

        Args:
            model_name (str): The name of the model,
            version (int): The version of the model.
            description (str, optional): Why the model was deleted, this will be added to the model changelog.
                Default to None.
        """
        model_stage = self._safe_get_stage(model_name, version)

        if model_stage is None:
            return

        if (
            input(
                f"Model named `{model_name}`, version {version} is in stage {model_stage}, "
                "type the model name to continue deletion:"
            )
            != model_name
        ):
            warnings.warn("Model name did not match, aborting deletion")
            return

        self.fabric.print(f"Deleting model {model_name} version {version}")
        self.client.delete_model_version(model_name, version)

        registered_model_description = self.client.get_registered_model(model_name).description

        new_model_description = "## **Deletion:**\n"
        new_model_description += f"### Version {version} from stage: {model_stage}\n"
        new_model_description += self._get_author_and_date()
        new_model_description += self._generate_description(description)

        self.client.update_registered_model(model_name, registered_model_description + new_model_description)

    def register_best_models(
        self,
        experiment_name: str,
        models_info: Dict[str, Dict[str, Any]],
        metric: str = "Test/cumulative_reward",
        mode: Literal["max", "min"] = "max",
    ) -> Dict[str, ModelVersion] | None:
        """Register the best model from an experiment.

        Args:
            experiment_name (str): The name of the experiment.
            models_info (Dict[str, Dict[str, Any]]): A dictionary containing models information
                (path, description and tags).
            metric (str): The metric to use to determine the best model.
                Default to "Test/cumulative_reward".
            mode (Literal["max", "min"]): The mode to use to determine the best model, either "max" or "min".
                Defaulto to "max".

        Returns:
            The registered models version if successful, otherwise None.
        """
        if mode not in ["max", "min"]:
            raise ValueError(f"Mode must be either 'max' or 'min', got {mode}")

        experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
        runs = self.client.search_runs(experiment_ids=[experiment_id])

        if len(runs) == 0:
            self.fabric.print(f"No runs found for experiment {experiment_name}")
            return None

        best_run: Run | None = None
        best_run_artifacts: Sequence[str] | Set[str] | None = None
        models_path = [v["path"] for v in models_info.values()]
        for run in runs:
            run_artifacts = [x.path for x in self.client.list_artifacts(run.info.run_id) if x.path in models_path]

            if len(run_artifacts) == 0 or run.data.metrics.get(metric) is None:
                # If we don't find the given model path, skip this run
                # If the run has not the target metric, skip this run
                continue

            if best_run is None:
                best_run = run
                best_run_artifacts = set(run_artifacts)
                continue
            if mode == "max":
                if run.data.metrics[metric] > best_run.data.metrics[metric]:
                    best_run = run
            else:
                if run.data.metrics[metric] < best_run.data.metrics[metric]:
                    best_run = run

        if best_run is None:
            self.fabric.print(f"No runs found for experiment {experiment_name} with the given metric")
            return None

        models_version = {}
        for k, v in models_info.items():
            if v["path"] in best_run_artifacts:
                best_model_uri = f"runs:/{best_run.info.run_id}/{v['path']}"
                models_version[k] = self.register_model(
                    model_location=best_model_uri, model_name=v["name"], tags=v["tags"], description=v["description"]
                )

        return models_version

    def download_model(self, model_name: str, version: int, output_path: str) -> None:
        """Download the model with the given version to the given output path.

        Args:
            model_name (str): The name of the model.
            version (int): The version of the model.
            output_path (str): The path to save the model to.
        """
        artifact_uri = self.client.get_model_version_download_uri(model_name, version)
        self.fabric.print(f"Downloading model {model_name} version {version} from {artifact_uri} to {output_path}")
        if not os.path.exists(output_path):
            self.fabric.print(f"Creating output path {output_path}")
            os.makedirs(output_path)
        mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=output_path)

    @staticmethod
    def _generate_description(description: str | None = None) -> str:
        """Generate the description markdown template."""
        if description is None:
            return ""

        return DESCRIPTION_MD_TEMPLATE.format(description)

    @staticmethod
    def _get_author_and_date() -> str:
        """Get the author and date markdown template."""
        author_and_date = f"### Author: {getpass.getuser()}\n"
        author_and_date += f"### Date: {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S %Z')}\n"

        return author_and_date

    def _safe_get_stage(self, model_name: str, version: int) -> str | None:
        """Get the stage of a model version.

        Args:
            model_name (str): The name of the model.
            version (int): The version of the model

        Returns:
            The stage of the model version if it exists, otherwise None.
        """
        try:
            model_stage = self.client.get_model_version(model_name, version).current_stage
            return model_stage
        except RestException:
            self.fabric.print(f"Model named {model_name} with version {version} does not exist")
            return None
