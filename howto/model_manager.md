# Model Manager

SheepRL makes it possible to register trained models on MLFLow, so as to keep track of model versions and stages.

It requires MLflow 3 (`pip install sheeprl[mlflow]` installs `mlflow>=3.11,<4`). The models are logged with `mlflow.pytorch.log_model` in the `pickle` format: the whole module is saved, as with `torch.save`.

## Register models with training
The configurations of the model manager are placed in the `./sheeprl/configs/model_manager/` folder, and the default configuration is defined as follows:
```yaml
# ./sheeprl/configs/model_manager/default.yaml

disabled: True
models: {}
```
Since the algorithms have different models, then the `models` parameter is set to an empty python dictionary, and each agent will define its own configuration. The `disabled` parameter indicates whether or not the user wants to register the agent when the training is finished (`False` means that the agent will be registered, otherwise not).

> [!NOTE]
>
> To register the models at the end of the training, the logger must be the MLFlow one (`logger@metric.logger=mlflow`, see the [logs and checkpoints howto](./logs_and_checkpoints.md#mlflow)), otherwise an error is raised. Instead, the models can be [registered from a checkpoint](#register-models-from-checkpoints) even if the logger used during the training is Tensorboard: the only requirement is that an instance of the MLFlow server is running and is accessible, moreover, it is necessary to specify its URI in the `MLFLOW_TRACKING_URI` environment variable.

To better understand how to define the configurations of the models you want to register, take a look at the DreamerV3 model manager configuration:
```yaml
# ./sheeprl/configs/model_manager/dreamer_v3.yaml

defaults:
  - default
  - _self_

models: 
  world_model:
    model_name: "${exp_name}_world_model"
    description: "DreamerV3 World Model used in ${env.id} Environment"
    tags: {}
  actor:
    model_name: "${exp_name}_actor"
    description: "DreamerV3 Actor used in ${env.id} Environment"
    tags: {}
  critic:
    model_name: "${exp_name}_critic"
    description: "DreamerV3 Critic used in ${env.id} Environment"
    tags: {}
  target_critic:
    model_name: "${exp_name}_target_critic"
    description: "DreamerV3 Target Critic used in ${env.id} Environment"
    tags: {}
  moments:
    model_name: "${exp_name}_moments"
    description: "DreamerV3 Moments used in ${env.id} Environment"
    tags: {}
```
For each model, it is necessary to define the `model_name`, the `description`, and the `tags` (i.e., a python dictionary with strings as keys and values). The keys that can be specified are defined by the `MODELS_TO_REGISTER` variable in the `./sheeprl/algos/<algo_name>/utils.py`. For DreamerV3, it is defined as follows: `MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}`.
If you do not want to log some models, then, you just need to remove it from the configuration file.

> [!NOTE]
>
> The name of the models in the `MODELS_TO_REGISTER` variable is equal to the name of the models in the training state of the algorithm, i.e. the fields of its `TrainState` subclass in the `./sheeprl/algos/<algo_name>/<algo_name>.py` file (e.g. the `world_model`, `actor`, `critic`, `target_critic` and `moments` fields of `DreamerV3State`). The only exception are the exploration critics of P2E-DV3, which are registered as `critic_exploration_<critic>`, `target_critic_exploration_<critic>` and `moments_exploration_<critic>`, for every `<critic>` in `algo.critics_exploration`.
>
> Make sure that the models specified in the configuration file are a subset of the models defined by the `MODELS_TO_REGISTER` variable.

## Register models from checkpoints
Another possibility is to register the models after the training, by manually selecting the checkpoint where to retrieve the agent. To do this, it is possible to run the `sheeprl_model_manager.py` script (or directly `sheeprl-registration`) by properly specifying the `checkpoint_path`, the `model_manager`, and the MLFlow-related configurations.
The default configurations are defined in the `./sheeprl/configs/model_manager_config.yaml` file, that is reported below:

```yaml
# ./sheeprl/configs/model_manager_config.yaml

# @package _global_
defaults:
  - _self_
  - model_manager: ???
  - override hydra/hydra_logging: disabled
  - override hydra/job_logging: disabled

hydra:
  output_subdir: null
  run:
    dir: .

checkpoint_path: ???
run:
  id: null
  name: ${now:%Y-%m-%d_%H-%M-%S}_${exp_name}
experiment:
  id: null
  name: ${exp_name}_${now:%Y-%m-%d_%H-%M-%S}
tracking_uri: ${oc.env:MLFLOW_TRACKING_URI}
```

As before, it is necessary to specify the `model_manager` configurations (the models we want to register with names, descriptions, and tags). Moreover, it is mandatory to set the `checkpoint_path`, which must be the path to the `ckpt` file created during the training. The models are built with the configuration of the training (the `config.yaml` file next to the `checkpoint` folder) and their weights are loaded from the checkpoint (`sheeprl.core.load_trained_state`). Finally, the `run` and `experiment` parameters contain the MLFlow configurations:
* If you set the `run.id` to a value different from `null`, then all the other parameters are ignored, indeed, the models will be logged and registered under the run with the specified ID.
* If you want to create a new run (with a name equal to `run.name`) and put it into an existing experiment, then you have to set `run.id=null` and `experiment.id=<experiment_id>`.
* If you set `experiment.id=null` and `run.id=null`, then a new experiment and a new run are created with the specified names.

> [!NOTE]
>
> Also, in this case, the models specified in the `model_manager` configuration must be a subset of the `MODELS_TO_REGISTER` variable.

For instance, you can register the DreamerV3 models from a checkpoint with the following command:

```bash
python sheeprl_model_manager.py model_manager=dreamer_v3 checkpoint_path=/path/to/checkpoint.ckpt
```

if you have installed SheepRL from a cloned repo, or

```bash
sheeprl-registration model_manager=dreamer_v3 checkpoint_path=/path/to/checkpoint.ckpt
```

if you have installed SheepRL from PyPi.

## Delete, Transition and Download Models
The MLFlow model manager enables the deletion of the registered models, moving them from one stage to another or downloading them.
`download_model` saves the files of a version of a registered model in the given folder: the module is in `data/model.pth`, and it can be loaded with `torch.load("<folder>/data/model.pth", weights_only=False)`.
[This notebook](../examples/model_manager.ipynb) contains a tutorial on how to use the MLFlow model manager. We recommend taking a look to see what APIs the model manager makes available.
