# How to control logging and checkpointing in SheepRL

## Logging

By default the logging of metrics is enabled with the following settings:

```yaml
# ./sheeprl/configs/metric/default.yaml

defaults:
  - _self_
  - /logger@logger: tensorboard

log_every: 5000
disable_timer: False

# Level of Logging:
#   0: No log
#   1: Log everything
log_level: 1

# Metric related parameters. Please have a look at
# https://torchmetrics.readthedocs.io/en/stable/references/metric.html#torchmetrics.Metric
# for more information
sync_on_compute: False

aggregator:
  _target_: sheeprl.utils.metric.MetricAggregator
  raise_on_missing: False
  metrics:
    Rewards/rew_avg: 
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
    Game/ep_len_avg: 
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
```
where 

* `logger` is the configuration of the logger you want to use for logging. There are two possible values: `tensorboard` (default) and `mlflow`, but one can define and choose its own logger.
* `log_every` is the number of policy steps (number of steps played in the environment, e.g. if one has 2 processes with 4 environments per process then the policy steps are 2*4=8) between two consecutive logging operations. For more info about the policy steps, check the [Work with Steps Tutorial](./work_with_steps.md).
* `disable_timer` is a boolean flag that enables/disables the timer to measure both the time spent in the environment and the time spent during the agent training. The timer class used can be found [here](../sheeprl/utils/timer.py).
* `log_level` is the level of logging: $0$ means no log (it disables also the timer), whereas $1$ means logging everything.
* `sync_on_compute` is a boolean flag that enables/disables the synchronization of the metrics on compute.
* `aggregator` is the aggregator of the metrics, `raise_on_missing` is a boolean flag that enables/disables the raising of an exception when a metric to be logged is missing, and `metrics` is a dictionary that contains the metrics to log. Every metric should be an instance of a class that inherits from `torchmetrics.Metric` (for more information, check [here](https://torchmetrics.readthedocs.io/en/stable/references/metric.html#torchmetrics.Metric)).

So, if one wants to disable everything related to logging, he/she can set `log_level` to $0$ if one wants to disable the timer, he/she can set `disable_timer` to `True`.

### Loggers
Two loggers are made available: the Tensorboard logger and the MLFlow one. In any case, it is possible to define or choose another logger.
The configurations of the loggers are under the `./sheeprl/configs/logger/` folder.

#### Tensorboard
Let us start with the Tensorboard logger, which is the default logger used in SheepRL.

```yaml
# ./sheeprl/configs/logger/tensorboard.yaml

# For more information, check https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.loggers.TensorBoardLogger.html
_target_: lightning.fabric.loggers.TensorBoardLogger
name: ${run_name}
root_dir: logs/runs/${root_dir}
version: null
default_hp_metric: True
prefix: ""
sub_dir: null
```
As shown in the configurations, it is necessary to specify the `_target_` class to instantiate. For the Tensorboard logger, it is necessary to specify the `name` and the `root_dir` arguments equal to the `run_name` and `logs/runs/<root_dir>` parameters, respectively, because we want that all the logs and files (configs, checkpoint, videos, ...) are under the same folder for a specific experiment.

> [!NOTE]
>
> In general we want the path of the logs files to be in the same folder created by Hydra when the experiment is launched, so make sure to properly define the `root_dir` and `name` parameters of the logger so that it is within the folder created by hydra (defined by the `hydra.run.dir` parameter). The tensorboard logger will save the logs in the `<root_dir>/<name>/<version>/<sub_dir>/` folder (if `sub_dir` is defined, otherwise in the `<root_dir>/<name>/<version>/` folder).

The documentation of the TensorboardLogger class can be found [here](https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.loggers.TensorBoardLogger.html).

#### MLFlow
Another possibility provided by SheepRL is [MLFlow](https://mlflow.org/docs/2.8.0/index.html).

```yaml
# ./sheeprl/configs/logger/mlflow.yaml

# For more information, check https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html#lightning.pytorch.loggers.mlflow.MLFlowLogger
_target_: lightning.pytorch.loggers.MLFlowLogger
experiment_name: ${exp_name}
tracking_uri: ${oc.env:MLFLOW_TRACKING_URI}
run_name: ${algo.name}_${env.id}_${now:%Y-%m-%d_%H-%M-%S}
tags: null
save_dir: null
prefix: ""
artifact_location: null
run_id: null
log_model: false
```

The parameters that can be specified for creating the MLFlow logger are explained [here](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html#lightning.pytorch.loggers.mlflow.MLFlowLogger).

You can specify the MLFlow logger instead of the Tensorboard one in the CLI, by adding the `logger@metric.logger=mlflow` argument. In this way, hydra will take the configurations defined in the `./sheeprl/configs/logger/mlflow.yaml` file.

```bash
python sheeprl.py exp=ppo exp_name=ppo-cartpole logger@metric.logger=mlflow
```

> [!NOTE]
>
> If you are using an MLFlow server, you can specify the `tracking_uri` in the config file or with the `MLFLOW_TRACKING_URI` environment variable (that is the default value in the configs).

### Logged metrics

Every algorithm should specify a set of default metrics to log, called `AGGREGATOR_KEYS`, under its own `utils.py` file. For instance, the default metrics logged by DreamerV2 are the following:

```python
AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/post_entropy",
    "State/prior_entropy",
    "State/kl",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
```

These keys refer to the metrics that will be updated in the code (i.e., `aggregator.update(key, value)`). Moreover, these keys will be used as filters for the metrics specified in the `metric.log.aggregator.metrics` config. In particular, only the metrics present in both the `metric.log.aggregator.metrics` and the `AGGREGATOR_KEYS` will be logged.

For example, let us suppose we have defined the following metrics in the config file:
```yaml
aggregator:
  _target_: sheeprl.utils.metric.MetricAggregator
  raise_on_missing: False
  metrics:
    key0:
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
    key1:
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
    key2:
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
    key3:
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
    key4:
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
```

And the `AGGREGATOR_KEYS` set of the algorithm is defined as follows:
```python
AGGREGATOR_KEYS = {"key0", "key2", "key5"}
```

Then, the metrics that will be logged are the `key0` and the `key2`. The `key5` is not logged because it is not in the configs; instead, the `key1`, `key3`, and `key4` are not logged because they are not in the `AGGREGATOR_KEYS` set of the algorithm.

## Checkpointing

By default the checkpointing is enabled with the following settings:

```yaml
every: 100
resume_from: null
save_last: True
keep_last: 5
```

meaning that:

* `every` is the number of policy steps (number of steps played in the environment, e.g. if one has 2 processes with 4 environments per process then the policy steps are 2*4=8) between two consecutive checkpointing operations. For more info about the policy steps, check the [Work with Steps Tutorial](./work_with_steps.md).
* `resume_from` is the path of the checkpoint to resume from. If `null`, then the checkpointing is not resumed.
* `save_last` is a boolean flag that enables/disables the saving of the last checkpoint.
* `keep_last` is the number of checkpoints you want to keep during the experiment. If `null`, all the checkpoints are kept.

> [!NOTE]
>
> When restarting an experiment from a specific checkpoint (`resume_from=/path/to/checkpoint.ckpt`), it is **mandatory** to pass as arguments the same configurations of the experiment you want to restart. This is due to the way Hydra creates the folder in which it saves configs: if you do not pass the same configurations, you may have an unexpected log directory (i.e., the folder is created in the wrong folder).