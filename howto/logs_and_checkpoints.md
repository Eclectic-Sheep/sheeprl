# How to control logging and checkpointing in SheepRL

## Logging

By default the logging of metrics is enabled with the following settings:

```yaml
# ./sheeprl/configs/metric/default.yaml

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

* `log_every` is the number of policy steps (number of steps played in the environment, e.g. if one has 2 processes with 4 environment per process then the policy steps are 2*4=8) between two consecutive logging operations
* `disable_timer` is a boolean flag that enables/disables the timer to measure both the time spent in the environment and the time spent during the agent training. The timer class used can be found [here](/sheeprl/utils/timer.py)
* `log_level` is the level of logging: 0 means no log (it disables also the timer), 1 means log everything
* `sync_on_compute` is a boolean flag that enables/disables the synchronization of the metrics on compute
* `aggregator` is the aggregator of the metrics, `raise_on_missing` is a boolean flag that enables/disables the raising of an exception when a metric is missing, and `metrics` is a dictionary that contains the metrics to log. Every metric should be an instance of a class that inherits from `torchmetrics.Metric` (for more information, check [here](https://torchmetrics.readthedocs.io/en/stable/references/metric.html#torchmetrics.Metric)).

So, if one want to disable everything related to logging, he/she can set `log_level` to 0l if one wants to disable the timer, he/she can set `disable_timer` to `True`.

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
    "Params/exploration_amout",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}
```


