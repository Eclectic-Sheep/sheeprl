# Register an external algorithm

Suppose that we have installed SheepRL through pip with `pip install sheeprl[box2d,atari,dev,test]` and we want to add a new (external) SoTA algorithm called `ext_sota` without directly adding the new algorithm to the SheepRL codebase, i.e. without the need to clone the repo locally.

We can start by creating two new folders called `my_awesome_algo` and `my_awesome_configs`, the former will contain the implementation of the algorithm, the latter the configs needed to run the experiment and configure our new algorithm.

An external algorithm is written exactly as an algorithm of SheepRL: read [the how-to on how to register a new algorithm](./register_new_algorithm.md) first, it explains the training loop of `sheeprl.core` and the methods an algorithm implements. Here we write the same actor-critic, with these files under the `my_awesome_algo` folder:

```bash
my_awesome_algo
├── __init__.py
├── agent.py
├── evaluate.py
├── loss.py
├── ext_sota.py
└── utils.py
```

## The agent, the loss functions and the utils

`agent.py` and `loss.py` are the ones of [the how-to on how to register a new algorithm](./register_new_algorithm.md#the-agent). `utils.py` defines the metrics the algorithm can log (`AGGREGATOR_KEYS`), the models it can register (`MODELS_TO_REGISTER`) and the functions used by the training: here they come from PPO.

```python
# my_awesome_algo/utils.py
from __future__ import annotations

# The observations, the test episode and the model logging of PPO fit this algorithm too
from sheeprl.algos.ppo.utils import log_models, normalize_obs, prepare_obs, test  # noqa: F401

# The metrics the algorithm can log and the models it can register
AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/policy_loss", "Loss/value_loss"}
MODELS_TO_REGISTER = {"agent"}
```

> [!NOTE]
>
> The registration of the models of a checkpoint with `sheeprl-registration` (see [the how-to on the model manager](./model_manager.md)) looks for the algorithms in the `sheeprl.algos` package only. An external algorithm can still register its models at the end of the training, with `log_models`, as shown below.

## Algorithm implementation

The algorithm is implemented in the `ext_sota.py` file: the training state, the writer, the subclass of `Algorithm` and the entrypoint decorated with `register_algorithm`. It is the `sota.py` file of [the how-to on how to register a new algorithm](./register_new_algorithm.md#algorithm-implementation), with the modules of `my_awesome_algo` in place of the ones of `sheeprl.algos.sota`:

```python
"""An actor-critic written on the shared training loop of SheepRL."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from torch import Tensor
from torch.optim import Optimizer

from my_awesome_algo.agent import build_agent
from my_awesome_algo.loss import policy_loss, value_loss
from my_awesome_algo.utils import normalize_obs, prepare_obs, test
from sheeprl.algos.ppo.agent import PPOAgent, PPOPolicy
from sheeprl.core import (
    Act,
    Algorithm,
    EnvStep,
    ReplayStore,
    TrainSchedule,
    TrainState,
    Writer,
    autocast,
    rollout_store,
    run,
    setup_module,
    update,
)
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae


@dataclass
class ExtSOTAState(TrainState):
    # Saved in the checkpoints as `agent` and `optimizer`
    agent: PPOAgent
    optimizer: Optimizer


class RolloutWriter(Writer):
    """Writes every step in the rollout, with the columns of the actions of `PPOPolicy.act`."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    def write(self, rollout: ReplayStore, step: EnvStep, act: Act) -> None:
        num_envs = len(step.rewards)
        dones = np.logical_or(step.terminated, step.truncated)
        data = {k: step.obs[k][np.newaxis] for k in self.obs_keys}
        # The environments played the indices of the discrete actions, the rollout stores them one-hot
        data["actions"] = act.columns["actions"][np.newaxis]
        data["values"] = act.columns["values"][np.newaxis]
        data["rewards"] = step.rewards.reshape(1, num_envs, 1).astype(np.float32)
        data["dones"] = dones.reshape(1, num_envs, 1).astype(np.uint8)
        rollout.add(data)
        # The observations after the last step of the rollout bootstrap its returns
        rollout.last_step, rollout.last_act = step, act


class ExtSOTA(Algorithm):
    """Every iteration plays `algo.rollout_steps` steps, then trains for `algo.update_epochs` epochs of minibatches of
    the rollout."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        self.steps_per_iteration = cfg.algo.rollout_steps

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[ExtSOTAState, ReplayStore]:
        cfg = self.cfg
        agent = build_agent(cfg, obs_space, action_space)
        # On the device, in the precision of the run, with the same initial weights on every process
        agent.feature_extractor = setup_module(self.fabric, agent.feature_extractor)
        agent.actor = setup_module(self.fabric, agent.actor)
        agent.critic = setup_module(self.fabric, agent.critic)
        optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")
        optimizer = self.fabric.setup_optimizers(optimizer)

        # The steps of a rollout, in a ReplayBuffer, and the EpochSampler of the minibatches of its update
        rollout = rollout_store(self.fabric, cfg, log_dir, cfg.algo.rollout_steps)
        return ExtSOTAState(agent=agent, optimizer=optimizer), rollout

    def policy(self, state: ExtSOTAState) -> PPOPolicy:
        """The policy to play with: it shares its modules, and so its weights, with the trained agent."""
        cfg = self.cfg
        return PPOPolicy(
            state.agent.feature_extractor,
            state.agent.actor,
            state.agent.critic,
            fabric=self.fabric,
            obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
            cnn_keys=cfg.algo.cnn_keys.encoder,
        )

    def writer(self, state: ExtSOTAState, policy: PPOPolicy) -> RolloutWriter:
        return RolloutWriter(self.cfg)

    def batches(
        self, state: ExtSOTAState, rollout: ReplayStore, n_steps: Optional[int], iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
        data = rollout.read()

        # The returns and the advantages, bootstrapped with the value of the observations after the rollout
        with torch.inference_mode():
            next_obs = {k: rollout.last_step.next_obs[k] for k in obs_keys}
            next_obs = prepare_obs(self.fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
            next_values = state.agent.critic(state.agent.feature_extractor(next_obs))
            returns, advantages = gae(
                data["rewards"],
                data["values"],
                data["dones"],
                next_values,
                cfg.algo.rollout_steps,
                cfg.algo.gamma,
                cfg.algo.gae_lambda,
            )
            data["returns"] = returns.float()
            data["advantages"] = advantages.float()

        # [Rollout_Steps, Num_Envs, ...] -> [Rollout_Steps * Num_Envs, ...]
        data = {k: v.flatten(start_dim=0, end_dim=1).float() for k, v in data.items()}
        # The minibatches of the epochs, shuffled at every epoch
        yield from rollout.minibatches(data, cfg.algo.update_epochs)

    def train_step(self, state: ExtSOTAState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
        cfg = self.cfg.algo
        obs = normalize_obs(batch, cfg.cnn_keys.encoder, cfg.cnn_keys.encoder + cfg.mlp_keys.encoder)
        with autocast(self.fabric):
            actions = torch.split(batch["actions"], state.agent.actions_dim, dim=-1)
            _, logprobs, _, values = state.agent(obs, actions)
            pg_loss = policy_loss(logprobs, batch["advantages"])
            v_loss = value_loss(values, batch["returns"])
            loss = pg_loss + cfg.vf_coef * v_loss
        update(self.fabric, loss, state.optimizer, max_grad_norm=cfg.max_grad_norm)
        return {"Loss/policy_loss": pg_loss.detach(), "Loss/value_loss": v_loss.detach()}


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    algo = ExtSOTA(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        # The return of the trained agent, logged at the last policy step of the training
        test(algo.policy(state), fabric, cfg, log_dir, policy_step=policy_step)

    # Optional: register the trained models with MLflow
    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from my_awesome_algo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
```

## Evaluation

The `evaluate.py` file registers the evaluation of a checkpoint of the algorithm, as in [the how-to on how to register a new algorithm](./register_new_algorithm.md#evaluation):

```python
# my_awesome_algo/evaluate.py
from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from my_awesome_algo.ext_sota import ExtSOTA
from my_awesome_algo.utils import test
from sheeprl.core import load_trained_state
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="ext_sota")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)

    # The models are built as by the training (`build`) and restored from the checkpoint
    env = make_env(cfg, cfg.seed, 0, log_dir, "test", vector_env_idx=0)()
    algo = ExtSOTA(fabric, cfg)
    trained = load_trained_state(fabric, cfg, algo, state, env.observation_space, env.action_space)
    env.close()
    test(algo.policy(trained), fabric, cfg, log_dir)
```

## Config files

Once you have written your algorithm, you need to create three config files: one in `./my_awesome_configs/algo`, one in `./my_awesome_configs/exp` and the other one in `./my_awesome_configs/model_manager`.


```tree
.
├── my_awesome_algo
|   ├── __init__.py
│   ├── agent.py
│   ├── evaluate.py
│   ├── loss.py
│   ├── ext_sota.py
│   └── utils.py
└── my_awesome_configs
    ├── algo
    │   └── ext_sota.yaml
    ├── exp
    │   └── ext_sota.yaml
    └── model_manager
        └── ext_sota.yaml
    
```

#### Algo Configs

In the `./my_awesome_configs/algo/ext_sota.yaml` we need to specify all the configs needed to initialize and train your agent.
Here is an example of the `./my_awesome_configs/algo/ext_sota.yaml` config file:

```yaml
defaults:
  - default
  - /optim@optimizer: adam
  - _self_

# Must be equal to the name of the file with the implementation: `my_awesome_algo/ext_sota.py`
name: ext_sota

# Training recipe
gamma: 0.99
gae_lambda: 0.95
rollout_steps: 128
update_epochs: 1
vf_coef: 0.5
max_grad_norm: 0.5

# Agent
dense_units: 64
mlp_layers: 2
dense_act: torch.nn.Tanh
layer_norm: False
encoder:
  cnn_features_dim: 512
  mlp_features_dim: 64
  dense_units: ${algo.dense_units}
  mlp_layers: ${algo.mlp_layers}
  dense_act: ${algo.dense_act}
  layer_norm: ${algo.layer_norm}
actor:
  dense_units: ${algo.dense_units}
  mlp_layers: ${algo.mlp_layers}
  dense_act: ${algo.dense_act}
  layer_norm: ${algo.layer_norm}
critic:
  dense_units: ${algo.dense_units}
  mlp_layers: ${algo.mlp_layers}
  dense_act: ${algo.dense_act}
  layer_norm: ${algo.layer_norm}

# Override the parameters of `optim/adam.yaml`
optimizer:
  lr: 3e-4
  eps: 1e-4
```

> [!NOTE]
>
> With `/optim@optimizer: adam` under `defaults` you specify that your agent has one adam optimizer and you can access to its config with `algo.optimizer`.
>

If you need more than one optimizer, you can add more elements to `defaults`, for instance:
```yaml
defaults:
  - /optim@encoder.optimizer: adam
  - /optim@actor.optimizer: adam
```
will add two optimizers, one accessible with `algo.encoder.optimizer`, the other with `algo.actor.optimizer`.

> [!NOTE]
>
> The field `algo.name` **must** be set and **must** be equal to the name of the file.py, found under the `my_awesome_algo` folder, where the implementation of the algorithm is defined. For example, if your implementation is defined in a python file named `my_sota.py`, i.e. `my_awesome_algo/my_sota.py`, then `algo.name="my_sota"` 

#### Model Manager Configs

In the `./my_awesome_configs/model_manager/ext_sota.yaml` we need to specify all the configs needed to register your agent. You can specify a name, a description, and some tags for each model you want to register. The `disabled` parameter indicates whether or not you want to register your models.
Here is an example of the `./my_awesome_configs/model_manager/ext_sota.yaml` config file:

```yaml
defaults:
  - default
  - _self_

disabled: True
models:
  agent:
    model_name: "${exp_name}"
    description: "SOTA Agent in ${env.id} Environment"
    tags: {}
```

#### Experiment Configs

In the experiment config, you have to specify all the elements you want in your experiment and you can override all the parameters you want.
Here is an example of the `./my_awesome_configs/exp/ext_sota.yaml` config file:

```yaml
# @package _global_

defaults:
  - override /algo: ext_sota
  - override /env: gym
  # select the model manager configs
  - override /model_manager: ext_sota
  - _self_

algo:
  total_steps: 65536
  per_rank_batch_size: 64
  mlp_keys:
    encoder: [state]

env:
  id: CartPole-v1

# The buffer holds one rollout
buffer:
  size: ${algo.rollout_steps}
  memmap: False

# select which metrics to log
metric:
  aggregator:
    metrics:
      Loss/policy_loss:
        _target_: torchmetrics.MeanMetric
        sync_on_compute: ${metric.sync_on_compute}
      Loss/value_loss:
        _target_: torchmetrics.MeanMetric
        sync_on_compute: ${metric.sync_on_compute}
```

With `override /algo: ext_sota` in `defaults` you are specifying you want to use the new `ext_sota` algorithm, whereas, with `override /env: gym` you are specifying that you want to train your agent on a *Gymnasium* environment, `CartPole-v1`.

## Register the algorithm and the configs

To let the `register_algorithm` decorator add our new `ext_sota` algorithm to the available algorithms registry we need first to create a new file called for example `my_awesome_main.py` in the root of the project:

```tree
.
├── my_awesome_algo
|   ├── __init__.py
│   ├── agent.py
│   ├── evaluate.py
│   ├── loss.py
│   ├── ext_sota.py
│   └── utils.py
├── my_awesome_configs
|   ├── algo
|   |   └── ext_sota.yaml
|   ├── exp
|   |   └── ext_sota.yaml
|   └── model_manager
|       └── ext_sota.yaml
├── my_awesome_eval.py
└── my_awesome_main.py
```

containing the following:

```python
# my_awesome_main.py

# Importing the algorithm registers it in SheepRL
from my_awesome_algo import ext_sota  # noqa: F401

if __name__ == "__main__":
    # Imported after the registration, so that SheepRL finds the algorithm named by `algo.name`
    from sheeprl.cli import run

    run()
```

To evaluate its checkpoints, the file `my_awesome_eval.py` registers the evaluation too:

```python
# my_awesome_eval.py

# Importing the algorithm and its evaluation registers them in SheepRL
from my_awesome_algo import evaluate, ext_sota  # noqa: F401

if __name__ == "__main__":
    from sheeprl.cli import evaluation

    evaluation()
```

While to let SheepRL know about the new configs we need to add a new file called `.env` to the root of the project containing the following env variable:

```bash
SHEEPRL_SEARCH_PATH=file://my_awesome_configs;pkg://sheeprl.configs
```

This tells SheepRL to search for configs in the `my_awesome_configs` folder and in the `configs` folder under the installed `sheeprl` package. The `.env` file is read from the directory the scripts are run from; the variable can also be exported in the shell instead.

## Run the experiment

Then you can run your experiment with `python my_awesome_main.py exp=ext_sota`, and evaluate one of its checkpoints with `python my_awesome_eval.py checkpoint_path=/path/to/checkpoint.ckpt`.
