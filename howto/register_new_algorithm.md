# Register a new algorithm
Suppose that we want to add a new SoTA algorithm to sheeprl called `sota` so that we can train an agent simply with `python sheeprl.py exp=sota env=... env.id=...`.

Every algorithm in sheeprl is written on the same training loop, `sheeprl.core.loop.run`: the algorithm says how to build its models, how its policy chooses the actions, how to store what happens in the environments and how to train; the loop does the rest (the environments, the iterations, the logging, the checkpoints, resuming a run, the processes of a distributed run).

We start by creating a new folder called `sota` under `./sheeprl/algos/`, containing the following files:

```bash
algos
└── droq
...
└── sota
    ├── __init__.py
    ├── agent.py     # the models
    ├── evaluate.py  # the evaluation of a checkpoint (`sheeprl-eval`)
    ├── loss.py      # the loss functions
    ├── sota.py      # the algorithm and its entrypoint
    └── utils.py     # the metrics and the models it can log, the test episode, the model registration
```

## The training loop
`run(fabric, cfg, algo)` calls the methods of the algorithm `algo` in this order:

```python
state, store = algo.build(observation_space, action_space, schedule, log_dir)
# When resuming: `state.load_state_dict(checkpoint)` and, for off-policy algorithms, the replay buffer of the checkpoint
policy = algo.policy(state)
collector = Collector(env, policy, algo.writer(state, policy), store, schedule, cadence, algo.random_warmup)
# Reset the environments and create the memory of the policy (`policy.init_states(env.num_envs)`)
collector.reset()
for iteration in schedule.iterations():
    # Play `algo.steps_per_iteration` steps in every environment. At every step the collector:
    # - chooses the actions: `act = policy.act(env.obs)`, or `policy.random(env)` before `algo.learning_starts`;
    # - steps the environments: `step = env.step(act.env_actions)`;
    # - writes the step in the store: `writer.write(store, step, act)`;
    # - resets the memory of the policy for the environments that start a new episode: `policy.reset_state(env_idxes)`;
    # - counts the policy steps and logs the episodes ended in `step`
    collector.collect(algo.steps_per_iteration)
    # Train: `n_steps` is None for on-policy algorithms, the gradient steps asked by `algo.replay_ratio` for
    # off-policy ones (0 before `algo.learning_starts`: no training then)
    n_steps = schedule.gradient_steps(iteration)
    if n_steps != 0:
        for batch in algo.batches(state, store, n_steps, iteration):
            metrics = algo.train_step(state, batch, gradient_step)
    algo.end_iteration(state, iteration)
    # Log the metrics every `metric.log_every` policy steps, save a checkpoint every `checkpoint.every` policy steps
```

- `env` is an `Environment` (`sheeprl/core/environment.py`): the `cfg.env.num_envs` vectorized environments of the process. `env.obs` holds the current observations, `env.step(actions)` steps the environments and returns an `EnvStep` (the observations the actions were chosen from, the next observations, the rewards, `terminated`, `truncated`, `final_obs` with the last observation of the episodes that have just ended, `restarted` with the environments created again after a crash, `episodes` with the episodes that have just ended, and the `info` of the environments, which the core and the writers don't read), and `env.random_actions()` samples random actions. The training loop creates a `GymEnvironment`: gymnasium environments created with `make_env` and seeded differently on every process. The collector records the rewards and the lengths of the episodes ended in every `EnvStep` (`episodes`; `Rewards/rew_avg`, `Game/ep_len_avg`).
- `schedule` is a `TrainSchedule` (`sheeprl/core/schedule.py`): the policy steps played so far (`schedule.policy_step`), the number of iterations (`algo.total_steps` policy steps), the random actions before `algo.learning_starts` and the gradient steps of the off-policy algorithms (`algo.replay_ratio`), and where a resumed run starts.
- The checkpoints hold the training state returned by `build`, the counters needed to resume the run and, for off-policy algorithms with `buffer.checkpoint=True`, the replay buffer.

## The agent
The agent is defined in the `agent.py` file. It is made of plain `torch.nn.Module`s: the algorithm creates them in `build` and sets them up with Fabric there. The models that play in the environments are the same modules of the trained agent (not copies): they always play with the latest weights.

```python
from __future__ import annotations

from typing import Any, Dict, Sequence

import gymnasium
import numpy as np
import torch
from torch import Tensor

from sheeprl.core import Act, Policy


class SOTAAgent(torch.nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, obs: Dict[str, Tensor], actions: Sequence[Tensor] | None = None) -> ...:
        ...


class SOTAPolicy(torch.nn.Module, Policy):
    """The policy that plays in the environments: it is built from the modules of the agent."""

    def __init__(self, ...):
        ...

    def forward(self, obs: Dict[str, Tensor]) -> ...:
        """The actions sampled from the observations as tensors: pure torch, so it can be compiled or exported."""
        ...

    def act(self, obs: Dict[str, np.ndarray], greedy: bool = False) -> Act:
        """The actions to play for the observations of the environments (`Policy`), with the columns the writer stores
        (e.g. their log-probabilities); the most likely ones with `greedy` (the test episode)."""
        ...

    def get_actions(self, obs: Dict[str, Tensor], greedy: bool = False) -> Sequence[Tensor]:
        ...


def build_agent(cfg: Dict[str, Any], obs_space: gymnasium.spaces.Dict, action_space: gymnasium.Space) -> SOTAAgent:
    """The agent with its initial weights, on the CPU: the algorithm sets it up with Fabric."""
    return SOTAAgent(...)
```

In this guide `sota` is an actor-critic that uses the agent of PPO (`sheeprl/algos/ppo/agent.py`, a good example of an agent that works with both image and vector observations and with every action space):

```python
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym

from sheeprl.algos.ppo.agent import PPOAgent


def actions_dim_of(action_space: gym.Space) -> Tuple[Sequence[int], bool]:
    """The dimensions of the actions (one per discrete action) and whether they are continuous."""
    is_continuous = isinstance(action_space, gym.spaces.Box)
    if is_continuous:
        return tuple(action_space.shape), True
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return tuple(action_space.nvec.tolist()), False
    return (action_space.n,), False


def build_agent(cfg: Dict[str, Any], obs_space: gym.spaces.Dict, action_space: gym.Space) -> PPOAgent:
    """The agent with its initial weights, on the CPU: the algorithm sets it up with Fabric."""
    actions_dim, is_continuous = actions_dim_of(action_space)
    return PPOAgent(
        actions_dim=actions_dim,
        obs_space=obs_space,
        encoder_cfg=cfg.algo.encoder,
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        cnn_keys=cfg.algo.cnn_keys.encoder,
        mlp_keys=cfg.algo.mlp_keys.encoder,
        screen_size=cfg.env.screen_size,
        distribution_cfg=cfg.distribution,
        is_continuous=is_continuous,
    )
```

## Loss functions
All the loss functions to be optimized by the agent during the training should be defined under the `loss.py` file, even though is not strictly necessary:

```python
import torch.nn.functional as F
from torch import Tensor


def policy_loss(logprobs: Tensor, advantages: Tensor) -> Tensor:
    return -(logprobs * advantages).mean()


def value_loss(values: Tensor, returns: Tensor) -> Tensor:
    return F.mse_loss(values, returns)
```

## Algorithm implementation
The algorithm is implemented in the `sota.py` file. It contains:

1. **The training state**: a dataclass that subclasses `TrainState` and lists everything that changes during the training (modules, optimizers, learning-rate schedulers, annealed coefficients as tensors, counters). The training loop saves it in the checkpoints, one entry per field with the name of the field, and restores it when a run is resumed. It holds no logic.
2. **The writer**: a `Writer` (`sheeprl/core/collector.py`) whose `write(store, step, act)` writes in the store a step of the environments (`EnvStep`) played with the actions `act`. The actions come from the policy, a `Policy` (usually the policy module of the agent): `act(obs, greedy)` returns an `Act` with the actions to play in the environments (`env_actions`), the columns to store (`columns`, e.g. the actions one-hot, their log-probabilities, the values of the observations) and anything else the writer needs (`extras`); `random(env)` returns random actions (by default those of the environments), `init_states(num_envs)` creates the memory of the policy for `num_envs` environments before their first step (e.g. a recurrent state; nothing by default), and `reset_state(env_idxes)` forgets it for the environments that start a new episode (nothing by default). The environments reset themselves.
3. **The algorithm**: a subclass of `Algorithm` (`sheeprl/core/algorithm.py`), with:
   - `steps_per_iteration`: the steps played by every environment in an iteration (the rollout length of an on-policy algorithm; 1, the default, for an off-policy one);
   - `off_policy`: whether it trains on a replay buffer (see [Off-policy algorithms](#off-policy-algorithms));
   - `random_warmup`: whether an off-policy algorithm plays random actions until `algo.learning_starts` (the default) or its policy from the first step (e.g. to finetune a policy);
   - `restart_crashed_envs`: whether a crashed environment is created again instead of stopping the run (its writer then handles `EnvStep.restarted`);
   - `build(obs_space, action_space, schedule, log_dir)`: creates the training state and the store of the collected data;
   - `policy(state)`: returns the policy, which shares its weights with the trained modules of `state`;
   - `writer(state, policy)`: returns the writer;
   - `batches(state, store, n_steps, iteration)`: prepares the training data of an iteration and yields one batch per gradient step;
   - `train_step(state, batch, step)`: one gradient step on a batch; it returns the metrics to log, as tensors (don't read them with `.item()`: they are read once per log interval). `step` counts the gradient steps of the process since the start of the training, e.g. to update a target network every few steps;
   - `test(state, log_dir, policy_step)` (optional): plays a test episode with the policy and logs its return (`sheeprl.core.run_test`): its most likely actions, or sampled ones with `greedy_test = False`;
   - `end_iteration(state, iteration)` (optional): what changes once per iteration, after the training (e.g. annealed coefficients); it returns values logged at every iteration.
4. **The entrypoint**: a function decorated with `register_algorithm` that runs the algorithm with `run`, then tests and registers the trained models.

Some functions of `sheeprl.core` help with Fabric:

- `setup_module(fabric, module)` moves a module to the device and runs it in the precision of the run (`fabric.precision`); with several processes, it copies the weights of rank 0 to the others, so every process starts from the same weights. Set up every module you train with it, and the optimizers with `fabric.setup_optimizers`;
- `update(fabric, loss, optimizer, max_grad_norm=...)` does one optimizer step: it computes the gradients of the weights of `optimizer` only, averages them over the processes, clips them (when `max_grad_norm > 0`) and steps the optimizer. It returns the norm of the gradients before clipping (`None` without clipping). Every process must call it the same number of times;
- `autocast(fabric)` wraps the forward passes and the loss of one update in the precision of the run; close it before calling `update`.

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

from sheeprl.algos.ppo.agent import PPOAgent, PPOPolicy
from sheeprl.algos.sota.agent import build_agent
from sheeprl.algos.sota.loss import policy_loss, value_loss
from sheeprl.utils.obs import normalize_obs, prepare_obs
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
class SOTAState(TrainState):
    # Saved in the checkpoints as `agent` and `optimizer`
    agent: PPOAgent
    optimizer: Optimizer


class SOTAWriter(Writer):
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


class SOTA(Algorithm):
    """Every iteration plays `algo.rollout_steps` steps, then trains for `algo.update_epochs` epochs of minibatches of
    the rollout."""

    def __init__(self, fabric: Fabric, cfg: Dict[str, Any]) -> None:
        super().__init__(fabric, cfg)
        self.steps_per_iteration = cfg.algo.rollout_steps

    def build(
        self, obs_space: gym.spaces.Dict, action_space: gym.Space, schedule: TrainSchedule, log_dir: str
    ) -> Tuple[SOTAState, ReplayStore]:
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
        return SOTAState(agent=agent, optimizer=optimizer), rollout

    def policy(self, state: SOTAState) -> PPOPolicy:
        """The policy to play with: it shares its modules, and so its weights, with the trained agent."""
        cfg = self.cfg
        return PPOPolicy(
            state.agent.feature_extractor,
            state.agent.actor,
            state.agent.critic,
            device=self.fabric.device,
            obs_keys=cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder,
            cnn_keys=cfg.algo.cnn_keys.encoder,
        )

    def writer(self, state: SOTAState, policy: PPOPolicy) -> SOTAWriter:
        return SOTAWriter(self.cfg)

    def batches(
        self, state: SOTAState, rollout: ReplayStore, n_steps: Optional[int], iteration: int
    ) -> Iterator[Dict[str, Tensor]]:
        cfg = self.cfg
        obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
        data = rollout.read()

        # The returns and the advantages, bootstrapped with the value of the observations after the rollout
        with torch.inference_mode():
            next_obs = {k: rollout.last_step.next_obs[k] for k in obs_keys}
            next_obs = prepare_obs(
                self.fabric.device, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs
            )
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

    def train_step(self, state: SOTAState, batch: Dict[str, Tensor], step: int) -> Dict[str, Tensor]:
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
    algo = SOTA(fabric, cfg)
    state, log_dir, policy_step = run(fabric, cfg, algo)

    if fabric.is_global_zero and cfg.algo.run_test:
        # The return of the trained agent, logged at the last policy step of the training
        algo.test(state, log_dir, policy_step=policy_step)

    # Optional: register the trained models with MLflow
    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sota.utils import log_models
        from sheeprl.utils.mlflow import register_model

        register_model(fabric, log_models, cfg, {"agent": state.agent})
```

With several processes, every process plays its own environments and trains on its own data, while `update` averages the gradients of the processes: every process must do the same number of gradient steps. To train on the data of all the processes instead, gather it in `batches` with `fabric.all_gather` and split it with a `DistributedSampler`, as PPO does with `buffer.share_data=True` (`sheeprl/algos/ppo/ppo.py`).

### Off-policy algorithms
An off-policy algorithm sets `off_policy = True` and returns a replay buffer as its store: `sheeprl.core.transition_store` to train on single steps (as SAC does), `sheeprl.core.sequence_store` to train on sequences of one environment (as the Dreamers do). Then:

- its configuration must have `algo.learning_starts` (the policy steps played with random actions before the training starts), `algo.replay_ratio` (the gradient steps per policy step) and `algo.per_rank_pretrain_steps` (the gradient steps the first training does besides the ones of the replay ratio);
- the collector plays the random actions of `policy.random(env)` while `schedule.warmup(schedule.policy_step)` is true (unless `random_warmup = False`): by default the random actions of the environments, as SAC does (`sheeprl/algos/sac/sac.py`). A policy that stores its actions in another form overrides it, e.g. the Dreamers store the random discrete actions one-hot (`DreamerPolicy` in `sheeprl/algos/dreamer_policy.py`);
- `batches` receives the number of gradient steps of the iteration, `n_steps`, and yields exactly `n_steps` batches sampled from the buffer;
- the replay buffer is saved in the checkpoints when `buffer.checkpoint=True`. A run resumed with its buffer doesn't play random actions again; one resumed without it fills a new buffer with its policy for `algo.learning_starts` policy steps first.

### Utils
The observations are moved to the device by `prepare_obs` and the images of the batches are normalized by `normalize_obs` (`sheeprl/utils/obs.py`); the test episode at the end of the training is the one of every algorithm (`Algorithm.test`). The `log_models` function imported above is defined in the `sheeprl.algos.sota.utils` module, here the one of PPO (`sheeprl/algos/ppo/utils.py`): it registers the models with MLflow at the end of the training.

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import gymnasium as gym
from lightning import Fabric

# The model logging of PPO fits this algorithm too
from sheeprl.algos.ppo.utils import log_models  # noqa: F401
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

# The metrics the algorithm can log and the models it can register
AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/policy_loss", "Loss/value_loss"}
MODELS_TO_REGISTER = {"agent"}


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, "ModelInfo"]:
    """Register the models of a checkpoint (`sheeprl-registration`)."""
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    from sheeprl.algos.sota.sota import SOTA
    from sheeprl.core import load_trained_state

    # The models are built as by the training, with its configuration
    algo = SOTA(fabric, cfg.to_log)
    trained = load_trained_state(fabric, cfg.to_log, algo, state, env.observation_space, env.action_space)
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True):
        agent = unwrap_fabric(trained.agent)
        model_info = {"agent": mlflow.pytorch.log_model(agent, name="agent", serialization_format="pickle")}
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
```

### Evaluation
To evaluate a checkpoint with `sheeprl-eval` (see [the how-to on evaluation](./eval_your_agent.md)), the `evaluate.py` file registers an evaluation function for the algorithm. `load_trained_state` builds the training state with the `build` of the algorithm and restores it from the checkpoint:

```python
from __future__ import annotations

from typing import Any, Dict

from lightning import Fabric

from sheeprl.algos.sota.sota import SOTA
from sheeprl.core import evaluate as evaluate_trained
from sheeprl.utils.registry import register_evaluation


@register_evaluation(algorithms="sota")
def evaluate(fabric: Fabric, cfg: Dict[str, Any], state: Dict[str, Any]):
    # The models are built as by the training (`build`), restored from the checkpoint and tested (`Algorithm.test`)
    evaluate_trained(fabric, cfg, state, SOTA(fabric, cfg))
```

### Metrics and Model Manager
Each algorithm logs its own metrics: the ones returned by `train_step` (aggregated over the gradient steps and logged every `metric.log_every` policy steps), the ones returned by `end_iteration` (logged at every iteration) and the episode statistics recorded by the environments (`Rewards/rew_avg` and `Game/ep_len_avg`). To define which are the metrics that can be logged, you need to define the `AGGREGATOR_KEYS` variable in the `./sheeprl/algos/sota/utils.py` file. It must be a set of strings (the name of the metrics to log). Then, you can decide which metrics to log by defining the `metric.aggregator.metrics` in the configs.

> **Remember**
>
> The intersection between the keys in the `AGGREGATOR_KEYS` and the ones in the `metric.aggregator.metrics` config will be logged.

As for metrics, you have to specify which are the models that can be registered after training, you need to define the `MODELS_TO_REGISTER` variable in the `./sheeprl/algos/sota/utils.py` file. It must be a set of strings (the name of the models you want to register). As before, you can easily select which agents to register by defining the `model_manager.models` in the configs. Also in this case, the models that will be registered are the intersection between the `MODELS_TO_REGISTER` variable and the keys of the `model_manager.models` config.

## Config files
Once you have written your algorithm, you need to create three config files: one in `./sheeprl/configs/algo`, one in `./sheeprl/configs/exp` and one in `./sheeprl/configs/model_manager`.

```bash
configs
└── algo
    ├── default.yaml
    ├── dreamer_v1.yaml
    ...
    └── sota.yaml
...
└── exp
    ├── default.yaml
    ├── dreamer_v1.yaml
    ...
    └── sota.yaml
...
└── model_manager
    ├── default.yaml
    ├── dreamer_v1.yaml
    ...
    └── sota.yaml
```

#### Algo Configs
In the `./sheeprl/configs/algo/sota.yaml` we need to specify all the configs needed to initialize and train your agent. The `algo/default.yaml` config, which it extends, holds the keys every algorithm has: `name`, `total_steps`, `per_rank_batch_size`, `run_test` and the observation keys (`cnn_keys` and `mlp_keys`).
Here is an example of the `./sheeprl/configs/algo/sota.yaml` config file:

```yaml
defaults:
  - default
  - /optim@optimizer: adam
  - _self_

# Must be equal to the name of the file with the implementation: `sheeprl/algos/sota/sota.py`
name: sota

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
> The field `algo.name` **must** be set and **must** be equal to the name of the file.py, found under the `sheeprl/algos/sota` folder, where the implementation of the algorithm is defined. For example, if your implementation is defined in a python file named `my_sota.py`, i.e. `sheeprl/algos/sota/my_sota.py`, then `algo.name="my_sota"` 

#### Model Manager Configs
In the `./sheeprl/configs/model_manager/sota.yaml` we need to specify all the configs needed to register your agent. You can specify a name, a description, and some tags for each model you want to register. The `disabled` parameter indicates whether or not you want to register your models.
Here is an example of the `./sheeprl/configs/model_manager/sota.yaml` config file:

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
Here is an example of the `./sheeprl/configs/exp/sota.yaml` config file:

```yaml
# @package _global_

defaults:
  - override /algo: sota
  - override /env: gym
  # select the model manager configs
  - override /model_manager: sota
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

With `override /algo: sota` in `defaults` you are specifying you want to use the new `sota` algorithm, whereas, with `override /env: gym` you are specifying that you want to train your agent on a *Gymnasium* environment, `CartPole-v1`.

## Register Algorithm

To let the `register_algorithm` and `register_evaluation` decorators add our new `sota` algorithm and its evaluation to the registries we need to import them in `./sheeprl/__init__.py`:

```diff
import os

from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = os.path.dirname(__file__)


from sheeprl.utils.imports import _IS_TORCH_GREATER_EQUAL_2_0

if not _IS_TORCH_GREATER_EQUAL_2_0:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_0)

# fmt: off
from sheeprl.algos.a2c import a2c  # noqa: F401
...
from sheeprl.algos.sac_ae import sac_ae  # noqa: F401
+from sheeprl.algos.sota import sota  # noqa: F401

from sheeprl.algos.a2c import evaluate as a2c_evaluate  # noqa: F401, isort:skip
...
from sheeprl.algos.sac_ae import evaluate as sac_ae_evaluate  # noqa: F401, isort:skip
+from sheeprl.algos.sota import evaluate as sota_evaluate  # noqa: F401, isort:skip
# fmt: on
```

Then if you run `python sheeprl/available_agents.py` you should see that `sota` appears in the list of all the available agents:

```bash
                                 SheepRL Agents
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Module              ┃ Algorithm           ┃ Entrypoint ┃ Evaluated by        ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ sheeprl.algos.a2c   │ a2c                 │ main       │ sheeprl.algos.a2c.… │
...
│ sheeprl.algos.sac_… │ sac_ae              │ main       │ sheeprl.algos.sac_… │
│ sheeprl.algos.sota  │ sota                │ main       │ sheeprl.algos.sota… │
└─────────────────────┴─────────────────────┴────────────┴─────────────────────┘
```

Now you can train the agent with `python sheeprl.py exp=sota` and evaluate one of its checkpoints with `python sheeprl_eval.py checkpoint_path=...`.
