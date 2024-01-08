# Register an external algorithm

Suppose that we have installed SheepRL through pip with `pip install sheeprl[box2d,atari,dev,test]` and we want to add a new (external) SoTA algorithm called `ext_sota` without directly adding the new algorithm to the SheepRL codebase, i.e. without the need to clone the repo locally. 

We can start by creating two new folders called `my_awesome_algo` and `my_awesome_configs`, the former will contain the implementation of the algorithm, the latter the configs needed to run the experiment and configure our new algorithm. 

Reading our [how-to on how to register a new algorithm](../howto/register_new_algorithm.md) we can see that we need to create at least four new files under the `my_awesome_algo` folder:

```bash
my_awesome_algo
├── __init__.py
├── agent.py
├── loss.py
├── ext_sota.py
└── utils.py
```

## The agent

The agent is the core of the algorithm and it is defined in the `agent.py` file. It must contain at least single function called `build_agent` that returns a `torch.nn.Module` wrapped with Fabric:

```python
from __future__ import annotations

from typing import Any, Dict, Sequence

import gymnasium
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
import torch
from torch import Tensor


class SOTAAgent(torch.nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tensor:
        ...


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    observation_space: gymnasium.spaces.Dict,
    state: Dict[str, Any] | None = None,
) -> _FabricModule:

    # Define the agent here
    agent = SOTAAgent(...)

    # Load the state from the checkpoint
    if state:
        agent.load_state_dict(state)

    # Setup the agent with Fabric
    agent = fabric.setup_model(agent)

    return agent
```

## Loss functions

All the loss functions to be optimized by the agent during the training should be defined under the `loss.py` file, even though is not strictly necessary:

```python
import torch
import ...

def loss1(...) -> Tensor:
    ...

def loss2(...) -> Tensor:
    ...
```

## Algorithm implementation

The real algorithm implementation has to be placed under the `ext_sota.py` file, which needs to contain a single entrypoint decorated with the `register_algorithm` decorator:

```python
import copy
import os
import time
from datetime import datetime

import gymnasium as gym
import hydra
import torch
from lightning.fabric import Fabric
from torchmetrics import MeanMetric, SumMetric
from sheeprl.data import ReplayBuffer
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.logger import get_logger, get_log_dir
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import unwrap_fabric

from my_awesome_algo.agent import build_agent
from my_awesome_algo.loss import loss1, loss2
from my_awesome_algo.utils import normalize_obs, test


def train(
    fabric: Fabric,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, torch.Tensor],
    aggregator: MetricAggregator,
    cfg: Dict[str, Any],
):
    l1 = loss1(...)
    l2 = loss2(...)
    loss = 0.5 * (l1 + l2)

    optimizer.zero_grad(set_to_none=True)
    fabric.backward(loss)
    optimizer.step()

    # Update metrics
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/loss1", l1.detach())
        aggregator.update("Loss/loss2", l2.detach())


@register_algorithm(decoupled=False)
def ext_sota_main(fabric: Fabric, cfg: Dict[str, Any]):
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    observation_space = envs.single_observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: "
            "`cnn_keys.encoder=[rgb]` or `mlp_keys.encoder=[state]`"
        )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(envs.single_action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        envs.single_action_space.shape
        if is_continuous
        else (envs.single_action_space.nvec.tolist() if is_multidiscrete else [envs.single_action_space.n])
    )

    # Create the agent model: this should be a torch.nn.Module to be accelerated with Fabric
    # Given that the environment has been created with the `make_env` method, the agent
    # forward method must accept as input a dictionary like {"obs1_name": obs1, "obs2_name": obs2, ...}.
    # The agent should be able to process both image and vector-like observations.
    agent = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["agent"] if cfg.checkpoint.resume_from else None,
    )

    # Define the optimizer
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters(), _convert_="all")

    # Load the state from the checkpoint
    if cfg.checkpoint.resume_from:
        optimizer.load_state_dict(state["optimizer"])

    # Setup agent and optimizer with Fabric
    optimizer = fabric.setup_optimizers(optimizer)

    # Create a metric aggregator to log the metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Local data
    rb = ReplayBuffer(
        cfg.buffer.size,
        cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=obs_keys,
    )

    # Global variables
    last_train = 0
    train_step = 0
    start_step = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["update"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["update"] * cfg.env.num_envs * cfg.algo.rollout_steps if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * cfg.algo.rollout_steps * world_size)
    num_updates = cfg.algo.total_steps // policy_steps_per_update if not cfg.dry_run else 1
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    # Get the first environment observation and start the optimization
    step_data = {}
    next_obs = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    for k in obs_keys:
        if k in cfg.algo.cnn_keys.encoder:
            next_obs[k] = next_obs[k].reshape(cfg.env.num_envs, -1, *next_obs[k].shape[-2:])
        step_data[k] = next_obs[k][np.newaxis]

    for update in range(start_step, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs * world_size

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                with torch.no_grad():
                    # Sample an action given the observation received by the environment
                    normalized_obs = normalize_obs(next_obs, cfg.algo.cnn_keys.encoder, obs_keys)
                    torch_obs = {
                        k: torch.as_tensor(normalized_obs[k], dtype=torch.float32, device=device) for k in obs_keys
                    }
                    actions = agent.module(torch_obs)
                    if is_continuous:
                        real_actions = torch.cat(actions, -1).cpu().numpy()
                    else:
                        real_actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1).cpu().numpy()
                    actions = torch.cat(actions, -1).cpu().numpy()

                # Single environment step
                obs, rewards, dones, truncated, info = envs.step(real_actions.reshape(envs.action_space.shape))
                dones = np.logical_or(dones, truncated).reshape(cfg.env.num_envs, -1).astype(np.uint8)
                rewards = rewards.reshape(cfg.env.num_envs, -1)

            # Update the step data
            step_data["dones"] = dones[np.newaxis]
            step_data["actions"] = actions[np.newaxis]
            step_data["rewards"] = rewards[np.newaxis]
            if cfg.buffer.memmap:
                step_data["returns"] = np.zeros_like(rewards, shape=(1, *rewards.shape))
                step_data["advantages"] = np.zeros_like(rewards, shape=(1, *rewards.shape))

            # Append data to buffer
            rb.add(step_data, validate_args=False)

            # Update the observation and dones
            next_obs = {}
            for k in obs_keys:
                _obs = obs[k]
                if k in cfg.algo.cnn_keys.encoder:
                    _obs = _obs.reshape(cfg.env.num_envs, -1, *_obs.shape[-2:])
                step_data[k] = _obs[np.newaxis]
                next_obs[k] = _obs

            if cfg.metric.log_level > 0 and "final_info" in info:
                for i, agent_ep_info in enumerate(info["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        if aggregator and "Rewards/rew_avg" in aggregator:
                            aggregator.update("Rewards/rew_avg", ep_rew)
                        if aggregator and "Game/ep_len_avg" in aggregator:
                            aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Transform the data into PyTorch Tensors
        local_data = rb.to_tensor(dtype=None, device=device)

        # Train the agent
        train(fabric, agent, optimizer, local_data, aggregator, cfg)

        # Log metrics
        if policy_step - last_log >= cfg.metric.log_every or update == num_updates or cfg.dry_run:
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint model
        if (
            (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every)
            or cfg.dry_run
            or update == num_updates
        ):
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_step": update,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(agent.module, fabric, cfg, log_dir)

    # Optional part in case you want to give the possibility to register your models with MLFlow
    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from my_awesome_algo.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"agent": agent}
        register_model(fabric, log_models, cfg, models_to_log)
```

where `log_models`, `test` and `normalize_obs` have to be defined in the `my_awesome_algo.utils` module, for example like this: 

```python
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict

import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

from my_awesome_algo.agent import SOTAAgent

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


@torch.no_grad()
def test(agent: SOTAAgent, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    o = env.reset(seed=cfg.seed)[0]
    obs = {}
    for k in o.keys():
        if k in cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
            if k in cfg.algo.cnn_keys.encoder:
                torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255 - 0.5
            if k in cfg.algo.mlp_keys.encoder:
                torch_obs = torch_obs.float()
            obs[k] = torch_obs

    while not done:
        # Act greedly through the environment
        if agent.is_continuous:
            actions = torch.cat(agent.get_greedy_actions(obs), dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in agent.get_greedy_actions(obs)], dim=-1)

        # Single environment step
        o, reward, done, truncated, _ = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        obs = {}
        for k in o.keys():
            if k in cfg.algo.mlp_keys.encoder + cfg.algo.cnn_keys.encoder:
                torch_obs = torch.from_numpy(o[k]).to(fabric.device).unsqueeze(0)
                if k in cfg.algo.cnn_keys.encoder:
                    torch_obs = torch_obs.reshape(1, -1, *torch_obs.shape[-2:]) / 255 - 0.5
                if k in cfg.algo.mlp_keys.encoder:
                    torch_obs = torch_obs.float()
                obs[k] = torch_obs

        if cfg.dry_run:
            done = True
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0:
        fabric.log_dict({"Test/cumulative_reward": cumulative_rew}, 0)
    env.close()


def normalize_obs(
    obs: Dict[str, np.ndarray | Tensor], cnn_keys: Sequence[str], obs_keys: Sequence[str]
) -> Dict[str, np.ndarray | Tensor]:
    return {k: obs[k] / 255 - 0.5 if k in cnn_keys else obs[k] for k in obs_keys}


def log_models(
    cfg: Dict[str, Any],
    models_to_log: Dict[str, torch.nn.Module | _FabricModule],
    run_id: str,
    experiment_id: str | None = None,
    run_name: str | None = None,
) -> Dict[str, "ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=True) as _:
        model_info = {}
        unwrapped_models = {}
        for k in cfg.model_manager.models.keys():
            if k not in models_to_log:
                warnings.warn(f"Model {k} not found in models_to_log, skipping.", category=UserWarning)
                continue
            unwrapped_models[k] = unwrap_fabric(models_to_log[k])
            model_info[k] = mlflow.pytorch.log_model(unwrapped_models[k], artifact_path=k)
        mlflow.log_dict(cfg, "config.json")
    return model_info
```

### Metrics and Model Manager

Each algorithm logs its own metrics during training or environment interaction. To define which are the metrics that can be logged, you need to define the `AGGREGATOR_KEYS` variable in the `./my_awesome_algo/utils.py` file. It must be a set of strings (the name of the metrics to log). Then, you can decide which metrics to log by defining the `metric.aggregator.metrics` in the configs.

> **Remember**
>
> The intersection between the keys in the `AGGREGATOR_KEYS` and the ones in the `metric.aggregator.metrics` config will be logged.

As for metrics, you have to specify which are the models that can be registered after training, you need to define the `MODELS_TO_REGISTER` variable in the `./my_awesome_algo/utils.py` file. It must be a set of strings (the name of the variables of the models you want to register). As before, you can easily select which agents to register by defining the `model_manager.models` in the configs. Also in this case, the models that will be registered are the intersection between the `MODELS_TO_REGISTER` variable and the keys of the `model_manager.models` config.

In this case, the `./my_awesome_algo/utils.py` file could be defined as below:

```python
# `./my_awesome_algo/utils.py`

...

AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/loss1", "Loss/loss2"}
MODELS_TO_REGISTER = {"agent"}

...
```

## Config files

Once you have written your algorithm, you need to create three config files: one in `./my_awesome_configs/algo`, one in `./my_awesome_configs/exp` and the other one in `./my_awesome_configs/model_manager`.


```tree
.
├── my_awesome_algo
|   ├── __init__.py
│   ├── agent.py
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

name: ext_sota  # This must be set! And must be equal to the name of the file.py, found under the `my_awesome_algo` folder, where the implementation of the algorithm is defined

# Algorithm-related paramters
...

# Agent model parameters 
# This is just an example where we suppose we have an Actor-Critic agent
# with both an MLP and a CNN encoder
mlp_layers: 2
dense_units: 64
layer_norm: False
max_grad_norm: 0.0
dense_act: torch.nn.Tanh
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

# Override parameters coming from `adam.yaml` config
optimizer:
  lr: 1e-3
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

disabled: False
models: 
  agent:
    model_name: "${exp_name}"
    description: "SOTA Agent in ${env.id} Environment"
    tags: {}
```

#### Experiment Configs

In the second file, you have to specify all the elements you want in your experiment and you can override all the parameters you want.
Here is an example of the `./my_awesome_configs/exp/ext_sota.yaml` config file:

```yaml
# @package _global_

defaults:
  - override /algo: ext_sota
  - override /env: atari
  # select the model manager configs
  - override /model_manager: ext_sota
  - _self_

algo:
  total_steps: 65536
  per_rank_batch_size: 64

buffer:
  share_data: False

# override environment id
env:
  env:
    id: MsPacmanNoFrameskip-v4

# select which metrics to log
metric:
  aggregator:
    metrics:
      Loss/loss1:
        _target_: torchmetrics.MeanMetric
        sync_on_compute: ${metric.sync_on_compute}
      Loss/loss2:
        _target_: torchmetrics.MeanMetric
        sync_on_compute: ${metric.sync_on_compute}
```

With `override /algo: ext_sota` in `defaults` you are specifying you want to use the new `ext_sota` algorithm, whereas, with `override /env: atari` you are specifying that you want to train your agent on an *Atari* environment.

## Register the algorithm and the configs

To let the `register_algorithm` decorator add our new `ext_sota` algorithm to the available algorithms registry we need first to create a new file called for example `my_awesome_main.py` in the root of the project:

```tree
.
├── my_awesome_algo
|   ├── __init__.py
│   ├── agent.py
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
└── my_awesome_main.py
```

containing the following:

```python
# my_awesome_main.py

# This will trigger the algorithm registration of SheepRL
from my_awesome_algo import ext_sota  # noqa: F401

if __name__ == "__main__":
    # This must be imported after the algorithm registration, otherwise SheepRL
    # will not be able to find the new algorithm given the name specified
    # in the `algo.name` field of the `./my_awesome_configs/algo/ext_sota.yaml` config file
    from sheeprl.cli import run

    run()
```

While to let SheepRL know about the new configs we need to add a new file called `.env` to the root of the project containing the following env variable:

```bash
SHEEPRL_SEARCH_PATH=file://my_awesome_configs;pkg://sheeprl.configs
```

This tells SheepRL to search for configs in the `my_awesome_configs` folder and in the `configs` folder under the installed `sheeprl` package.

## Run the experiment

Then you can run your experiment with `python my_awesome_main.py exp=ext_sota env=... env.id=...`.
