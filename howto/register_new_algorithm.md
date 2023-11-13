# Register a new algorithm
Suppose that we want to add a new SoTA algorithm to sheeprl called `sota` so that we can train an agent simply with `python sheeprl.py exp=sota env=... env.id=...`.  

We start by creating a new folder called `sota` under `./sheeprl/algos/`, containing the following files:

```bash
algos
└── droq
...
└── sota
    ├── __init__.py
    ├── loss.py
    ├── sota.py
    └── utils.py
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
The real algorithm implementation has to be placed under the `sota.py` file, which needs to contain a single entrypoint decorated with the `register_algorithm` decorator:

```python
import copy
import os
import time
from datetime import datetime

import gymnasium as gym
import hydra
import torch
from gymnasium.vector import SyncVectorEnv
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Adam
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.sota.loss import loss1, loss2
from sheeprl.algos.sota.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.models.models import MLP
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger, get_log_dir
from sheeprl.utils.timer import timer


def train(
    fabric: Fabric,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
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
    aggregator.update("Loss/loss1", l1.detach())
    aggregator.update("Loss/loss2", l2.detach())


@register_algorithm(decoupled=False)
def sota_main(fabric: Fabric, cfg: Dict[str, Any]):
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger = create_tensorboard_logger(fabric, cfg)
    if fabric.is_global_zero:
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
                logger.log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            ),
            for i in range(cfg.env.num_envs)
        ]
    )

    # Create the agent model: this should be a torch.nn.Module to be accelerated with Fabric
    # Given that the environment has been created with the `make_dict_env` method, the agent
    # forward method must accept as input a dictionary like {"obs1_name": obs1, "obs2_name": obs2, ...}.
    # The agent should be able to process both image and vector-like observations.
    agent = ...

    # Define the agent and the optimizer and set up them with Fabric
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=list(agent.parameters()))
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Create a metric aggregator to log the metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(),
                "Game/ep_len_avg": MeanMetric(),
                "Loss/value_loss": MeanMetric(),
                "Loss/policy_loss": MeanMetric(),
                "Loss/entropy_loss": MeanMetric(),
            }
        )

    # Local data
    rb = ReplayBuffer(cfg.algo.rollout_steps, cfg.env.num_envs, device=device, memmap=cfg.buffer.memmap)
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    last_log = 0
    last_train = 0
    train_step = 0
    policy_step = 0
    last_checkpoint = 0
    policy_steps_per_update = int(cfg.env.num_envs * cfg.algo.rollout_steps * world_size)
    num_updates = cfg.total_steps // policy_steps_per_update if not cfg.dry_run else 1

    # Warning for log and checkpoint every
    if cfg.metric.log_every % policy_steps_per_update != 0:
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
    o = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    next_obs = {}
    for k in o.keys():
        if k in obs_keys:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device)
            if k in cfg.cnn_keys.encoder:
                torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
            if k in cfg.mlp_keys.encoder:
                torch_obs = torch_obs.float()
            step_data[k] = torch_obs
            next_obs[k] = torch_obs
    next_done = torch.zeros(cfg.env.num_envs, 1, dtype=torch.float32).to(fabric.device)  # [N_envs, 1]

    for update in range(1, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs * world_size

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
                with torch.no_grad():
                    # Sample an action given the observation received by the environment
                    # This calls the `forward` method of the PyTorch module, escaping from Fabric
                    # because we don't want this to be a synchronization point
                    action = agent.module(next_obs)

                # Single environment step
                o, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                rewards = torch.tensor(reward).view(cfg.env.num_envs, -1)  # [N_envs, 1]
                done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))  # [N_envs, 1]
                done = done.view(cfg.env.num_envs, -1).float()

            # Update the step data
            step_data["dones"] = next_done
            step_data["actions"] = action
            step_data["rewards"] = rewards

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            # Update the observation and done
            obs = {}
            for k in o.keys():
                if k in obs_keys:
                    torch_obs = torch.from_numpy(o[k]).to(fabric.device)
                    if k in cfg.cnn_keys.encoder:
                        torch_obs = torch_obs.view(cfg.env.num_envs, -1, *torch_obs.shape[-2:])
                    if k in cfg.mlp_keys.encoder:
                        torch_obs = torch_obs.float()
                    step_data[k] = torch_obs
                    obs[k] = torch_obs
            next_obs = obs
            next_done = done

            if "final_info" in info:
                for i, agent_ep_info in enumerate(info["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        # Train the agent
        train(fabric, agent, optimizer, local_data, aggregator, cfg)

        # Log metrics
        if policy_step - last_log >= cfg.metric.log_every or update == num_updates or cfg.dry_run:
            # Sync distributed metrics
            metrics_dict = aggregator.compute()
            fabric.log_dict(metrics_dict, policy_step)
            aggregator.reset()

            # Sync distributed timers
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
    if fabric.is_global_zero:
        test(actor.module, envs, fabric, cfg)
```

## Config files
Once you have written your algorithm, you need to create two config files: one in `./sheeprl/configs/algo` and the other in `./sheeprl/configs/exp`.

> **Note**
>
> The name of the two files should be the same as the algorithm, so in our case, it is `sota.yaml`

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
```

#### Algo configs
In the `./sheeprl/configs/algo/sota.yaml` we need to specify all the configs needed to initialize and train your agent.
Here is an example of the `./sheeprl/configs/algo/sota.yaml` config file:

```yaml
defaults:
  - default
  - /optim@optimizer: adam
  - _self_

name: sota  # This must be set!

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

> **Note**
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

> **Note**
>
> The field `algo.name` **must** be set and **must** be equal to the name of the file.py, found under the `sheeprl/algos/sota` folder, where the implementation of the algorithm is defined. For example, if your implementation is defined in a python file named `my_sota.py`, i.e. `sheeprl/algos/sota/my_sota.py`, then `algo.name="my_sota"` 

#### Experiment config
In the second file, you have to specify all the elements you want in your experiment and you can override all the parameters you want.
Here is an example of the `./sheeprl/configs/exp/sota.yaml` config file:

```yaml
# @package _global_

defaults:
  - override /algo: sota
  - override /env: atari
  - _self_

total_steps: 65536
per_rank_batch_size: 64
buffer:
  share_data: False

# override environment id
env:
  env:
    id: MsPacmanNoFrameskip-v4
```

With `override /algo: sota` in `defaults` you are specifying you want to use the new `sota` algorithm, whereas, with `override /env: gym` you are specifying that you want to train your agent on an *Atari* environment.

## Register Algorithm

To let the `register_algorithm` decorator add our new `sota` algorithm to the available algorithms registry we need to import it in `./sheeprl/__init__.py`: 

```diff
import os

ROOT_DIR = os.path.dirname(__file__)

from dotenv import load_dotenv

load_dotenv()

from sheeprl.utils.imports import _IS_TORCH_GREATER_EQUAL_2_0

if not _IS_TORCH_GREATER_EQUAL_2_0:
    raise ModuleNotFoundError(_IS_TORCH_GREATER_EQUAL_2_0)

# Needed because MineRL 0.4.4 is not compatible with the latest version of numpy
import numpy as np

from sheeprl.algos.dreamer_v1 import dreamer_v1 as dreamer_v1
from sheeprl.algos.dreamer_v2 import dreamer_v2 as dreamer_v2
from sheeprl.algos.dreamer_v3 import dreamer_v3 as dreamer_v3
from sheeprl.algos.droq import droq as droq
from sheeprl.algos.p2e_dv1 import p2e_dv1 as p2e_dv1
from sheeprl.algos.p2e_dv2 import p2e_dv2 as p2e_dv2
from sheeprl.algos.p2e_dv3 import p2e_dv3 as p2e_dv3
from sheeprl.algos.ppo import ppo as ppo
from sheeprl.algos.ppo import ppo_decoupled as ppo_decoupled
from sheeprl.algos.ppo_recurrent import ppo_recurrent as ppo_recurrent
from sheeprl.algos.sac import sac as sac
from sheeprl.algos.sac import sac_decoupled as sac_decoupled
from sheeprl.algos.sac_ae import sac_ae as sac_ae
+from sheeprl.algos.sota import sota as sota

np.float = np.float32
np.int = np.int64
np.bool = bool

__version__ = "0.4.3"
```

Then if you run `python sheeprl/available_agents.py` you should see that `sota` appears in the list of all the available agents:

```bash
SheepRL Agents                             
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Module                      ┃ Algorithm     ┃ Entrypoint ┃ Decoupled ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ sheeprl.algos.dreamer_v1    │ dreamer_v1    │ main       │ False     │
│ sheeprl.algos.dreamer_v2    │ dreamer_v2    │ main       │ False     │
│ sheeprl.algos.dreamer_v3    │ dreamer_v3    │ main       │ False     │
│ sheeprl.algos.sac           │ sac           │ main       │ False     │
│ sheeprl.algos.sac           │ sac_decoupled │ main       │ True      │
│ sheeprl.algos.droq          │ droq          │ main       │ False     │
│ sheeprl.algos.p2e_dv1       │ p2e_dv1       │ main       │ False     │
│ sheeprl.algos.p2e_dv2       │ p2e_dv2       │ main       │ False     │
│ sheeprl.algos.p2e_dv3       │ p2e_dv3       │ main       │ False     │
│ sheeprl.algos.ppo           │ ppo           │ main       │ False     │
│ sheeprl.algos.ppo           │ ppo_decoupled │ main       │ True      │
│ sheeprl.algos.ppo_recurrent │ ppo_recurrent │ main       │ False     │
│ sheeprl.algos.sac_ae        │ sac_ae        │ main       │ False     │
│ sheeprl.algos.sota          │ sota          │ sota_main  │ False     │
└─────────────────────────────┴───────────────┴────────────┴───────────┘
```