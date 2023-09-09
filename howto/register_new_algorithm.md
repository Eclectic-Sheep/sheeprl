# Register a new algorithm
Suppose that we want to add a new SoTA algorithm to sheeprl called `sota`, so that we can train an agent simply with `python sheeprl.py sota exp=... env=... env.id=...` or accelerated by fabric with `lightning run model sheeprl.py sota exp=... env=... env.id=...`.  

We start from creating a new folder called `sota` under `./sheeprl/algos/`, containing the following files:

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
from torchmetrics import MeanMetric

from sheeprl.algos.sota.loss import loss1, loss2
from sheeprl.algos.sota.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.models.models import MLP
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.env import make_dict_env
from sheeprl.utils.logger import create_tensorboard_logger


def train(
    fabric: Fabric,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    cfg: DictConfig,
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


@register_algorithm()
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize Fabric
    fabric = Fabric()
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg, "sota")
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_dict_env(
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

    # Create the agent model: this should be a torch.nn.Module to be acceleratesd with Fabric
    agent = ...

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=list(agent.parameters()))
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Create a metric aggregator to log the metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(),
                "Game/ep_len_avg": MeanMetric(),
                "Time/step_per_second": MeanMetric(),
                "Loss/value_loss": MeanMetric(),
                "Loss/policy_loss": MeanMetric(),
                "Loss/entropy_loss": MeanMetric(),
            }
        )

    # Local data
    rb = ReplayBuffer(cfg.algo.rollout_steps, cfg.env.num_envs, device=device, memmap=cfg.buffer.memmap)
    step_data = TensorDict({}, batch_size=[cfg.env.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.perf_counter()
    single_global_rollout = int(cfg.env.num_envs * cfg.algo.rollout_steps * world_size)
    num_updates = cfg.total_steps // single_global_rollout if not cfg.dry_run else 1

    # Linear learning rate scheduler
    if cfg.algo.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)

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
    next_done = torch.zeros(cfg.env.num_envs, 1, dtype=torch.float32)  # [N_envs, 1]

    for update in range(1, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            global_step += cfg.env.num_envs * world_size

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
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                        aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        # Train the agent
        train(fabric, agent, optimizer, local_data, aggregator, cfg)

        # Log metrics
        metrics_dict = aggregator.compute()
        fabric.log("Time/step_per_second", int(global_step / (time.perf_counter() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

    envs.close()
    if fabric.is_global_zero:
        test(actor.module, envs, fabric, cfg)


if __name__ == "__main__":
    main()
```

## Config files
Once you have written your algorithm, you need to create two configs file: one in `./sheeprl/configs/algo` and the other in `./sheeprl/configs/exp`.

> **Note**
>
> The name of the two file should be the same of the algorithm, so in our case it is `sota.yaml`

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
Here an example of the `./sheeprl/configs/algo/sota.yaml` config file:

```yaml
defaults:
  - default
  - /optim@optimizer: adam
  - _self_

name: sota

anneal_lr: False
gamma: 0.99
gae_lambda: 0.95
update_epochs: 10
loss_reduction: mean
normalize_advantages: False
clip_coef: 0.2
anneal_clip_coef: False
clip_vloss: False
ent_coef: 0.0
anneal_ent_coef: False
vf_coef: 1.0

dense_units: 64
mlp_layers: 2
dense_act: torch.nn.Tanh
layer_norm: False
max_grad_norm: 0.0
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
Will add two optimizers, one accesible with `algo.encoder.optimizer`, the other with `algo.actor.optimizer`.

#### Experiment config
In the second file you have to specify all the elements you want in your experiment and you can override all the parameters you want.
Here an example of the `./sheeprl/configs/exp/sota.yaml` config file:

```yaml
# @package _global_

defaults:
  - override /algo: sota
  - override /env: atari
  - _self_

per_rank_batch_size: 64
total_steps: 65536
buffer:
  share_data: False
rollout_steps: 128

# override environment id
env:
  env:
    id: MsPacmanNoFrameskip-v4
```

With `override /algo: sota` in `defaults` you are specifing you want to use the new `sota` algorithm, whereas, with `override /env: gym` you are specifing that you want to train your agent on an *Atari* environment.

## Register Algorithm

To let the `register_algorithm` decorator add our new `sota` algorithm to the available algorithms registry we need to import it in `./sheeprl/__init__.py`: 

```diff
from dotenv import load_dotenv

from sheeprl.algos.ppo import ppo, ppo_decoupled
from sheeprl.algos.ppo_recurrent import ppo_recurrent
from sheeprl.algos.sac import sac, sac_decoupled
from sheeprl.algos.sac_ae import sac_ae
+from sheeprl.algos.sota import sota

try:
    from sheeprl.algos.ppo import ppo_atari
except ModuleNotFoundError:
    pass

load_dotenv()
```

After doing that, when we run `python sheeprl.py` we should see `sota` under the `Commands` section:

```bash
(sheeprl) ➜  sheeprl git:(main) ✗ python sheeprl.py
Usage: sheeprl.py [OPTIONS] COMMAND [ARGS]...

  SheepRL zero-code command line utility.

Options:
  --sheeprl_help  Show this message and exit.

Commands:
  dreamer_v1
  dreamer_v2
  dreamer_v3
  droq
  p2e_dv1
  p2e_dv2
  ppo
  ppo_decoupled
  ppo_recurrent
  sac
  sac_ae
  sac_decoupled
  sota
```