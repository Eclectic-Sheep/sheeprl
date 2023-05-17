# Register a new algorithm
Suppose that we want to add a new SoTA algorithm to sheeprl called `sota`, so that we can train an agent simply with `python sheeprl.py sota --arg1=... --arg2=...` or accelerated by fabric with `lightning run model sheeprl.py sota --args1=... --arg2=...`.  

We start from creating a new folder called `sota` under `./sheeprl/algos/`, containing the following files:

```bash
algos
└── droq
...
└── sota
    ├── __init__.py
    ├── args.py
    ├── loss.py
    ├── sota.py
    └── utils.py
```

## CLI arguments
To add some CLI arguments to our new algorithm we create the `SOTArgs` in the `args.py` file:

```python
from dataclasses import dataclass

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class SOTArgs(StandardArgs):
    arg1: int = Arg(default=42, help="Help string for arg1")
    arg2: bool = Arg(default=False, help="Help string for arg2")
    ...
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
from dataclasses import asdict
from datetime import datetime

import gymnasium as gym
import torch
from gymnasium.vector import SyncVectorEnv
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Adam
from torchmetrics import MeanMetric

from sheeprl.algos.sota.args import SOTArgs
from sheeprl.algos.sota.loss import loss1, loss2
from sheeprl.algos.sota.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.models.models import MLP
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import make_env


def train(
    fabric: Fabric,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: SOTArgs,
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
def main():
    parser = HfArgumentParser(SOTArgs)
    args: SOTArgs = parser.parse_args_into_dataclasses()[0]

    # Initialize Fabric
    fabric = Fabric()
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(args.seed)

    # Set logger only on rank-0
    if rank == 0:
        run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        logger = TensorBoardLogger(
            root_dir=os.path.join("logs", "sota", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
            name=run_name,
        )
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(asdict(args))

    # Environment setup
    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args.capture_video,
                logger.log_dir if rank == 0 else None,
                "train",
                mask_velocities=args.mask_vel,
                vector_env_idx=i,
            )
            for i in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported")

    # Create the agent model: this should be a torch.nn.Module to be acceleratesd with Fabric
    agent = ...

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = Adam(list(agent.parameters()), lr=args.lr, eps=1e-4)
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
    rb = ReplayBuffer(args.rollout_steps, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_rollout = int(args.num_envs * args.rollout_steps * world_size)
    num_updates = args.total_steps // single_global_rollout if not args.dry_run else 1

    # Linear learning rate scheduler
    if args.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)

    with device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(envs.reset(seed=args.seed)[0], dtype=torch.float32)  # [N_envs, N_obs]
        next_done = torch.zeros(args.num_envs, 1, dtype=torch.float32)  # [N_envs, 1]

    for update in range(1, num_updates + 1):
        for _ in range(0, args.rollout_steps):
            global_step += args.num_envs * world_size

            with torch.no_grad():
                # Sample an action given the observation received by the environment
                # This calls the `forward` method of the PyTorch module, escaping from Fabric
                # because we don't want this to be a synchronization point
                action = agent.module(next_obs)

            # Single environment step
            obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                obs = torch.tensor(obs)  # [N_envs, N_obs]
                rewards = torch.tensor(reward).view(args.num_envs, -1)  # [N_envs, 1]
                done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))  # [N_envs, 1]
                done = done.view(args.num_envs, -1).float()

            # Update the step data
            step_data["dones"] = next_done
            step_data["actions"] = action
            step_data["rewards"] = rewards
            step_data["observations"] = next_obs

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            # Update the observation and done
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
        train(fabric, agent, optimizer, local_data, aggregator, args)

        # Log metrics
        metrics_dict = aggregator.compute()
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

    envs.close()
    if fabric.is_global_zero:
        test(actor.module, envs, fabric, args)


if __name__ == "__main__":
    main()
```

To let the `register_algorithm` decorator add our new `sota` algorithm to the available algorithms registry we need to import it in `./sheeprl/__init__.py`: 

```diff
from dotenv import load_dotenv

from sheeprl.algos.droq import droq
from sheeprl.algos.ppo import ppo, ppo_decoupled
from sheeprl.algos.ppo_continuous import ppo_continuous
from sheeprl.algos.ppo_recurrent import ppo_recurrent
from sheeprl.algos.sac import sac, sac_decoupled
+from sheeprl.algos.sota import sota

try:
    from sheeprl.algos.ppo import ppo_atari
except ModuleNotFoundError:
    pass

load_dotenv()
```

After doing that, when we run `python sheeprl.py` we should see `sota` under the `Commands` section:

```bash
(sheeprl) ➜  fabric_rl git:(master) ✗ python sheeprl.py
Usage: sheeprl.py [OPTIONS] COMMAND [ARGS]...

  Fabric-RL zero-code command line utility.

Options:
  --fabricrl_help  Show this message and exit.

Commands:
  droq
  ppo
  ppo_continuous
  ppo_decoupled
  ppo_recurrent
  sac
  sac_decoupled
  sota
```

While if we run `python sheeprl.py sota -h` we should see the CLI arguments that we have defined in the `args.py`, plus the ones inherited from the `StandardArgs`:

```bash
(sheeprl) ➜  fabric_rl git:(feature/registry) ✗ python sheeprl.py sota -h
UserWarning: This script was launched without the Lightning CLI. Consider to launch the script with `lightning run model ...` to scale it with Fabric
  warnings.warn(
usage: sota.py [-h] [--exp_name EXP_NAME] [--seed SEED] [--dry_run [DRY_RUN]] [--torch_deterministic [TORCH_DETERMINISTIC]]
               [--env_id ENV_ID] [--num_envs NUM_ENVS] [--arg1 ARG1] [--arg2 [ARG2]]

optional arguments:
  -h, --help            show this help message and exit
  --exp_name EXP_NAME   the name of this experiment (default: default)
  --seed SEED           seed of the experiment (default: 42)
  --dry_run [DRY_RUN]   whether to dry-run the script and exit (default: False)
  --torch_deterministic [TORCH_DETERMINISTIC]
                        if toggled, `torch.backends.cudnn.deterministic=True` (default: False)
  --env_id ENV_ID       the id of the environment (default: CartPole-v1)
  --num_envs NUM_ENVS   the number of parallel game environments (default: 4)
  --arg1 ARG1           Help string for arg1 (default: 42)
  --arg2 [ARG2]         Help string for arg2 (default: False)
```