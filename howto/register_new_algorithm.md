# Register a new algorithm
Suppose that we want to add a new SoTA algorithm to sheeprl called `sota` so that we can train an agent simply with `python sheeprl.py exp=sota env=... env.id=...`.  

We start by creating a new folder called `sota` under `./sheeprl/algos/`, containing the following files:

```bash
algos
└── droq
...
└── sota
    ├── __init__.py
    ├── agent.py
    ├── loss.py
    ├── sota.py
    └── utils.py
```

## The agent
The agent is the core of the algorithm and it is defined in the `agent.py` file. It must contain at least single function called `build_agent` that returns at least a tuple composed of two `torch.nn.Module` wrapped with Fabric; the first one is the agent used during the training phase, while the other one is the one used during the environment interaction:

```python
from __future__ import annotations

import copy
from typing import Any, Dict, Sequence, Tuple

import gymnasium
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
import torch
from torch import Tensor

from sheeprl.utils.fabric import get_single_device_fabric


class SOTAAgent(torch.nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tensor:
        ...

class SOTAAgentPlayer(torch.nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tensor:
        ...

    def get_actions(self, obs: Dict[str, torch.Tensor], greedy: bool = False) -> Tensor:
        ...


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    observation_space: gymnasium.spaces.Dict,
    state: Dict[str, Any] | None = None,
) -> Tuple[_FabricModule, _FabricModule]:

    # Define the agent here
    agent = SOTAAgent(...)

    # Load the state from the checkpoint
    if state:
        agent.load_state_dict(state)

    # Setup player agent
    player = copy.deepcopy(agent)

    # Setup the agent with Fabric
    agent = fabric.setup_model(agent)

    # Setup the player agent with a single-device Fabric
    fabric_player = get_single_device_fabric(fabric)
    player = fabric_player.setup_module(player)

    # Tie weights between the agent and the player
    for agent_p, player_p in zip(agent.parameters(), player.parameters()):
        player_p.data = agent_p.data
    return agent, player
```

The player agent is wrapped with a **single-device Fabric**, in this way we maintain the same precision and device of the main Fabric object, but with the player agent being able to interct with the environment skipping possible distributed synchronization points.  

If the agent is composed of multiple models, each one with its own forward method, it is advisable to wrap each one of them with the main Fabric object; the same happens for the player agent, where each of the models has to be consequently wrapped with the single-device Fabric obejct. Here we have the example of the **PPOAgent**:

```python
from __future__ import annotations

import copy
from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium
import hydra
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, OneHotCategorical

from sheeprl.models.models import MLP, MultiEncoder, NatureCNN
from sheeprl.utils.fabric import get_single_device_fabric


class CNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features_dim: int,
        screen_size: int,
        keys: Sequence[str],
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = (in_channels, screen_size, screen_size)
        self.output_dim = features_dim
        self.model = NatureCNN(in_channels=in_channels, features_dim=features_dim, screen_size=screen_size)

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-3)
        return self.model(x)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_dim: int | None,
        keys: Sequence[str],
        dense_units: int = 64,
        mlp_layers: int = 2,
        dense_act: nn.Module = nn.ReLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.input_dim = input_dim
        self.output_dim = features_dim if features_dim else dense_units
        self.model = MLP(
            input_dim,
            features_dim,
            [dense_units] * mlp_layers,
            activation=dense_act,
            norm_layer=[nn.LayerNorm for _ in range(mlp_layers)] if layer_norm else None,
            norm_args=[{"normalized_shape": dense_units} for _ in range(mlp_layers)] if layer_norm else None,
        )

    def forward(self, obs: Dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in self.keys], dim=-1)
        return self.model(x)


class PPOActor(nn.Module):
    def __init__(self, actor_backbone: torch.nn.Module, actor_heads: torch.nn.ModuleList, is_continuous: bool) -> None:
        super().__init__()
        self.actor_backbone = actor_backbone
        self.actor_heads = actor_heads
        self.is_continuous = is_continuous

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.actor_backbone(x)
        return [head(x) for head in self.actor_heads]


class PPOAgent(nn.Module):
    def __init__(
        self,
        actions_dim: Sequence[int],
        obs_space: gymnasium.spaces.Dict,
        encoder_cfg: Dict[str, Any],
        actor_cfg: Dict[str, Any],
        critic_cfg: Dict[str, Any],
        cnn_keys: Sequence[str],
        mlp_keys: Sequence[str],
        screen_size: int,
        distribution_cfg: Dict[str, Any],
        is_continuous: bool = False,
    ):
        super().__init__()
        self.is_continuous = is_continuous
        self.distribution_cfg = distribution_cfg
        self.actions_dim = actions_dim
        in_channels = sum([prod(obs_space[k].shape[:-2]) for k in cnn_keys])
        mlp_input_dim = sum([obs_space[k].shape[0] for k in mlp_keys])
        cnn_encoder = (
            CNNEncoder(in_channels, encoder_cfg.cnn_features_dim, screen_size, cnn_keys)
            if cnn_keys is not None and len(cnn_keys) > 0
            else None
        )
        mlp_encoder = (
            MLPEncoder(
                mlp_input_dim,
                encoder_cfg.mlp_features_dim,
                mlp_keys,
                encoder_cfg.dense_units,
                encoder_cfg.mlp_layers,
                hydra.utils.get_class(encoder_cfg.dense_act),
                encoder_cfg.layer_norm,
            )
            if mlp_keys is not None and len(mlp_keys) > 0
            else None
        )
        self.feature_extractor = MultiEncoder(cnn_encoder, mlp_encoder)
        features_dim = self.feature_extractor.output_dim
        self.critic = MLP(
            input_dims=features_dim,
            output_dim=1,
            hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
            activation=hydra.utils.get_class(critic_cfg.dense_act),
            norm_layer=[nn.LayerNorm for _ in range(critic_cfg.mlp_layers)] if critic_cfg.layer_norm else None,
            norm_args=(
                [{"normalized_shape": critic_cfg.dense_units} for _ in range(critic_cfg.mlp_layers)]
                if critic_cfg.layer_norm
                else None
            ),
        )
        actor_backbone = (
            MLP(
                input_dims=features_dim,
                output_dim=None,
                hidden_sizes=[actor_cfg.dense_units] * actor_cfg.mlp_layers,
                activation=hydra.utils.get_class(actor_cfg.dense_act),
                flatten_dim=None,
                norm_layer=[nn.LayerNorm] * actor_cfg.mlp_layers if actor_cfg.layer_norm else None,
                norm_args=(
                    [{"normalized_shape": actor_cfg.dense_units} for _ in range(actor_cfg.mlp_layers)]
                    if actor_cfg.layer_norm
                    else None
                ),
            )
            if actor_cfg.mlp_layers > 0
            else nn.Identity()
        )
        if is_continuous:
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, sum(actions_dim) * 2)])
        else:
            actor_heads = nn.ModuleList([nn.Linear(actor_cfg.dense_units, action_dim) for action_dim in actions_dim])
        self.actor = PPOActor(actor_backbone, actor_heads, is_continuous)

    def forward(
        self, obs: Dict[str, Tensor], actions: Optional[List[Tensor]] = None
    ) -> Tuple[Sequence[Tensor], Tensor, Tensor, Tensor]:
        feat = self.feature_extractor(obs)
        actor_out: List[Tensor] = self.actor(feat)
        values = self.critic(feat)
        if self.is_continuous:
            mean, log_std = torch.chunk(actor_out[0], chunks=2, dim=-1)
            std = log_std.exp()
            normal = Independent(Normal(mean, std), 1)
            if actions is None:
                actions = normal.sample()
            else:
                # always composed by a tuple of one element containing all the
                # continuous actions
                actions = actions[0]
            log_prob = normal.log_prob(actions)
            return tuple([actions]), log_prob.unsqueeze(dim=-1), normal.entropy().unsqueeze(dim=-1), values
        else:
            should_append = False
            actions_logprobs: List[Tensor] = []
            actions_entropies: List[Tensor] = []
            actions_dist: List[Distribution] = []
            if actions is None:
                should_append = True
                actions: List[Tensor] = []
            for i, logits in enumerate(actor_out):
                actions_dist.append(OneHotCategorical(logits=logits))
                actions_entropies.append(actions_dist[-1].entropy())
                if should_append:
                    actions.append(actions_dist[-1].sample())
                actions_logprobs.append(actions_dist[-1].log_prob(actions[i]))
            return (
                tuple(actions),
                torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True),
                torch.stack(actions_entropies, dim=-1).sum(dim=-1, keepdim=True),
                values,
            )


class PPOPlayer(nn.Module):
    def __init__(self, feature_extractor: MultiEncoder, actor: PPOActor, critic: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.critic = critic
        self.actor = actor

    def forward(self, obs: Dict[str, Tensor]) -> Tuple[Sequence[Tensor], Tensor, Tensor]:
        feat = self.feature_extractor(obs)
        values = self.critic(feat)
        actor_out: List[Tensor] = self.actor(feat)
        if self.actor.is_continuous:
            mean, log_std = torch.chunk(actor_out[0], chunks=2, dim=-1)
            std = log_std.exp()
            normal = Independent(Normal(mean, std), 1)
            actions = normal.sample()
            log_prob = normal.log_prob(actions)
            return tuple([actions]), log_prob.unsqueeze(dim=-1), values
        else:
            actions_dist: List[Distribution] = []
            actions_logprobs: List[Tensor] = []
            actions: List[Tensor] = []
            for i, logits in enumerate(actor_out):
                actions_dist.append(OneHotCategorical(logits=logits))
                actions.append(actions_dist[-1].sample())
                actions_logprobs.append(actions_dist[-1].log_prob(actions[i]))
            return (
                tuple(actions),
                torch.stack(actions_logprobs, dim=-1).sum(dim=-1, keepdim=True),
                values,
            )

    def get_values(self, obs: Dict[str, Tensor]) -> Tensor:
        feat = self.feature_extractor(obs)
        return self.critic(feat)

    def get_actions(self, obs: Dict[str, Tensor], greedy: bool = False) -> Sequence[Tensor]:
        feat = self.feature_extractor(obs)
        actor_out: List[Tensor] = self.actor(feat)
        if self.actor.is_continuous:
            mean, log_std = torch.chunk(actor_out[0], chunks=2, dim=-1)
            if greedy:
                actions = mean
            else:
                std = log_std.exp()
                normal = Independent(Normal(mean, std), 1)
                actions = normal.sample()
            return tuple([actions])
        else:
            actions: List[Tensor] = []
            actions_dist: List[Distribution] = []
            for logits in actor_out:
                actions_dist.append(OneHotCategorical(logits=logits))
                if greedy:
                    actions.append(actions_dist[-1].mode)
                else:
                    actions.append(actions_dist[-1].sample())
            return tuple(actions)


def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
    agent_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[PPOAgent, PPOPlayer]:
    agent = PPOAgent(
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
    if agent_state:
        agent.load_state_dict(agent_state)

    # Setup player agent
    player = PPOPlayer(copy.deepcopy(agent.feature_extractor), copy.deepcopy(agent.actor), copy.deepcopy(agent.critic))

    # Setup training agent
    agent.feature_extractor = fabric.setup_module(agent.feature_extractor)
    agent.critic = fabric.setup_module(agent.critic)
    agent.actor = fabric.setup_module(agent.actor)

    # Setup player agent
    fabric_player = get_single_device_fabric(fabric)
    player.feature_extractor = fabric_player.setup_module(player.feature_extractor)
    player.critic = fabric_player.setup_module(player.critic)
    player.actor = fabric_player.setup_module(player.actor)

    # Tie weights between the agent and the player
    for agent_p, player_p in zip(agent.feature_extractor.parameters(), player.feature_extractor.parameters()):
        player_p.data = agent_p.data
    for agent_p, player_p in zip(agent.actor.parameters(), player.actor.parameters()):
        player_p.data = agent_p.data
    for agent_p, player_p in zip(agent.critic.parameters(), player.critic.parameters()):
        player_p.data = agent_p.data
    return agent, player
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
from lightning.fabric import Fabric
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.sota.agent import build_agent
from sheeprl.algos.sota.loss import loss1, loss2
from sheeprl.algos.sota.utils import normalize_obs, test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.logger import get_logger, get_log_dir
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import unwrap_fabric


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
def sota_main(fabric: Fabric, cfg: Dict[str, Any]):
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
    fabric.print(f"Log dir: {log_dir}")

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
    agent, player = build_agent(
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
                    actions = player.get_actions(torch_obs)
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
        from sheeprl.algos.sota.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"agent": agent}
        register_model(fabric, log_models, cfg, models_to_log)
```

where `log_models`, `test` and `normalize_obs` have to be defined in the `sheeprl.algo.sota.utils` module, for example like this: 

```python
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict

import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule

from sheeprl.algos.sota.agent import SOTAAgentPlayer
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


@torch.no_grad()
def test(agent: SOTAAgentPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
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
        actions = agent.get_actions(obs, greedy=True)
        if agent.is_continuous:
            actions = torch.cat(actions, dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

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
Each algorithm logs its own metrics, during training or environment interaction. To define which are the metrics that can be logged, you need to define the `AGGREGATOR_KEYS` variable in the `./sheeprl/algos/sota/utils.py` file. It must be a set of strings (the name of the metrics to log). Then, you can decide which metrics to log by defining the `metric.aggregator.metrics` in the configs.

> **Remember**
>
> The intersection between the keys in the `AGGREGATOR_KEYS` and the ones in the `metric.aggregator.metrics` config will be logged.

As for metrics, you have to specify which are the models that can be registered after training, you need to define the `MODELS_TO_REGISTER` variable in the `./sheeprl/algos/sota/utils.py` file. It must be a set of strings (the name of the variables of the models you want to register). As before, you can easily select which agents to register by defining the `model_manager.models` in the configs. Also in this case, the models that will be registered are the intersection between the `MODELS_TO_REGISTER` variable and the keys of the `model_manager.models` config.

In this case, the `./sheeprl/algos/sota/utils.py` file could be defined as below:

```python
# `./sheeprl/algos/sota/utils.py`

...

AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/loss1", "Loss/loss2"}
MODELS_TO_REGISTER = {"agent"}

...
```

## Config files
Once you have written your algorithm, you need to create two config files: one in `./sheeprl/configs/algo` and the other in `./sheeprl/configs/exp`.


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

#### Algo Configs
In the `./sheeprl/configs/algo/sota.yaml` we need to specify all the configs needed to initialize and train your agent.
Here is an example of the `./sheeprl/configs/algo/sota.yaml` config file:

```yaml
defaults:
  - default
  - /optim@optimizer: adam
  - _self_

name: sota  # This must be set! And must be equal to the name of the file.py, found under the `./sheeprl/algos/sota/` folder, where the implementation of the algorithm is defined

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
> The field `algo.name` **must** be set and **must** be equal to the name of the file.py, found under the `sheeprl/algos/sota` folder, where the implementation of the algorithm is defined. For example, if your implementation is defined in a python file named `my_sota.py`, i.e. `sheeprl/algos/sota/my_sota.py`, then `algo.name="my_sota"` 

#### Model Manager Configs
In the `./sheeprl/configs/model_manager/sota.yaml` we need to specify all the configs needed to register your agent. You can specify a name, a description, and some tags for each model you want to register. The `disabled` parameter indicates whether or not you want to register your models.
Here is an example of the `./sheeprl/configs/model_manager/sota.yaml` config file:

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
Here is an example of the `./sheeprl/configs/exp/sota.yaml` config file:

```yaml
# @package _global_

defaults:
  - override /algo: sota
  - override /env: atari
  # select the model manager configs
  - override /model_manager: sota
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

With `override /algo: sota` in `defaults` you are specifying you want to use the new `sota` algorithm, whereas, with `override /env: atari` you are specifying that you want to train your agent on an *Atari* environment.

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