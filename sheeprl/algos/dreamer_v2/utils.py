from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Sequence

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from lightning import Fabric
from torch import Tensor
from torch.distributions import Independent, OneHotCategoricalStraightThrough

from sheeprl.data.buffers import EnvIndependentReplayBuffer, EpisodeBuffer, SequentialReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

    from sheeprl.algos.dreamer_v1.agent import PlayerDV1
    from sheeprl.algos.dreamer_v2.agent import PlayerDV2


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
MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic"}


def compute_stochastic_state(logits: Tensor, discrete: int = 32, sample=True) -> Tensor:
    """
    Compute the stochastic state from the logits computed by the transition or representaiton model.

    Args:
        logits (Tensor): logits from either the representation model or the transition model.
        discrete (int, optional): the size of the Categorical variables.
            Defaults to 32.
        sample (bool): whether or not to sample the stochastic state.
            Default to True.

    Returns:
        The sampled stochastic state.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    stochastic_state = dist.rsample() if sample else dist.mode
    return stochastic_state


def init_weights(m: nn.Module, mode: str = "uniform"):
    """
    Initialize the parameters of the m module acording to the Xavier method, by default the uniform one (the default
    initializer of the layers of Keras, which the official implementation uses), with zero biases.

    Args:
        m (nn.Module): the module to be initialized.
        mode (str): `uniform`, `normal` or `zero`. Default: `uniform`.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if mode == "normal":
            nn.init.xavier_normal_(m.weight.data)
        elif mode == "uniform":
            nn.init.xavier_uniform_(m.weight.data)
        elif mode == "zero":
            nn.init.constant_(m.weight.data, 0)
        else:
            raise RuntimeError(f"Unrecognized initialization: {mode}. Choose between: `normal`, `uniform` and `zero`")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def build_optimizer(optimizer_cfg: Dict[str, Any], params: Any) -> torch.optim.Optimizer:
    """The optimizer of `optimizer_cfg` for `params`, with the weight decay of DreamerV2: before every step the weights
    are multiplied by `1 - weight_decay`, whatever the learning rate (`Optimizer._apply_weight_decay` of
    https://github.com/danijar/dreamerv2). Adam is built as AdamW with the weight decay divided by the learning rate,
    which decays the weights that way: the weight decay of Adam is added to the gradients, where its normalization makes
    it negligible."""
    optimizer_cfg = dict(optimizer_cfg)
    weight_decay = optimizer_cfg.get("weight_decay") or 0
    if weight_decay > 0 and optimizer_cfg["_target_"] == "torch.optim.Adam":
        optimizer_cfg["_target_"] = "torch.optim.AdamW"
        optimizer_cfg["weight_decay"] = weight_decay / optimizer_cfg["lr"]
    return hydra.utils.instantiate(optimizer_cfg, params=params, _convert_="all")


# The most batches sampled (and moved to the device) at once: the first training can do many gradient steps
# (`algo.per_rank_pretrain_steps`)
MAX_SAMPLED_BATCHES = 16


def sample_batches(
    fabric: Fabric, cfg: Dict[str, Any], buffer: EnvIndependentReplayBuffer | EpisodeBuffer, n_steps: int
) -> Iterator[Dict[str, Tensor]]:
    """The batches of sequences of the `n_steps` gradient steps of an iteration, sampled `MAX_SAMPLED_BATCHES` at a
    time: with `buffer.online`, they start with the sequences of the online queue of the buffer."""
    for first in range(0, n_steps, MAX_SAMPLED_BATCHES):
        n_samples = min(MAX_SAMPLED_BATCHES, n_steps - first)
        sample = buffer.sample_tensors(
            batch_size=cfg.algo.per_rank_batch_size,
            sequence_length=cfg.algo.per_rank_sequence_length,
            n_samples=n_samples,
            dtype=None,
            device=fabric.device,
            from_numpy=cfg.buffer.from_numpy,
            online=cfg.buffer.online,
        )  # [N_Samples, Sequence_Length, Batch_Size, ...]
        for i in range(n_samples):
            yield {k: v[i].float() for k, v in sample.items()}


def env_buffer_size(fabric: Fabric, cfg: Dict[str, Any], dry_run_size: int) -> int:
    """The capacity of the replay buffer of every environment: `buffer.size` split among the environments of all the
    processes, or `dry_run_size` in a dry run. It must hold a sequence of `algo.per_rank_sequence_length` steps (a dry
    run makes it large enough)."""
    sequence_length = cfg.algo.per_rank_sequence_length
    if cfg.dry_run:
        return max(dry_run_size, sequence_length)
    size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size)
    if size < sequence_length:
        raise ValueError(
            f"The replay buffer of every environment holds `buffer.size // (env.num_envs * world_size)` = {size} "
            f"steps, fewer than a sequence (`algo.per_rank_sequence_length={sequence_length}`): increase `buffer.size`"
        )
    return size


def build_buffer(
    fabric: Fabric, cfg: Dict[str, Any], log_dir: str, dry_run_size: int
) -> EnvIndependentReplayBuffer | EpisodeBuffer:
    """The replay buffer of `buffer.type`: one buffer of sequences per environment (`sequential`), or a buffer of
    whole episodes (`episode`), sampled with their ends prioritized with `buffer.prioritize_ends`.

    Every process holds `buffer.size // world_size` steps: in a sequential buffer they are split among the environments
    of the process, while the episodes of all the environments of the process share the episode buffer. In a dry run
    the buffers hold `dry_run_size` steps.
    """
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder
    memmap_dir = os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}")
    buffer_type = cfg.buffer.type.lower()
    if buffer_type == "sequential":
        return EnvIndependentReplayBuffer(
            env_buffer_size(fabric, cfg, dry_run_size),
            n_envs=cfg.env.num_envs,
            obs_keys=obs_keys,
            memmap=cfg.buffer.memmap,
            memmap_dir=memmap_dir,
            buffer_cls=SequentialReplayBuffer,
            seed=cfg.seed + fabric.global_rank,
        )
    elif buffer_type == "episode":
        return EpisodeBuffer(
            cfg.buffer.size // fabric.world_size if not cfg.dry_run else dry_run_size,
            minimum_episode_length=1 if cfg.dry_run else cfg.algo.per_rank_sequence_length,
            n_envs=cfg.env.num_envs,
            obs_keys=obs_keys,
            prioritize_ends=cfg.buffer.prioritize_ends,
            memmap=cfg.buffer.memmap,
            memmap_dir=memmap_dir,
        )
    raise ValueError(f"Unrecognized buffer type: must be one of `sequential` or `episode`, received: {buffer_type}")


def actor_objective(
    objective_mix: Optional[float], is_continuous: bool, dynamics: Tensor, reinforce: Callable[[], Tensor]
) -> Tensor:
    """The objective of the DreamerV2 actor: `objective_mix` times the REINFORCE objective (`reinforce()`) plus
    `1 - objective_mix` times the dynamics backpropagation (`dynamics`, the lambda-values). `None` (the default of
    `algo.actor.objective_mix`): the dynamics for continuous actions, REINFORCE for discrete ones, as DreamerV2 does
    (`actor_grad: auto`)."""
    if objective_mix is None:
        objective_mix = 0.0 if is_continuous else 1.0
    if objective_mix == 0:
        return dynamics
    if objective_mix == 1:
        return reinforce()
    return objective_mix * reinforce() + (1 - objective_mix) * dynamics


def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    bootstrap: Optional[Tensor] = None,
    horizon: int = 15,
    lmbda: float = 0.95,
) -> Tensor:
    if bootstrap is None:
        bootstrap = torch.zeros_like(values[-1:])
    agg = bootstrap
    next_val = torch.cat((values[1:], bootstrap), dim=0)
    inputs = rewards + continues * next_val * (1 - lmbda)
    lv = []
    for i in reversed(range(horizon)):
        agg = inputs[i] + continues[i] * lmbda * agg
        lv.append(agg)
    return torch.cat(list(reversed(lv)), dim=0)


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k, v in obs.items():
        torch_obs[k] = torch.from_numpy(v.copy()).to(fabric.device).float()
        if k in cnn_keys:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1, *v.shape[-2:]) / 255 - 0.5
        else:
            torch_obs[k] = torch_obs[k].view(1, num_envs, -1)

    return torch_obs


@torch.no_grad()
def test(
    player: "PlayerDV2" | "PlayerDV1",
    fabric: Fabric,
    cfg: Dict[str, Any],
    log_dir: str,
    test_name: str = "",
    greedy: bool = True,
    policy_step: int = 0,
):
    """Test the model on the environment with the frozen model.

    Args:
        player (PlayerDV2 | PlayerDV1): the agent which contains all the models needed to play.
        fabric (Fabric): the fabric instance.
        cfg (Dict[str, Any]): the hyper-parameters.
        log_dir (str): the logging directory.
        test_name (str): the name of the test.
            Default to "".
        greedy (bool): whether or not to sample actions.
            Default to True.
    """
    env: gym.Env = make_env(cfg, cfg.seed, 0, log_dir, "test" + (f"_{test_name}" if test_name != "" else ""))()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]
    player.num_envs = 1
    player.init_states()
    while not done:
        # Act greedly through the environment
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        real_actions = player.get_actions(
            torch_obs, greedy, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        )
        if player.actor.is_continuous:
            real_actions = torch.stack(real_actions, -1).cpu().numpy()
        else:
            real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

        # Single environment step
        obs, reward, done, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
        done = done or truncated or cfg.dry_run
        cumulative_rew += reward
    fabric.print("Test - Reward:", cumulative_rew)
    if cfg.metric.log_level > 0 and len(fabric.loggers) > 0:
        fabric.logger.log_metrics({"Test/cumulative_reward": cumulative_rew}, policy_step)
    env.close()


def log_models_from_checkpoint(
    fabric: Fabric, env: gym.Env | gym.Wrapper, cfg: Dict[str, Any], state: Dict[str, Any]
) -> Sequence["ModelInfo"]:
    if not _IS_MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_IS_MLFLOW_AVAILABLE))
    import mlflow  # noqa

    from sheeprl.algos.dreamer_v2.agent import build_agent

    # Create the models
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    world_model, actor, critic, target_critic, _ = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        env.observation_space,
        state["world_model"],
        state["actor"],
        state["critic"],
        state["target_critic"],
    )

    # Log the model, create a new run if `cfg.run_id` is None.
    model_info = {}
    with mlflow.start_run(run_id=cfg.run.id, experiment_id=cfg.experiment.id, run_name=cfg.run.name, nested=True) as _:
        model_info["world_model"] = mlflow.pytorch.log_model(
            unwrap_fabric(world_model), name="world_model", serialization_format="pickle"
        )
        model_info["actor"] = mlflow.pytorch.log_model(
            unwrap_fabric(actor), name="actor", serialization_format="pickle"
        )
        model_info["critic"] = mlflow.pytorch.log_model(
            unwrap_fabric(critic), name="critic", serialization_format="pickle"
        )
        model_info["target_critic"] = mlflow.pytorch.log_model(
            unwrap_fabric(target_critic), name="target_critic", serialization_format="pickle"
        )
        mlflow.log_dict(cfg.to_log, "config.json")
    return model_info
