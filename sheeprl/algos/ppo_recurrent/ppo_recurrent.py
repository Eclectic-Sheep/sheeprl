import copy
import itertools
import os
import pathlib
import warnings
from contextlib import nullcontext
from math import prod
from typing import List

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase, pad_sequence
from torch.distributed.algorithms.join import Join
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torchmetrics import MeanMetric, SumMetric

from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent
from sheeprl.algos.ppo_recurrent.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay, print_config


def train(
    fabric: Fabric,
    agent: RecurrentPPOAgent,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    cfg: DictConfig,
):
    num_sequences = data.shape[1]
    if cfg.per_rank_num_batches > 0:
        batch_size = num_sequences // cfg.per_rank_num_batches
        batch_size = batch_size if batch_size > 0 else num_sequences
    else:
        batch_size = 1
    with Join([agent._forward_module]) if fabric.world_size > 1 else nullcontext():
        for _ in range(cfg.algo.update_epochs):
            states = ((data["actor_hxs"], data["actor_cxs"]), (data["critic_hxs"], data["critic_cxs"]))
            sampler = BatchSampler(
                RandomSampler(range(num_sequences)),
                batch_size=batch_size,
                drop_last=False,
            )  # Random sampling sequences
            for idxes in sampler:
                batch = data[:, idxes]
                mask = batch["mask"].unsqueeze(-1)
                action_logits, new_values, _ = agent(
                    batch["observations"],
                    state=tuple([tuple([s[:1, idxes] for s in state]) for state in states]),
                    mask=mask,
                )
                dist = Categorical(logits=action_logits.unsqueeze(-2))

                normalized_advantages = batch["advantages"][mask]
                if cfg.algo.normalize_advantages and len(normalized_advantages) > 1:
                    normalized_advantages = normalize_tensor(normalized_advantages)

                # Policy loss
                pg_loss = policy_loss(
                    dist.log_prob(batch["actions"])[mask],
                    batch["logprobs"][mask],
                    normalized_advantages,
                    cfg.algo.clip_coef,
                    "mean",
                )

                # Value loss
                v_loss = value_loss(
                    new_values[mask],
                    batch["values"][mask],
                    batch["returns"][mask],
                    cfg.algo.clip_coef,
                    cfg.algo.clip_vloss,
                    "mean",
                )

                # Entropy loss
                ent_loss = entropy_loss(dist.entropy()[mask], "mean")

                # Equation (9) in the paper
                loss = pg_loss + cfg.algo.vf_coef * v_loss + cfg.algo.ent_coef * ent_loss

                optimizer.zero_grad(set_to_none=True)
                fabric.backward(loss)
                if cfg.algo.max_grad_norm > 0.0:
                    fabric.clip_gradients(agent, optimizer, max_norm=cfg.algo.max_grad_norm)
                optimizer.step()

                # Update metrics
                aggregator.update("Loss/policy_loss", pg_loss.detach())
                aggregator.update("Loss/value_loss", v_loss.detach())
                aggregator.update("Loss/entropy_loss", ent_loss.detach())


@register_algorithm(decoupled=True)
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print_config(cfg)
    initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)
    initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)

    if "minedojo" in cfg.env.env._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by PPO Recurrent agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    if cfg.buffer.share_data:
        warnings.warn(
            "The script has been called with `buffer.share_data=True`: with recurrent PPO only gradients are shared"
        )

    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        root_dir = cfg.root_dir
        run_name = cfg.run_name
        state = fabric.load(cfg.checkpoint.resume_from)
        ckpt_path = pathlib.Path(cfg.checkpoint.resume_from)
        cfg = OmegaConf.load(ckpt_path.parent.parent.parent / ".hydra" / "config.yaml")
        cfg.checkpoint.resume_from = str(ckpt_path)
        cfg.per_rank_batch_size = state["batch_size"] // fabric.world_size
        cfg.root_dir = root_dir
        cfg.run_name = run_name

    if len(cfg.cnn_keys.encoder) > 0:
        warnings.warn(
            "PPO recurrent algorithm cannot allow to use images as observations, the CNN keys will be ignored"
        )
        cfg.cnn_keys.encoder = []

    # Create TensorBoardLogger. This will create the logger only on the
    # rank-0 process
    logger, log_dir = create_tensorboard_logger(fabric, cfg, "ppo_recurrent")
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

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
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    if not isinstance(action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported by the PPO Recurrent agent")
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the PPO "
                "Recurrent agent. "
                f"Provided environment: {cfg.env.id}"
            )

    # Define the agent and the optimizer
    obs_dim = sum([prod(observation_space[k].shape) for k in cfg.mlp_keys.encoder])
    agent = RecurrentPPOAgent(
        observation_dim=obs_dim,
        action_dim=action_space.n,
        lstm_hidden_size=cfg.algo.lstm.hidden_size,
        actor_hidden_size=cfg.algo.actor.dense_units,
        actor_pre_lstm_hidden_size=cfg.algo.actor.pre_lstm_hidden_size,
        critic_hidden_size=cfg.algo.critic.dense_units,
        critic_pre_lstm_hidden_size=cfg.algo.critic.pre_lstm_hidden_size,
        num_envs=cfg.env.num_envs,
    )
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters())

    # Load the state from the checkpoint
    if cfg.checkpoint.resume_from:
        agent.load_state_dict(state["agent"])
        optimizer.load_state_dict(state["optimizer"])

    # Setup agent and optimizer with Fabric
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Create a metric aggregator to log the metrics
    aggregator = MetricAggregator(
        {
            "Rewards/rew_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Game/ep_len_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/value_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/policy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            "Loss/entropy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
        }
    ).to(device)

    # Local data
    rb = ReplayBuffer(
        cfg.algo.rollout_steps,
        cfg.env.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    step_data = TensorDict({}, batch_size=[1, cfg.env.num_envs], device=device)

    # Global variables
    last_train = 0
    train_step = 0
    start_step = state["update"] // fabric.world_size if cfg.checkpoint.resume_from else 1
    policy_step = state["update"] * cfg.env.num_envs * cfg.algo.rollout_steps if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
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

    # Linear learning rate scheduler
    if cfg.algo.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)
        if cfg.checkpoint.resume_from:
            scheduler.load_state_dict(state["scheduler"])

    with device:
        # Get the first environment observation and start the optimization
        next_state = agent.initial_states
        next_done = torch.zeros(1, cfg.env.num_envs, 1, dtype=torch.float32)  # [1, N_envs, 1]
        o = envs.reset(seed=cfg.seed)[0]
        next_obs = torch.cat([torch.tensor(o[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1).unsqueeze(
            0
        )

    for update in range(start_step, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs * world_size

            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric(sync_on_compute=False)):
                with torch.no_grad():
                    # Sample an action given the observation received by the environment
                    action_logits, values, state = agent.module(next_obs, state=next_state)
                    dist = Categorical(logits=action_logits.unsqueeze(-2))
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                # Single environment step
                obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))
                done = np.logical_or(done, truncated)

            with device:
                obs = torch.cat(
                    [torch.tensor(obs[k], dtype=torch.float32) for k in cfg.mlp_keys.encoder], dim=-1
                ).unsqueeze(0)
                done = torch.tensor(done, dtype=torch.float32).view(1, cfg.env.num_envs, -1)  # [1, N_envs, 1]
                reward = torch.tensor(reward, dtype=torch.float32).view(1, cfg.env.num_envs, -1)  # [1, N_envs, 1]

            step_data["dones"] = next_done
            step_data["values"] = values
            step_data["actions"] = action
            step_data["rewards"] = reward
            step_data["logprobs"] = logprob
            step_data["observations"] = next_obs
            step_data["actor_hxs"] = next_state[0][0]
            step_data["actor_cxs"] = next_state[0][1]
            step_data["critic_hxs"] = next_state[1][0]
            step_data["critic_cxs"] = next_state[1][1]
            if cfg.buffer.memmap:
                step_data["returns"] = torch.zeros_like(reward)
                step_data["advantages"] = torch.zeros_like(reward)

            # Append data to buffer
            rb.add(step_data)

            # Update observation, done and recurrent state
            next_obs = obs
            next_done = done
            if cfg.algo.reset_recurrent_state_on_done:
                next_state = tuple([tuple([(1 - done) * e for e in s]) for s in state])
            else:
                next_state = state

            if "final_info" in info:
                for i, agent_ep_info in enumerate(info["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.no_grad():
            next_value, _ = agent.module.get_values(next_obs, critic_state=next_state[1])
            returns, advantages = gae(
                rb["rewards"],
                rb["values"],
                rb["dones"],
                next_value,
                next_done,
                cfg.algo.rollout_steps,
                cfg.algo.gamma,
                cfg.algo.gae_lambda,
            )

            # Add returns and advantages to the buffer
            rb["returns"] = returns.float()
            rb["advantages"] = advantages.float()

        # Get the training data as a TensorDict
        local_data = rb.buffer

        # Train the agent
        # 1. Split data into episodes (for every environment)
        episodes: List[TensorDictBase] = []
        for env_id in range(cfg.env.num_envs):
            env_data = local_data[:, env_id]  # [N_steps, *]
            episode_ends = env_data["dones"].nonzero(as_tuple=True)[0]
            episode_ends = episode_ends.tolist()
            episode_ends.append(cfg.algo.rollout_steps)
            start = 0
            for ep_end_idx in episode_ends:
                stop = ep_end_idx
                # Do not include the done, since when we encounter a done it means that
                # the episode has started
                episode = env_data[start:stop]
                if len(episode) > 0:
                    episodes.append(episode)
                start = stop
        # 2. Split every episode into sequences of length `per_rank_batch_size`
        if cfg.per_rank_batch_size is not None and cfg.per_rank_batch_size > 0:
            sequences = list(itertools.chain.from_iterable([ep.split(cfg.per_rank_batch_size) for ep in episodes]))
        else:
            sequences = episodes
        padded_sequences = pad_sequence(sequences, batch_first=False, return_mask=True)  # [Seq_len, Num_seq, *]

        with timer("Time/train_time", SumMetric(sync_on_compute=cfg.metric.sync_on_compute)):
            train(fabric, agent, optimizer, padded_sequences, aggregator, cfg)
        train_step += world_size

        if cfg.algo.anneal_lr:
            fabric.log("Info/learning_rate", scheduler.get_last_lr()[0], policy_step)
            scheduler.step()
        else:
            fabric.log("Info/learning_rate", cfg.algo.optimizer.lr, policy_step)

        fabric.log("Info/clip_coef", cfg.algo.clip_coef, policy_step)
        if cfg.algo.anneal_clip_coef:
            cfg.algo.clip_coef = polynomial_decay(
                update, initial=initial_clip_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        fabric.log("Info/ent_coef", cfg.algo.ent_coef, policy_step)
        if cfg.algo.anneal_ent_coef:
            cfg.algo.ent_coef = polynomial_decay(
                update, initial=initial_ent_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        # Log metrics
        if policy_step - last_log >= cfg.metric.log_every or update == num_updates or cfg.dry_run:
            # Sync distributed metrics
            metrics_dict = aggregator.compute()
            fabric.log_dict(metrics_dict, policy_step)
            aggregator.reset()

            # Sync distributed timers
            timer_metrics = timer.compute()
            fabric.log(
                "Time/sps_train",
                (train_step - last_train) / timer_metrics["Time/train_time"],
                policy_step,
            )
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
                "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
                "update": update * world_size,
                "batch_size": cfg.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            cfg,
            None,
            0,
            logger.log_dir,
            "test",
            vector_env_idx=0,
        )()
        test(agent.module, test_env, fabric, cfg)


if __name__ == "__main__":
    main()
