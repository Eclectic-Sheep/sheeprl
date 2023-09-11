import copy
import itertools
import os
import time
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
from torchmetrics import MeanMetric

from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent
from sheeprl.algos.ppo_recurrent.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay


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
    initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)
    initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)

    if cfg.buffer.share_data:
        warnings.warn("The script has been called with --share-data: with recurrent PPO only gradients are shared")

    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

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
                cfg.env.id,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank,
                cfg.env.capture_video,
                logger.log_dir if rank == 0 else None,
                "train",
                mask_velocities="mask_velocities" in cfg.env and cfg.env.mask_velocities,
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported by the PPO recurrent agent")
    if len(envs.single_observation_space.shape) > 1:
        raise ValueError(
            "Only environments with vector-only observations are supported by the PPO recurrent agent. "
            f"Provided environment: {cfg.env.id}"
        )

    # Define the agent and the optimizer and setup them with Fabric
    obs_dim = prod(envs.single_observation_space.shape)
    agent = fabric.setup_module(
        RecurrentPPOAgent(
            observation_dim=obs_dim,
            action_dim=envs.single_action_space.n,
            lstm_hidden_size=cfg.algo.lstm.hidden_size,
            actor_hidden_size=cfg.algo.actor.dense_units,
            actor_pre_lstm_hidden_size=cfg.algo.actor.pre_lstm_hidden_size,
            critic_hidden_size=cfg.algo.critic.dense_units,
            critic_pre_lstm_hidden_size=cfg.algo.critic.pre_lstm_hidden_size,
            num_envs=cfg.env.num_envs,
        )
    )
    optimizer = fabric.setup_optimizers(hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters()))

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
                "Game/ep_len_avg": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
                "Time/step_per_second": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
                "Loss/value_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
                "Loss/policy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
                "Loss/entropy_loss": MeanMetric(sync_on_compute=cfg.metric.sync_on_compute),
            }
        )

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
    policy_step = 0
    last_log = 0
    last_checkpoint = 0
    start_time = time.perf_counter()
    policy_steps_per_update = int(cfg.env.num_envs * cfg.algo.rollout_steps * world_size)
    num_updates = cfg.total_steps // policy_steps_per_update if not cfg.dry_run else 1
    last_log = 0

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

    with device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(envs.reset(seed=cfg.seed)[0], dtype=torch.float32).unsqueeze(0)  # [1, N_envs, N_obs]
        next_done = torch.zeros(1, cfg.env.num_envs, 1, dtype=torch.float32)  # [1, N_envs, 1]
        next_state = agent.initial_states

    for update in range(1, num_updates + 1):
        for _ in range(0, cfg.algo.rollout_steps):
            policy_step += cfg.env.num_envs * world_size

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
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, N_envs, N_obs]
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
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: policy_step={policy_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                        aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

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

        # Prepare data
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
        train(fabric, agent, optimizer, padded_sequences, aggregator, cfg)

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
        if policy_step - last_log >= cfg.metric.log_every or cfg.dry_run:
            last_log = policy_step
            metrics_dict = aggregator.compute()
            fabric.log("Time/step_per_second", int(policy_step / (time.perf_counter() - start_time)), policy_step)
            fabric.log_dict(metrics_dict, policy_step)
            aggregator.reset()

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
                "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            cfg.env.id,
            None,
            0,
            cfg.env.capture_video,
            fabric.logger.log_dir,
            "test",
            mask_velocities="mask_velocities" in cfg.env and cfg.env.mask_velocities,
            vector_env_idx=0,
        )()
        test(agent.module, test_env, fabric, cfg)


if __name__ == "__main__":
    main()
