import copy
import os
import time
from typing import Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.wrappers import _FabricModule
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch import nn
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import MeanMetric

from sheeprl.algos.ppo.agent import PPOAgent
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_dict_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, normalize_tensor, polynomial_decay


def train(
    fabric: Fabric,
    agent: Union[nn.Module, _FabricModule],
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    cfg: DictConfig,
):
    """Train the agent on the data collected from the environment."""
    indexes = list(range(data.shape[0]))
    if cfg.buffer.share_data:
        sampler = DistributedSampler(
            indexes,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=cfg.seed,
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=cfg.per_rank_batch_size, drop_last=False)

    for epoch in range(cfg.algo.update_epochs):
        if cfg.buffer.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            batch = data[batch_idxes]
            normalized_obs = {
                k: batch[k] / 255 - 0.5 if k in cfg.cnn_keys.encoder else batch[k]
                for k in cfg.mlp_keys.encoder + cfg.cnn_keys.encoder
            }
            _, logprobs, entropy, new_values = agent(
                normalized_obs, torch.split(batch["actions"], agent.actions_dim, dim=-1)
            )

            if cfg.algo.normalize_advantages:
                batch["advantages"] = normalize_tensor(batch["advantages"])

            # Policy loss
            pg_loss = policy_loss(
                logprobs,
                batch["logprobs"],
                batch["advantages"],
                cfg.algo.clip_coef,
                cfg.algo.loss_reduction,
            )

            # Value loss
            v_loss = value_loss(
                new_values,
                batch["values"],
                batch["returns"],
                cfg.algo.clip_coef,
                cfg.algo.clip_vloss,
                cfg.algo.loss_reduction,
            )

            # Entropy loss
            ent_loss = entropy_loss(entropy, cfg.algo.loss_reduction)

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


@register_algorithm()
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    if "minedojo" in cfg.env.env._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by PPO agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    initial_ent_coef = copy.deepcopy(cfg.algo.ent_coef)
    initial_clip_coef = copy.deepcopy(cfg.algo.clip_coef)

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
    logger, log_dir = create_tensorboard_logger(fabric, cfg, "ppo")
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_dict_env(
                cfg,
                cfg.seed + rank * cfg.num_envs + i,
                rank * cfg.num_envs,
                logger.log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.num_envs)
        ]
    )
    observation_space = envs.single_observation_space

    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if cfg.cnn_keys.encoder + cfg.mlp_keys.encoder == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: `--cnn_keys rgb` "
            "or `--mlp_keys state` "
        )
    fabric.print("Encoder CNN keys:", cfg.cnn_keys.encoder)
    fabric.print("Encoder MLP keys:", cfg.mlp_keys.encoder)
    obs_keys = cfg.cnn_keys.encoder + cfg.mlp_keys.encoder

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(envs.single_action_space, gym.spaces.MultiDiscrete)
    actions_dim = (
        envs.single_action_space.shape
        if is_continuous
        else (envs.single_action_space.nvec.tolist() if is_multidiscrete else [envs.single_action_space.n])
    )
    # Create the actor and critic models
    agent = PPOAgent(
        actions_dim=actions_dim,
        obs_space=observation_space,
        encoder_cfg=cfg.algo.encoder,
        actor_cfg=cfg.algo.actor,
        critic_cfg=cfg.algo.critic,
        cnn_keys=cfg.cnn_keys.encoder,
        mlp_keys=cfg.mlp_keys.encoder,
        screen_size=cfg.env.screen_size,
        is_continuous=is_continuous,
    )

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters())
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
    rb = ReplayBuffer(
        cfg.rollout_steps,
        cfg.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=obs_keys,
    )
    step_data = TensorDict({}, batch_size=[cfg.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.perf_counter()
    single_global_rollout = int(cfg.num_envs * cfg.rollout_steps * world_size)
    num_updates = cfg.total_steps // single_global_rollout if not cfg.dry_run else 1

    # Linear learning rate scheduler
    if cfg.algo.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)

    # Get the first environment observation and start the optimization
    o = envs.reset(seed=cfg.seed)[0]  # [N_envs, N_obs]
    next_obs = {}
    for k in o.keys():
        if k in obs_keys:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device)
            if k in cfg.cnn_keys.encoder:
                torch_obs = torch_obs.view(cfg.num_envs, -1, *torch_obs.shape[-2:])
            if k in cfg.mlp_keys.encoder:
                torch_obs = torch_obs.float()
            step_data[k] = torch_obs
            next_obs[k] = torch_obs
    next_done = torch.zeros(cfg.num_envs, 1, dtype=torch.float32).to(fabric.device)  # [N_envs, 1]

    for update in range(1, num_updates + 1):
        for _ in range(0, cfg.rollout_steps):
            global_step += cfg.num_envs * world_size

            with torch.no_grad():
                # Sample an action given the observation received by the environment
                normalized_obs = {
                    k: next_obs[k] / 255 - 0.5 if k in cfg.cnn_keys.encoder else next_obs[k] for k in obs_keys
                }
                actions, logprobs, _, value = agent(normalized_obs)
                if is_continuous:
                    real_actions = torch.cat(actions, -1).cpu().numpy()
                else:
                    real_actions = np.concatenate([act.argmax(dim=-1).cpu().numpy() for act in actions], axis=-1)
                actions = torch.cat(actions, -1)

            # Single environment step
            o, reward, done, truncated, info = envs.step(real_actions)
            done = np.logical_or(done, truncated)

            with device:
                rewards = torch.tensor(reward, dtype=torch.float32).view(cfg.num_envs, -1)  # [N_envs, 1]
                done = torch.tensor(done, dtype=torch.float32).view(cfg.num_envs, -1)  # [N_envs, 1]

            # Update the step data
            step_data["dones"] = next_done
            step_data["values"] = value
            step_data["actions"] = actions
            step_data["logprobs"] = logprobs
            step_data["rewards"] = rewards

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            # Update the observation and done
            obs = {}
            for k in o.keys():
                if k in obs_keys:
                    torch_obs = torch.from_numpy(o[k]).to(fabric.device)
                    if k in cfg.cnn_keys.encoder:
                        torch_obs = torch_obs.view(cfg.num_envs, -1, *torch_obs.shape[-2:])
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

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.no_grad():
            normalized_obs = {
                k: next_obs[k] / 255 - 0.5 if k in cfg.cnn_keys.encoder else next_obs[k] for k in obs_keys
            }
            next_values = agent.get_value(normalized_obs)
            returns, advantages = gae(
                rb["rewards"],
                rb["values"],
                rb["dones"],
                next_values,
                next_done,
                cfg.rollout_steps,
                cfg.algo.gamma,
                cfg.algo.gae_lambda,
            )

            # Add returns and advantages to the buffer
            rb["returns"] = returns.float()
            rb["advantages"] = advantages.float()

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        if cfg.buffer.share_data and fabric.world_size > 1:
            # Gather all the tensors from all the world and reshape them
            gathered_data = fabric.all_gather(local_data.to_dict())  # Fabric does not work with TensorDict
            gathered_data = make_tensordict(gathered_data).view(-1)
        else:
            gathered_data = local_data

        train(fabric, agent, optimizer, gathered_data, aggregator, cfg)

        if cfg.algo.anneal_lr:
            fabric.log("Info/learning_rate", scheduler.get_last_lr()[0], global_step)
            scheduler.step()
        else:
            fabric.log("Info/learning_rate", cfg.algo.optimizer.lr, global_step)

        fabric.log("Info/clip_coef", cfg.algo.clip_coef, global_step)
        if cfg.algo.anneal_clip_coef:
            cfg.algo.clip_coef = polynomial_decay(
                update, initial=initial_clip_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        fabric.log("Info/ent_coef", cfg.algo.ent_coef, global_step)
        if cfg.algo.anneal_ent_coef:
            cfg.algo.ent_coef = polynomial_decay(
                update, initial=initial_ent_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        # Log metrics
        metrics_dict = aggregator.compute()
        fabric.log("Time/step_per_second", int(global_step / (time.perf_counter() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

        # Checkpoint model
        if (cfg.checkpoint_every > 0 and update % cfg.checkpoint_every == 0) or cfg.dry_run or update == num_updates:
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_step": update,
                "scheduler": scheduler.state_dict() if cfg.algo.anneal_lr else None,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{update}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero:
        test_env = make_dict_env(
            cfg,
            None,
            0,
            fabric.logger.log_dir,
            "test",
            vector_env_idx=0,
        )()
        test(agent.module, test_env, fabric, cfg)


if __name__ == "__main__":
    main()
