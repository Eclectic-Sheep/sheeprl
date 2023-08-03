import copy
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Union

import gymnasium as gym
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.wrappers import _FabricModule
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch import nn
from torch.optim import Adam
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import MeanMetric

from sheeprl.algos.ppo.agent import PPOAgent
from sheeprl.algos.ppo.args import PPOArgs
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, make_dict_env, normalize_tensor, polynomial_decay


def train(
    fabric: Fabric,
    agent: Union[nn.Module, _FabricModule],
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: PPOArgs,
    cnn_keys,
    mlp_keys,
):
    """Train the agent on the data collected from the environment."""
    indexes = list(range(data.shape[0]))
    if args.share_data:
        sampler = DistributedSampler(
            indexes,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=args.seed,
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)

    for epoch in range(args.update_epochs):
        if args.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            batch = data[batch_idxes]
            normalized_obs = {k: batch[k] / 255 - 0.5 if k in cnn_keys else batch[k] for k in mlp_keys + cnn_keys}
            _, logprobs, entropy, new_values = agent(
                normalized_obs, torch.split(batch["actions"], agent.actions_dim, dim=-1)
            )

            if args.normalize_advantages:
                batch["advantages"] = normalize_tensor(batch["advantages"])

            # Policy loss
            pg_loss = policy_loss(
                logprobs,
                batch["logprobs"],
                batch["advantages"],
                args.clip_coef,
                args.loss_reduction,
            )

            # Value loss
            v_loss = value_loss(
                new_values, batch["values"], batch["returns"], args.clip_coef, args.clip_vloss, args.loss_reduction
            )

            # Entropy loss
            ent_loss = entropy_loss(entropy, args.loss_reduction)

            # Equation (9) in the paper
            loss = pg_loss + args.vf_coef * v_loss + args.ent_coef * ent_loss

            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            if args.max_grad_norm > 0.0:
                fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()

            # Update metrics
            aggregator.update("Loss/policy_loss", pg_loss.detach())
            aggregator.update("Loss/value_loss", v_loss.detach())
            aggregator.update("Loss/entropy_loss", ent_loss.detach())


@register_algorithm()
def main():
    parser = HfArgumentParser(PPOArgs)
    args: PPOArgs = parser.parse_args_into_dataclasses()[0]
    initial_ent_coef = copy.deepcopy(args.ent_coef)
    initial_clip_coef = copy.deepcopy(args.clip_coef)

    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    world_collective = TorchCollective()
    if fabric.world_size > 1:
        world_collective.setup()
        world_collective.create_group()
    if rank == 0:
        root_dir = (
            args.root_dir
            if args.root_dir is not None
            else os.path.join("logs", "ppo", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        )
        logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
        fabric._loggers = [logger]
        log_dir = logger.log_dir
        fabric.logger.log_hyperparams(asdict(args))
        if fabric.world_size > 1:
            world_collective.broadcast_object_list([log_dir], src=0)

        # Save args as dict automatically
        args.log_dir = log_dir
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
        os.makedirs(log_dir, exist_ok=True)

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if args.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_dict_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args,
                logger.log_dir if rank == 0 else None,
                "train",
                mask_velocities=args.mask_vel,
                vector_env_idx=i,
            )
            for i in range(args.num_envs)
        ]
    )

    cnn_keys = []
    mlp_keys = []
    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        cnn_keys = []
        for k, v in envs.single_observation_space.spaces.items():
            if args.cnn_keys and k in args.cnn_keys:
                if len(v.shape) in {3, 4}:
                    cnn_keys.append(k)
                else:
                    fabric.print(
                        f"Found a CNN key which is not an image: `{k}` of shape {v.shape}. "
                        "Try to transform the observation from the environment into a 3D image"
                    )
        for k, v in envs.single_observation_space.spaces.items():
            if args.mlp_keys and k in args.mlp_keys:
                if len(v.shape) == 1:
                    mlp_keys.append(k)
                else:
                    fabric.print(
                        f"Found an MLP key which is not a vector: `{k}` of shape {v.shape}. "
                        "Try to flatten the observation from the environment"
                    )
    else:
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {envs.single_observation_space}")
    if cnn_keys == [] and mlp_keys == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: `--cnn_keys rgb` or `--mlp_keys state` "
        )
    fabric.print("CNN keys:", cnn_keys)
    fabric.print("MLP keys:", mlp_keys)

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
        obs_space=envs.single_observation_space,
        cnn_keys=cnn_keys,
        mlp_keys=mlp_keys,
        cnn_features_dim=args.cnn_features_dim,
        mlp_features_dim=args.mlp_features_dim,
        screen_size=args.screen_size,
        cnn_channels_multiplier=args.cnn_channels_multiplier,
        mlp_layers=args.mlp_layers,
        dense_units=args.dense_units,
        cnn_act=args.cnn_act,
        mlp_act=args.dense_act,
        layer_norm=args.layer_norm,
        is_continuous=is_continuous,
    )

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = Adam(list(agent.parameters()), lr=args.lr, eps=args.eps)
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
        args.rollout_steps,
        args.num_envs,
        device=device,
        memmap=args.memmap_buffer,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        obs_keys=cnn_keys + mlp_keys,
    )
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.perf_counter()
    single_global_rollout = int(args.num_envs * args.rollout_steps * world_size)
    num_updates = args.total_steps // single_global_rollout if not args.dry_run else 1

    # Linear learning rate scheduler
    if args.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)

    # Get the first environment observation and start the optimization
    o = envs.reset(seed=args.seed)[0]  # [N_envs, N_obs]
    next_obs = {}
    for k in o.keys():
        if k in mlp_keys + cnn_keys:
            torch_obs = torch.from_numpy(o[k]).to(fabric.device)
            if k in cnn_keys:
                torch_obs = torch_obs.view(args.num_envs, -1, *torch_obs.shape[-2:])
            if k in mlp_keys:
                torch_obs = torch_obs.float()
            step_data[k] = torch_obs
            next_obs[k] = torch_obs
    next_done = torch.zeros(args.num_envs, 1, dtype=torch.float32).to(fabric.device)  # [N_envs, 1]

    for update in range(1, num_updates + 1):
        for _ in range(0, args.rollout_steps):
            global_step += args.num_envs * world_size

            with torch.no_grad():
                # Sample an action given the observation received by the environment
                normalized_obs = {
                    k: next_obs[k] / 255 - 0.5 if k in cnn_keys else next_obs[k] for k in mlp_keys + cnn_keys
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
                rewards = torch.tensor(reward, dtype=torch.float32).view(args.num_envs, -1)  # [N_envs, 1]
                done = torch.tensor(done, dtype=torch.float32).view(args.num_envs, -1)  # [N_envs, 1]

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
                if k in mlp_keys + cnn_keys:
                    torch_obs = torch.from_numpy(o[k]).to(fabric.device)
                    if k in cnn_keys:
                        torch_obs = torch_obs.view(args.num_envs, -1, *torch_obs.shape[-2:])
                    if k in mlp_keys:
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
            normalized_obs = {k: next_obs[k] / 255 - 0.5 if k in cnn_keys else next_obs[k] for k in mlp_keys + cnn_keys}
            next_values = agent.get_value(normalized_obs)
            returns, advantages = gae(
                rb["rewards"],
                rb["values"],
                rb["dones"],
                next_values,
                next_done,
                args.rollout_steps,
                args.gamma,
                args.gae_lambda,
            )

            # Add returns and advantages to the buffer
            rb["returns"] = returns.float()
            rb["advantages"] = advantages.float()

        # Flatten the batch
        local_data = rb.buffer.view(-1)

        if args.share_data and fabric.world_size > 1:
            # Gather all the tensors from all the world and reshape them
            gathered_data = fabric.all_gather(local_data.to_dict())  # Fabric does not work with TensorDict
            gathered_data = make_tensordict(gathered_data).view(-1)
        else:
            gathered_data = local_data

        train(fabric, agent, optimizer, gathered_data, aggregator, args, cnn_keys, mlp_keys)

        if args.anneal_lr:
            fabric.log("Info/learning_rate", scheduler.get_last_lr()[0], global_step)
            scheduler.step()
        else:
            fabric.log("Info/learning_rate", args.lr, global_step)

        fabric.log("Info/clip_coef", args.clip_coef, global_step)
        if args.anneal_clip_coef:
            args.clip_coef = polynomial_decay(
                update, initial=initial_clip_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        fabric.log("Info/ent_coef", args.ent_coef, global_step)
        if args.anneal_ent_coef:
            args.ent_coef = polynomial_decay(
                update, initial=initial_ent_coef, final=0.0, max_decay_steps=num_updates, power=1.0
            )

        # Log metrics
        metrics_dict = aggregator.compute()
        fabric.log("Time/step_per_second", int(global_step / (time.perf_counter() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

        # Checkpoint model
        if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run or update == num_updates:
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": asdict(args),
                "update_step": update,
                "scheduler": scheduler.state_dict() if args.anneal_lr else None,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{update}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero:
        test_env = make_dict_env(
            args.env_id,
            None,
            0,
            args,
            fabric.logger.log_dir,
            "test",
            mask_velocities=args.mask_vel,
            vector_env_idx=0,
        )()
        test(agent.module, test_env, fabric, args, cnn_keys, mlp_keys)


if __name__ == "__main__":
    main()
