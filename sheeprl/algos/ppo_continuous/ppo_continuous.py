import copy
import os
import time
from dataclasses import asdict
from datetime import datetime
from math import prod

import gymnasium as gym
import torch
from gymnasium.vector import SyncVectorEnv
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.optim import Adam
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import MeanMetric

from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from sheeprl.algos.ppo_continuous.agent import PPOContinuousActor
from sheeprl.algos.ppo_continuous.args import PPOContinuousArgs
from sheeprl.algos.ppo_continuous.utils import test
from sheeprl.data import ReplayBuffer
from sheeprl.models.models import MLP
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import gae, make_env, normalize_tensor, polynomial_decay


def train(
    fabric: Fabric,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: PPOContinuousArgs,
):
    """Train the agent on the data collected from the environment."""
    indexes = list(range(data.batch_size[0]))
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
            _, logprobs, entropy = actor(batch["observations"], batch["actions"])
            new_values = critic(batch["observations"])

            if args.normalize_advantages:
                batch["advantages"] = normalize_tensor(batch["advantages"])

            # Policy loss
            pg_loss = policy_loss(logprobs, batch["logprobs"], batch["advantages"], args.clip_coef, args.loss_reduction)

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
                fabric.clip_gradients(actor, optimizer, max_norm=args.max_grad_norm)
                fabric.clip_gradients(critic, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()

            # Update metrics
            aggregator.update("Loss/policy_loss", pg_loss.detach())
            aggregator.update("Loss/value_loss", v_loss.detach())
            aggregator.update("Loss/entropy_loss", ent_loss.detach())


@register_algorithm()
def main():
    parser = HfArgumentParser(PPOContinuousArgs)
    args: PPOContinuousArgs = parser.parse_args_into_dataclasses()[0]
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
            else os.path.join("logs", "ppo_continuous", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
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
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
        os.makedirs(log_dir, exist_ok=True)

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
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported")

    # Create the actor and critic models
    act_dim = prod(envs.single_action_space.shape)
    obs_dim = prod(envs.single_observation_space.shape)
    actor = PPOContinuousActor(observation_dim=obs_dim, action_dim=act_dim)
    critic = MLP(input_dims=obs_dim, output_dim=1, hidden_sizes=(64, 64), activation=torch.nn.ReLU)

    # Define the agent and the optimizer and setup them with Fabric
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-4)
    actor = fabric.setup_module(actor)
    critic = fabric.setup_module(critic)
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
                action, logprob, _ = actor.module(next_obs)

                # Estimate critic value
                value = critic.module(next_obs)

            # Single environment step
            obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                obs = torch.tensor(obs)  # [N_envs, N_obs]
                rewards = torch.tensor(reward).view(args.num_envs, -1)  # [N_envs, 1]
                done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))  # [N_envs, 1]
                done = done.view(args.num_envs, -1).float()

            # Update the step data
            step_data["dones"] = next_done
            step_data["values"] = value
            step_data["actions"] = action
            step_data["logprobs"] = logprob
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

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.no_grad():
            next_values = critic(next_obs)
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

        train(fabric, actor, critic, optimizer, gathered_data, aggregator, args)

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
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

        # Checkpoint model
        if (args.checkpoint_every > 0 and update % args.checkpoint_every == 0) or args.dry_run or update == num_updates:
            state = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": asdict(args),
                "update_step": update,
                "scheduler": scheduler.state_dict() if args.anneal_lr else None,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{update}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    envs.close()
    if fabric.is_global_zero:
        test_env = make_env(
            args.env_id,
            None,
            0,
            args.capture_video,
            fabric.logger.log_dir,
            "test",
            mask_velocities=args.mask_vel,
            vector_env_idx=0,
        )()
        test(actor.module, test_env, fabric, args)


if __name__ == "__main__":
    main()
