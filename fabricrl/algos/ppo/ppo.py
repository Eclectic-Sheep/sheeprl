import argparse
import os
import time
from datetime import datetime

import torch
from gymnasium.vector import SyncVectorEnv
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.args import parse_args
from fabricrl.algos.ppo.loss import policy_loss, value_loss
from fabricrl.algos.ppo.utils import make_env, test
from fabricrl.data import ReplayBuffer
from fabricrl.models.models import MLP
from fabricrl.utils.metric import MetricAggregator
from fabricrl.utils.utils import gae, linear_annealing, normalize_tensor

__all__ = ["main"]


def train(
    fabric: Fabric,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: argparse.Namespace,
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
            actions_logits = actor(batch["observations"])
            new_values = critic(batch["observations"])

            dist = Categorical(logits=actions_logits.unsqueeze(-2))
            if args.normalize_advantages:
                batch["advantages"] = normalize_tensor(batch["advantages"])

            # Policy loss
            pg_loss = policy_loss(dist, batch, args.clip_coef)

            # Value loss
            v_loss = value_loss(new_values, batch["values"], batch["returns"], args.clip_coef, args.clip_vloss)

            # Entropy loss
            entropy = dist.entropy().mean()

            # Equation (9) in the paper, changed the sign since we minimize
            loss = -pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(actor, optimizer, max_norm=args.max_grad_norm)
            fabric.clip_gradients(critic, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()

            # Update metrics
            aggregator.update("Loss/policy_loss", pg_loss.detach())
            aggregator.update("Loss/value_loss", v_loss.detach())
            aggregator.update("Loss/entropy_loss", entropy.detach())


def main():
    args = parse_args()

    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "ppo", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")), name=run_name
    )

    # Initialize Fabric
    fabric = Fabric(loggers=logger)
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Log hyperparameters
    fabric.logger.experiment.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Environment setup
    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args.capture_video,
                logger.log_dir,
                "train",
                mask_velocities=args.mask_vel,
            )
            for i in range(args.num_envs)
        ]
    )

    # Create the actor and critic models
    actor = MLP(
        input_dims=envs.single_observation_space.shape,
        output_dim=envs.single_action_space.n,
        hidden_sizes=(64, 64),
        activation=torch.nn.ReLU,
    )
    critic = MLP(
        input_dims=envs.single_observation_space.shape, output_dim=1, hidden_sizes=(64, 64), activation=torch.nn.ReLU
    )

    # Define the models and the optimizer and set them up with Fabric
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate, eps=1e-4)
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
    rb = ReplayBuffer(args.num_steps, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[args.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_rollout = int(args.num_envs * args.num_steps * world_size)
    num_updates = args.total_timesteps // single_global_rollout

    with device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(envs.reset(seed=args.seed)[0])  # [N_envs, N_obs]
        next_done = torch.zeros(args.num_envs, 1)  # [N_envs, 1]

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            linear_annealing(optimizer, update, num_updates, args.learning_rate)
        fabric.log("Info/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        for _ in range(0, args.num_steps):
            global_step += args.num_envs * world_size

            with torch.inference_mode():
                # Sample an action given the observation received by the environment
                action_logits = actor.module(next_obs)
                dist = Categorical(logits=action_logits.unsqueeze(-2))
                action = dist.sample()
                logprob = dist.log_prob(action)

                # Estimate critic value
                value = critic.module(next_obs)

            # Update the step data
            step_data["dones"] = next_done
            step_data["values"] = value
            step_data["actions"] = action
            step_data["logprobs"] = logprob
            step_data["observations"] = next_obs

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(
                step_data["actions"].cpu().numpy().reshape(envs.action_space.shape)
            )

            with device:
                # Save reward for the last (observation, action) pair
                step_data["rewards"] = torch.tensor(reward).view(args.num_envs, -1)  # [N_envs, 1]

                # Update the observation
                next_obs = torch.tensor(next_obs)
                next_done = (
                    torch.logical_or(torch.tensor(done), torch.tensor(truncated)).view(args.num_envs, -1).float()
                )  # [N_envs, 1]

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                        aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        with torch.inference_mode():
            next_values = critic(next_obs)
            returns, advantages = gae(
                rb["rewards"],
                rb["values"],
                rb["dones"],
                next_values,
                next_done,
                args.num_steps,
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

        # Train the agent
        train(fabric, actor, critic, optimizer, gathered_data, aggregator, args)

        # Log metrics
        metrics_dict = aggregator.compute()
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

    envs.close()
    if fabric.is_global_zero:
        test(actor.module, device, fabric.logger.experiment, args)


if __name__ == "__main__":
    main()
