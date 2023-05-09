import os
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from math import prod

import gymnasium as gym
import torch
from gymnasium.vector import SyncVectorEnv
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.args import PPOArgs
from fabricrl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from fabricrl.algos.ppo.utils import make_env
from fabricrl.algos.ppo_recurrent.agent import RecurrentPPOAgent
from fabricrl.algos.ppo_recurrent.utils import test
from fabricrl.data import ReplayBuffer
from fabricrl.utils.metric import MetricAggregator
from fabricrl.utils.parser import HfArgumentParser
from fabricrl.utils.utils import gae, normalize_tensor

__all__ = ["main"]


def train(
    fabric: Fabric,
    agent: RecurrentPPOAgent,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: PPOArgs,
):
    for _ in range(args.update_epochs):
        states = agent.initial_states
        seq_sampler = BatchSampler(range(len(data)), batch_size=args.per_rank_batch_size, drop_last=False)
        for seq_idxes_batch in seq_sampler:
            batch = data[seq_idxes_batch]
            action_logits, new_values, states = agent(
                batch["observations"],
                batch["dones"],
                state=tuple([tuple([s.detach() for s in state]) for state in states]),
            )
            dist = Categorical(logits=action_logits.unsqueeze(-2))

            if args.normalize_advantages:
                batch["advantages"] = normalize_tensor(batch["advantages"])

            # Policy loss
            pg_loss = policy_loss(
                dist.log_prob(batch["actions"]),
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
            ent_loss = entropy_loss(dist.entropy(), args.loss_reduction)

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


def main():
    parser = HfArgumentParser(PPOArgs)
    args: PPOArgs = parser.parse_args_into_dataclasses()[0]

    if args.share_data:
        warnings.warn("The script has been called with --share-data: with recurrent PPO only gradients are shared")

    # Initialize Fabric
    fabric = Fabric()
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    world_size = fabric.world_size
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set logger only on rank-0
    if rank == 0:
        run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        logger = TensorBoardLogger(
            root_dir=os.path.join("logs", "ppo_recurrent", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
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
            )
            for i in range(args.num_envs)
        ]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError("Only discrete action space is supported")

    # Define the agent and the optimizer and setup them with Fabric
    obs_dim = prod(envs.single_observation_space.shape)
    agent = fabric.setup_module(
        RecurrentPPOAgent(observation_dim=obs_dim, action_dim=envs.single_action_space.n, num_envs=args.num_envs)
    )
    optimizer = fabric.setup_optimizers(Adam(params=agent.parameters(), lr=args.lr, eps=1e-4))

    # Metrics
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
    step_data = TensorDict({}, batch_size=[1, args.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_rollout = int(args.num_envs * args.rollout_steps * world_size)
    num_updates = args.total_steps // single_global_rollout

    # Linear learning rate scheduler
    if args.anneal_lr:
        from torch.optim.lr_scheduler import PolynomialLR

        scheduler = PolynomialLR(optimizer=optimizer, total_iters=num_updates, power=1.0)

    with device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(envs.reset(seed=args.seed)[0]).unsqueeze(0)  # [1, N_envs, N_obs]
        next_done = torch.zeros(1, args.num_envs, 1)  # [1, N_envs, 1]
        state = agent.initial_states

    for _ in range(1, num_updates + 1):
        initial_states = (
            tuple([s.clone() for s in agent.initial_states[0]]),
            tuple([s.clone() for s in agent.initial_states[1]]),
        )
        for _ in range(0, args.rollout_steps):
            global_step += args.num_envs * world_size

            with torch.inference_mode():
                # Sample an action given the observation received by the environment
                action_logits, values, state = agent.module(next_obs, next_done, state=state)
                dist = Categorical(logits=action_logits.unsqueeze(-2))
                action = dist.sample()
                logprob = dist.log_prob(action)

            step_data["dones"] = next_done
            step_data["values"] = values
            step_data["actions"] = action
            step_data["logprobs"] = logprob
            step_data["observations"] = next_obs

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                next_obs = torch.tensor(next_obs).unsqueeze(0)
                next_done = (
                    torch.logical_or(torch.tensor(done), torch.tensor(truncated)).view(1, args.num_envs, 1).float()
                )  # [1, N_envs, 1]

                # Save reward for the last (observation, action) pair
                step_data["rewards"] = torch.tensor(reward).view(1, args.num_envs, -1)  # [1, N_envs, N_rews]

            # Append data to buffer
            rb.add(step_data)

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
            next_value, _ = agent.module.get_values(next_obs, next_done, critic_state=state[1])
            returns, advantages = gae(
                rb["rewards"],
                rb["values"],
                rb["dones"],
                next_value,
                next_done,
                args.rollout_steps,
                args.gamma,
                args.gae_lambda,
            )

            # Add returns and advantages to the buffer
            rb["returns"] = returns.float()
            rb["advantages"] = advantages.float()

        # Get the training data as a TensorDict
        local_data = rb.buffer

        # Train the agent
        agent.initial_states = initial_states
        train(fabric, agent, optimizer, local_data, aggregator, args)

        # Learning rate annealing
        if args.anneal_lr:
            scheduler.step()
            fabric.log("Info/learning_rate", scheduler.get_last_lr()[0], global_step)
        else:
            fabric.log("Info/learning_rate", args.lr, global_step)

        # Log metrics
        metrics_dict = aggregator.compute()
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

    envs.close()
    if fabric.is_global_zero:
        test(agent.module, fabric, args)


if __name__ == "__main__":
    main()
