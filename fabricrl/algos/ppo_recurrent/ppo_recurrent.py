import argparse
import os
import time
import warnings
from datetime import datetime

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
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.args import parse_args
from fabricrl.algos.ppo.loss import policy_loss, value_loss
from fabricrl.algos.ppo.utils import make_env, test
from fabricrl.algos.ppo_recurrent.agent import RecurrentPPOAgent
from fabricrl.data import ReplayBuffer
from fabricrl.utils.metric import MetricAggregator
from fabricrl.utils.utils import gae, linear_annealing, normalize_tensor


def train(
    fabric: Fabric,
    agent: RecurrentPPOAgent,
    optimizer: torch.optim.Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: argparse.Namespace,
):
    for _ in range(args.update_epochs):
        env_idxes = torch.randperm(args.num_envs)
        env_idxes_batches = torch.tensor_split(env_idxes, args.envs_batch_size)
        for env_idxes_batch in env_idxes_batches:
            if env_idxes_batch.numel() == 0:
                continue
            batch = data[:, env_idxes_batch]
            action_logits, new_values, _ = agent(
                batch["observations"],
                batch["dones"],
                state=tuple([s[:, env_idxes_batch] for s in agent.initial_states]),
            )

            dist = Categorical(logits=action_logits.unsqueeze(-2))
            if args.normalize_advantages:
                batch["advantages"] = normalize_tensor(batch["advantages"])

            pg_loss = policy_loss(dist, batch, args.clip_coef)

            # Value loss
            v_loss = value_loss(
                new_values,
                batch["values"],
                batch["returns"],
                args.clip_coef,
                args.clip_vloss,
            )

            # Entropy loss
            entropy = dist.entropy().mean()

            # Equation (9) in the paper, changed the sign since we minimize
            loss = -pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()

            # Update metrics
            aggregator.update("Loss/policy_loss", pg_loss.detach())
            aggregator.update("Loss/value_loss", v_loss.detach())
            aggregator.update("Loss/entropy_loss", entropy.detach())


def main(args: argparse.Namespace):
    if args.share_data:
        warnings.warn("The script has been called with --share-data: with recurrent PPO only gradients are shared")

    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "ppo_recurrent", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
        name=run_name,
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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    agent = fabric.setup_module(RecurrentPPOAgent(envs))
    optimizer = fabric.setup_optimizers(Adam(params=agent.parameters(), lr=args.learning_rate, eps=1e-4))

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
    rb = ReplayBuffer(args.num_steps, args.num_envs, device=device)
    step_data = TensorDict({}, batch_size=[1, args.num_envs], device=device)

    # Global variables
    global_step = 0
    start_time = time.time()
    single_global_rollout = int(args.num_envs * args.num_steps * world_size)
    num_updates = args.total_timesteps // single_global_rollout

    with device:
        # Get the first environment observation and start the optimization
        next_obs = torch.tensor(envs.reset(seed=args.seed)[0]).unsqueeze(0)  # [1, N_envs, N_obs]
        next_done = torch.zeros(1, args.num_envs, 1)  # [1, N_envs, 1]
        state = agent.initial_states

    for update in range(1, num_updates + 1):
        # Learning rate annealing
        if args.anneal_lr:
            linear_annealing(optimizer, update, num_updates, args.learning_rate)
        fabric.log("Info/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        initial_states = (state[0].clone(), state[1].clone())
        for _ in range(0, args.num_steps):
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
        with torch.no_grad():
            next_value, _ = agent.module.get_values(next_obs, next_done, critic_state=state[1])
            returns, advantages = gae(
                rb["rewards"],
                rb["values"],
                rb["dones"],
                next_value,
                next_done,
                args.num_steps,
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

        # Log metrics
        metrics_dict = aggregator.compute()
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)
        fabric.log_dict(metrics_dict, global_step)
        aggregator.reset()

    envs.close()
    if fabric.is_global_zero:
        test(agent.module, device, fabric.logger.experiment, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
