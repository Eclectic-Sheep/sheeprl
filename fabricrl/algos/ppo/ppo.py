"""
Proximal Policy Optimization (PPO) - Accelerated with Lightning Fabric

Author: Federico Belotti @belerico
Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
Based on the paper: https://arxiv.org/abs/1707.06347

Requirements:
- gymnasium[box2d]>=0.27.1
- moviepy
- lightning
- torchmetrics
- tensorboard


Run it with:
    lightning run model --accelerator=cpu --strategy=ddp --devices=2 train_fabric.py
"""

import argparse
import math
import os
import time
from datetime import datetime
from distutils.util import strtobool
from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from lightning.pytorch import LightningModule
from tensordict import TensorDict, make_tensordict
from torch import Tensor
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from fabricrl.data.buffers import ReplayBuffer
from lightning.fabric.fabric import _is_using_cli


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default", help="the name of this experiment")

    # PyTorch arguments
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, GPU training will be used. "
        "This affects also the distributed backend used (NCCL (gpu) vs GLOO (cpu))",
    )
    parser.add_argument(
        "--player-on-gpu",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, player will run on GPU (used only by `train_fabric_decoupled.py` script). "
        "This affects also the distributed backend used (NCCL (gpu) vs GLOO (cpu))",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )

    # Distributed arguments
    parser.add_argument("--num-envs", type=int, default=2, help="the number of parallel game environments")
    parser.add_argument(
        "--share-data",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle sharing data between processes",
    )
    parser.add_argument("--per-rank-batch-size", type=int, default=64, help="the batch size for each rank")

    # Environment arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="the id of the environment")
    parser.add_argument(
        "--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout"
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )

    # PPO arguments
    parser.add_argument("--total-timesteps", type=int, default=2**16, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="the learning rate of the optimizer")
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation"
    )
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument(
        "--activation-function",
        type=str,
        default="relu",
        choices=["relu", "tanh"],
        help="The activation function of the model",
    )
    parser.add_argument(
        "--ortho-init",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles the orthogonal initialization of the model",
    )
    parser.add_argument(
        "--normalize-advantages",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    args = parser.parse_args()
    return args


def layer_init(
    layer: torch.nn.Module,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
    ortho_init: bool = True,
):
    if ortho_init:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linear_annealing(optimizer: torch.optim.Optimizer, update: int, num_updates: int, initial_lr: float):
    frac = 1.0 - (update - 1.0) / num_updates
    lrnow = frac * initial_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lrnow


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: Optional[str] = None, prefix: str = ""):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0 and run_name is not None:
                env = gym.wrappers.RecordVideo(
                    env, os.path.join(run_name, prefix + "_videos" if prefix else "videos"), disable_logger=True
                )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


@torch.no_grad()
def test(agent: "PPOLightningAgent", device: torch.device, logger: SummaryWriter, args: argparse.Namespace):
    env = make_env(args.env_id, args.seed, 0, args.capture_video, logger.log_dir, "test")()
    step = 0
    done = False
    cumulative_rew = 0
    next_obs = torch.tensor(env.reset(seed=args.seed)[0], device=device)
    while not done:
        # Act greedly through the environment
        action = agent.get_greedy_action(next_obs)

        # Single environment step
        next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
        done = done or truncated
        cumulative_rew += reward
        next_obs = torch.tensor(next_obs, device=device)
        step += 1
    logger.add_scalar("Test/cumulative_reward", cumulative_rew, 0)
    env.close()


class PPOLightningAgent(LightningModule):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        act_fun: str = "relu",
        ortho_init: bool = False,
        vf_coef: float = 1.0,
        ent_coef: float = 0.0,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        normalize_advantages: bool = False,
        **torchmetrics_kwargs,
    ):
        super().__init__()
        if act_fun.lower() == "relu":
            act_fun = torch.nn.ReLU()
        elif act_fun.lower() == "tanh":
            act_fun = torch.nn.Tanh()
        else:
            raise ValueError("Unrecognized activation function: `act_fun` must be either `relu` or `tanh`")
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages
        self.critic = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(math.prod(envs.single_observation_space.shape), 64),
                ortho_init=ortho_init,
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, 1), std=1.0, ortho_init=ortho_init),
        )
        self.actor = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(math.prod(envs.single_observation_space.shape), 64),
                ortho_init=ortho_init,
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, envs.single_action_space.n), std=0.01, ortho_init=ortho_init),
        )
        self.avg_pg_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_value_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_ent_loss = MeanMetric(**torchmetrics_kwargs)

    def get_action(self, x: Tensor, action: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.actor(x)
        distribution = Categorical(logits=logits.unsqueeze(-2))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()

    def get_greedy_action(self, x: Tensor) -> Tensor:
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: Tensor, action: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        action, log_prob, entropy = self.get_action(x, action)
        value = self.get_value(x)
        return action, log_prob, entropy, value

    def forward(self, x: Tensor, action: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.get_action_and_value(x, action)

    @torch.no_grad()
    def estimate_returns_and_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_obs: Tensor,
        next_done: Tensor,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[Tensor, Tensor]:
        next_value = self.get_value(next_obs)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = torch.logical_not(next_done)
                nextvalues = next_value
            else:
                nextnonterminal = torch.logical_not(dones[t + 1])
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

    def training_step(self, batch: Dict[str, Tensor]):
        # Get actions and values given the current observations
        _, newlogprob, entropy, newvalue = self(batch["observations"], batch["actions"].long())
        logratio = newlogprob - batch["logprobs"]
        ratio = logratio.exp()

        # Policy loss
        advantages = batch["advantages"]
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = policy_loss(batch["advantages"], ratio, self.clip_coef)

        # Value loss
        v_loss = value_loss(
            newvalue,
            batch["values"],
            batch["returns"],
            self.clip_coef,
            self.clip_vloss,
            self.vf_coef,
        )

        # Entropy loss
        ent_loss = entropy_loss(entropy, self.ent_coef)

        # Update metrics
        self.avg_pg_loss(pg_loss)
        self.avg_value_loss(v_loss)
        self.avg_ent_loss(ent_loss)

        # Overall loss
        return pg_loss + ent_loss + v_loss

    def on_train_epoch_end(self, global_step: int) -> None:
        # Log metrics and reset their internal state
        self.logger.log_metrics(
            {
                "Loss/policy_loss": self.avg_pg_loss.compute(),
                "Loss/value_loss": self.avg_value_loss.compute(),
                "Loss/entropy_loss": self.avg_ent_loss.compute(),
            },
            global_step,
        )
        self.reset_metrics()

    def reset_metrics(self):
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)


def train(
    fabric: Fabric,
    agent: PPOLightningAgent,
    optimizer: torch.optim.Optimizer,
    data: Dict[str, Tensor],
    global_step: int,
    args: argparse.Namespace,
):
    indexes = list(range(data["observations"].shape[0]))
    if args.share_data:
        sampler = DistributedSampler(
            indexes, num_replicas=fabric.world_size, rank=fabric.global_rank, shuffle=True, seed=args.seed
        )
    else:
        sampler = RandomSampler(indexes)
    sampler = BatchSampler(sampler, batch_size=args.per_rank_batch_size, drop_last=False)

    for epoch in range(args.update_epochs):
        if args.share_data:
            sampler.sampler.set_epoch(epoch)
        for batch_idxes in sampler:
            loss = agent.training_step({k: v[batch_idxes] for k, v in data.items()})
            optimizer.zero_grad(set_to_none=True)
            fabric.backward(loss)
            fabric.clip_gradients(agent, optimizer, max_norm=args.max_grad_norm)
            optimizer.step()
        agent.on_train_epoch_end(global_step)


def main(args: argparse.Namespace):
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    logger = TensorBoardLogger(
        root_dir=os.path.join("logs", "fabric_logs", datetime.today().strftime("%Y-%m-%d_%H-%M-%S")), name=run_name
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
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, args.seed + rank * args.num_envs + i, rank, args.capture_video, logger.log_dir, "train"
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Define the agent and the optimizer and setup them with Fabric
    agent: PPOLightningAgent = PPOLightningAgent(
        envs,
        act_fun=args.activation_function,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ortho_init=args.ortho_init,
        normalize_advantages=args.normalize_advantages,
    )
    optimizer = agent.configure_optimizers(args.learning_rate)
    agent, optimizer = fabric.setup(agent, optimizer)

    # Player metrics
    rew_avg = torchmetrics.MeanMetric().to(device)
    ep_len_avg = torchmetrics.MeanMetric().to(device)

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

            # Sample an action given the observation received by the environment
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)

            step_data["dones"] = next_done
            step_data["values"] = value
            step_data["actions"] = action
            step_data["logprobs"] = logprob
            step_data["observations"] = next_obs

            # Single environment step
            next_obs, reward, done, truncated, info = envs.step(action.cpu().numpy().reshape(envs.action_space.shape))

            with device:
                next_obs = torch.tensor(next_obs)
                next_done = (
                    torch.logical_or(torch.tensor(done), torch.tensor(truncated)).view(args.num_envs, -1).float()
                )  # [N_envs, 1]

                # Save reward for the last (observation, action) pair
                step_data["rewards"] = torch.tensor(reward).view(args.num_envs, -1)  # [N_envs, 1]

            # Append data to buffer
            rb.add(step_data.unsqueeze(0))

            if "final_info" in info:
                for i, agent_final_info in enumerate(info["final_info"]):
                    if agent_final_info is not None and "episode" in agent_final_info:
                        fabric.print(
                            f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                        )
                        rew_avg(agent_final_info["episode"]["r"][0])
                        ep_len_avg(agent_final_info["episode"]["l"][0])

        # Sync the metrics
        rew_avg_reduced = rew_avg.compute()
        if not rew_avg_reduced.isnan():
            fabric.log("Rewards/rew_avg", rew_avg_reduced, global_step)
        ep_len_avg_reduced = ep_len_avg.compute()
        if not ep_len_avg_reduced.isnan():
            fabric.log("Game/ep_len_avg", ep_len_avg_reduced, global_step)
        rew_avg.reset()
        ep_len_avg.reset()

        # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)
        returns, advantages = agent.estimate_returns_and_advantages(
            rb["rewards"],
            rb["values"],
            rb["dones"],
            next_obs,
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
            gathered_data = fabric.all_gather(
                local_data.to_dict()
            )  # Fabric does not work with TensorDict: I'll open them a PR!
            gathered_data = make_tensordict(gathered_data).view(-1)
        else:
            gathered_data = local_data

        # Train the agent
        train(fabric, agent, optimizer, gathered_data, global_step, args)
        fabric.log("Time/step_per_second", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    if fabric.is_global_zero:
        test(agent.module, device, fabric.logger.experiment, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
