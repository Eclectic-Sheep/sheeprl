import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from tensordict import TensorDict
from torch import Tensor
from torch.distributions import Categorical
from torch.optim import Adam, Optimizer
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.loss import entropy_loss, policy_loss, value_loss
from fabricrl.utils.utils import conditional_arange, layer_init


class PPOAgent(LightningModule):
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

    @torch.no_grad()
    def fast_estimate_returns_and_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_obs: Tensor,
        next_done: Tensor,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute returns and advantages following https://arxiv.org/abs/1506.02438

        Args:
            rewards (Tensor): all rewards collected from the last rollout
            values (Tensor): all values collected from the last rollout
            dones (Tensor): all dones collected from the last rollout
            next_obs (Tensor): next observation
            next_done (Tensor): next done
            num_steps (int): the number of steps played
            gamma (float): discout factor
            gae_lambda (float): lambda for GAE estimation
            state (Tuple[Tensor, Tensor]): recurrent state for both the actor and critic

        Returns:
            estimated returns
            estimated advantages
        """
        next_value = self.get_value(next_obs)
        if len(rewards.shape) == 3:
            t_steps = torch.cat(
                [
                    conditional_arange(num_steps, dones[:, dim, :].view(-1)).view(-1, 1)
                    for dim in range(rewards.shape[1])
                ],
                dim=1,
            ).unsqueeze(-1)
        elif len(rewards.shape) == 2:
            t_steps = conditional_arange(num_steps, dones.view(-1)).view(-1, 1)
        else:
            raise ValueError(f"Shape must be 2 or 3 dimensional, got {rewards.shape}")
        gt = (gamma * gae_lambda) ** t_steps
        next_values = torch.roll(values, -1, dims=0)
        next_values[-1] = next_value
        next_dones = torch.roll(dones, -1, dims=0)
        next_dones[-1] = next_done
        deltas = rewards + gamma * next_values * (1 - next_dones) - values
        cs = torch.flipud(deltas * gt).cumsum(dim=0)
        acc = torch.cummax(torch.where(torch.flipud(dones.bool()), cs, 0), 0)[0]
        acc[0] = 0
        dones[-1] = 0
        adv = torch.flipud(cs - acc) / gt
        adv = adv + dones * (deltas + gamma * gae_lambda * adv.roll(-1, 0))
        return adv + values, adv

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


class RecurrentPPOAgent(LightningModule):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        act_fun: str = "relu",
        hidden_size: int = 64,
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
        self.hidden_size = hidden_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages
        self.actor_fc = layer_init(
            torch.nn.Linear(math.prod(envs.single_observation_space.shape), self.hidden_size),
            ortho_init=ortho_init,
        )
        self.actor_rnn = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=False)
        if ortho_init:
            self._orthogonal_init_rnn(self.actor_rnn)
        self.actor = torch.nn.Sequential(
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, envs.single_action_space.n), std=0.01, ortho_init=ortho_init),
        )
        self.critic_fc = layer_init(
            torch.nn.Linear(math.prod(envs.single_observation_space.shape), self.hidden_size),
            ortho_init=ortho_init,
        )
        self.critic_rnn = torch.nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=False)
        if ortho_init:
            self._orthogonal_init_rnn(self.critic_rnn)
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, 1), std=1.0, ortho_init=ortho_init),
        )
        self.avg_pg_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_value_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_ent_loss = MeanMetric(**torchmetrics_kwargs)
        self._initial_states: Optional[Tuple[Tensor, Tensor]] = None

    @property
    def initial_states(self) -> Optional[Tuple[Tensor, Tensor]]:
        return self._initial_states

    @initial_states.setter
    def initial_states(self, value: Tuple[Tensor, Tensor]) -> None:
        self._initial_states = value

    def _orthogonal_init_rnn(self, rnn: torch.nn.RNNBase) -> None:
        # https://github.com/vwxyzjn/cleanrl/issues/358
        for name, param in rnn.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0)
            elif "weight" in name:
                torch.nn.init.orthogonal_(param[:128], 1.0)
                torch.nn.init.orthogonal_(param[128 : 128 * 2], 1.0)
                torch.nn.init.orthogonal_(param[128 * 2 : 128 * 3], 1.0)
                torch.nn.init.orthogonal_(param[128 * 3 :], 1.0)

    def get_action(self, x: Tensor, action: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Get action given the extracted feaures randomly.

        Args:
            x (Tensor): features extracted from the observations
            action (Tensor, optional): action to compute the log-probabilites from. If None, then
                actions are sampled from a categorical distribution.
                Defaults to None.

        Returns:
            sampled action
            log-probabilities of the sampled action
            entropy distribution
        """
        logits = self.actor(x)
        distribution = Categorical(logits=logits.unsqueeze(-2))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()

    def get_greedy_action(self, obs: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Get action given the observation greedily.

        Args:
            obs (Tensor): input observation
            state (Tensor): recurrent state

        Returns:
            sampled action
            new recurrent state
        """
        x = self.actor_fc(obs)
        x, state = self.actor_rnn(x, state)
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1), state

    def get_value(self, x: Tensor) -> Tensor:
        """Get critic value given the input features.

        Args:
            x (Tensor): input features.

        Returns:
            critic value
        """
        return self.critic(x)

    def get_action_and_value(
        self, obs: Tensor, done: Tensor, action: Tensor = None, state: Tuple[Tensor, Tensor] = (None, None)
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """Compute actions, log-probabilities, distribution entropy and critic values.

        This model forward computes actions, related log-probabilities, entropy distribution
        and the critic values given the observations received as input. If `action` is None,
        then the actions are sampled from a categorical distribution.

        Args:
            obs (Tensor): observations collected
            done (Tensor): dones flag collected
            action (Tensor, optional): actions played given the observations. If None, actions
                are sampled from a categorical distribution.
                Defaults to None.
            state (Tensor, optional): the recurrent states.
                Defaults to None.

        Returns:
            sampled actions if `action` is None, themselves otherwise
            log-probabilites of the actions
            entropy of the distribution
            critic value
            next recurrent state for both the actor and the critic
        """
        actor_state, critic_state = state

        # If no done is found, then we can run through all the sequence.
        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/policies.py#L22
        run_through_all = False
        if torch.all(done == 0.0):
            run_through_all = True

        x_actor = self.actor_fc(obs)
        if run_through_all:
            actor_hidden, actor_state = self.actor_rnn(x_actor, actor_state)
        else:
            actor_hidden = torch.empty_like(x_actor)
            for i, (ah, d) in enumerate(zip(x_actor, done)):
                ah, actor_state = self.actor_rnn(ah.unsqueeze(0), (1.0 - d).view(1, -1, 1) * actor_state)
                actor_hidden[i] = ah
        action, log_prob, entropy = self.get_action(actor_hidden, action)

        x_critic = self.critic_fc(obs)
        if run_through_all:
            critic_hidden, critic_state = self.critic_rnn(x_critic, critic_state)
        else:
            critic_hidden = torch.empty_like(x_critic)
            for i, (ch, d) in enumerate(zip(x_actor, done)):
                ch, critic_state = self.actor_rnn(ch.unsqueeze(0), (1.0 - d).view(1, -1, 1) * critic_state)
                critic_hidden[i] = ch
        value = self.get_value(x_critic)
        return action, log_prob, entropy, value, (actor_state, critic_state)

    def forward(
        self, obs: Tensor, done: Tensor, action: Tensor = None, state: Tuple[Tensor, Tensor] = (None, None)
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """Forward method of the LightningModule.

        This model forward computes actions, related log-probabilities, entropy distribution
        and the critic values given the observations received as input. If `action` is None,
        then the actions are sampled from a categorical distribution.

        Args:
            obs (Tensor): observations collected
            done (Tensor): dones flag collected
            action (Tensor, optional): actions played given the observations. If None, actions
                are sampled from a categorical distribution.
                Defaults to None.
            state (Tuple[Tensor, Tensor], optional): the recurrent states.
                Defaults to None.

        Returns:
            sampled actions if `action` is None, themselves otherwise
            log-probabilites of the actions
            entropy of the distribution
            critic value
            next recurrent states for both the actor and the critic
        """
        return self.get_action_and_value(obs, done, action, state)

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
        state: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Compute returns and advantages following https://arxiv.org/abs/1506.02438

        Args:
            rewards (Tensor): all rewards collected from the last rollout
            values (Tensor): all values collected from the last rollout
            dones (Tensor): all dones collected from the last rollout
            next_obs (Tensor): next observation
            next_done (Tensor): next done
            num_steps (int): the number of steps played
            gamma (float): discout factor
            gae_lambda (float): lambda for GAE estimation
            state (Tuple[Tensor, Tensor]): recurrent state for both the actor and critic

        Returns:
            estimated returns
            estimated advantages
        """
        _, _, _, next_value, _ = self.get_action_and_value(next_obs, next_done, state=state)
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

    @torch.no_grad()
    def fast_estimate_returns_and_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_obs: Tensor,
        next_done: Tensor,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
        state: Tuple[Tensor, Tensor],
    ):
        """Compute returns and advantages following https://arxiv.org/abs/1506.02438

        Args:
            rewards (Tensor): all rewards collected from the last rollout
            values (Tensor): all values collected from the last rollout
            dones (Tensor): all dones collected from the last rollout
            next_obs (Tensor): next observation
            next_done (Tensor): next done
            num_steps (int): the number of steps played
            gamma (float): discout factor
            gae_lambda (float): lambda for GAE estimation
            state (Tuple[Tensor, Tensor]): recurrent state for both the actor and critic

        Returns:
            estimated returns
            estimated advantages
        """
        _, _, _, next_value, _ = self.get_action_and_value(next_obs, next_done, state=state)
        if len(rewards.shape) == 3:
            t_steps = torch.cat(
                [
                    conditional_arange(num_steps, dones[:, dim, :].view(-1)).view(-1, 1)
                    for dim in range(rewards.shape[1])
                ],
                dim=1,
            ).unsqueeze(-1)
        elif len(rewards.shape) == 2:
            t_steps = conditional_arange(num_steps, dones.view(-1)).view(-1, 1)
        else:
            raise ValueError(f"Shape must be 2 or 3 dimensional, got {rewards.shape}")
        gt = (gamma * gae_lambda) ** t_steps
        next_values = torch.roll(values, -1, dims=0)
        next_values[-1] = next_value
        next_dones = torch.roll(dones, -1, dims=0)
        next_dones[-1] = next_done
        deltas = rewards + gamma * next_values * (1 - next_dones) - values
        cs = torch.flipud(deltas * gt).cumsum(dim=0)
        acc = torch.cummax(torch.where(torch.flipud(dones.bool()), cs, 0), 0)[0]
        acc[0] = 0
        dones[-1] = 0
        # mask = dones.nonzero(as_tuple=True)
        # adv = torch.flipud(cs - acc) / gt
        # adv[mask] = deltas[mask] + gamma * gae_lambda * adv[mask[0] + 1, mask[1]]
        adv = torch.flipud(cs - acc) / gt
        adv = adv + dones * (deltas + gamma * gae_lambda * adv.roll(-1, 0))
        return adv + values, adv

    def training_step(self, batch: TensorDict, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Single training step over a batch of data.

        Args:
            batch (TensorDict): the batch to optimize from
            state (Tuple[Tensor, Tensor]): recurrent state for both the actor and critic

        Returns:
            computed loss
            new recurrent state for both the actor and the critic
        """
        # Get actions and values given the current observations
        _, newlogprob, entropy, newvalue, state = self(
            batch["observations"], batch["dones"], batch["actions"].long(), state
        )
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
        return pg_loss + ent_loss + v_loss, state

    def on_train_epoch_end(self, global_step: int) -> None:
        """Log metrics and reset them to their initial state.

        Args:
            global_step (int): the global optimization step
        """
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
        """Reset metrics state"""
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    def configure_optimizers(self, **optimizer_kwargs) -> Optimizer:
        """Configure the optimizers.

        Returns:
            the optimizer
        """
        return Adam(self.parameters(), **optimizer_kwargs)
