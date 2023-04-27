from copy import deepcopy
from typing import Tuple

import torch
from lightning.pytorch import LightningModule
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential
from torch import Tensor
from torchmetrics import MeanMetric

from fabricrl.algos.ppo.loss import entropy_loss, policy_loss, value_loss


class PPOAgent(LightningModule):
    """PPO Agent"""

    def __init__(
        self,
        feature_extractor: TensorDictSequential,
        actor: TensorDictSequential,
        critic: TensorDictSequential,
        vf_coef: float = 1.0,
        ent_coef: float = 0.0,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        normalize_advantages: bool = False,
        **torchmetrics_kwargs,
    ):
        """PPO Agent"""
        super().__init__()

        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages

        self.feature_extractor = feature_extractor
        self.actor = actor
        self.critic = critic

        self.avg_pg_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_value_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_ent_loss = MeanMetric(**torchmetrics_kwargs)

    def forward(
        self,
        x: TensorDict,
    ) -> Tuple[Tensor, Tensor]:
        """Get action and value from the actor and critic networks"""
        tensordict_out = TensorDict({}, batch_size=x.batch_size, device=x.device)
        self.feature_extractor(x)
        self.actor(x, tensordict_out=tensordict_out)
        self.critic(x, tensordict_out=tensordict_out)
        return tensordict_out

    @torch.no_grad()
    def estimate_returns_and_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_step_data: TensorDict,
        num_steps: int,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[Tensor, Tensor]:
        """Estimate returns and advantages using GAE."""
        self.critic(next_step_data)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = torch.logical_not(next_step_data["dones"])
                nextvalues = next_step_data["values"]
            else:
                nextnonterminal = torch.logical_not(dones[t + 1])
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return returns, advantages

    def training_step(self, old_batch: TensorDict):
        # Get actions and values given the current observations
        new_batch = deepcopy(old_batch)
        new_batch = self(new_batch)
        actions_prob = self.actor.policy.actions_prob
        new_logprobs = self.actor.policy.get_logprob(old_batch["actions"])
        entropy = torch.sum(torch.stack([actions_prob[i].entropy() for i in range(len(actions_prob))], dim=-1))
        logratio = new_logprobs - old_batch["logprobs"]
        ratio = logratio.exp()

        # Policy loss
        advantages = new_batch["advantages"]
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = policy_loss(advantages, ratio, self.clip_coef)

        # Value loss
        v_loss = value_loss(
            new_batch["values"],
            old_batch["values"],
            new_batch["returns"],
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
        """Log metrics at the end of the training epoch"""
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
        """Reset metrics"""
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    def configure_optimizers(self, lr: float):
        """Configure optimizers"""
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
