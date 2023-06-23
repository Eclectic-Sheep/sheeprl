from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from sheeprl.algos.dreamer_v1.agent import RSSM


class RSSMP2E(RSSM):
    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        transition_model: nn.Module,
        min_std: Optional[float] = 0.1,
    ) -> None:
        super().__init__(recurrent_model, representation_model, transition_model, min_std)

    def dynamic(
        self, stochastic_state: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Perform one step of the dynamic learning:
            Recurrent model: compute the recurrent state from the previous latent space, the action taken by the agent,
                i.e., it computes the deterministic state (or ht).
            Transition model: predict the stochastic state from the recurrent output.
            Representation model: compute the stochasitc state from the recurrent state and from
                the embedded observations provided by the environment.
        For more information see [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551) and [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

        Args:
            stochastic_state (Tensor): the stochastic state.
            recurrent_state (Tensor): a tuple representing the recurrent state of the recurrent model.
            action (Tensor): the action taken by the agent.
            embedded_obs (Tensor): the embedded observations provided by the environment.

        Returns:
            The actual mean and std (Tuple[Tensor, Tensor]): the actual mean and std of the distribution of the latent state.
            The recurrent state (Tuple[Tensor, ...]): the recurrent state of the recurrent model.
            The stochastic state (Tensor): computed by the representation model from the recurrent state and the embbedded observation, aka posterior.
            The predicted mean and std (Tuple[Tensor, Tensor]): the predicted mean and std of the distribution of the latent state.
            The predicted stochastic state (Tensor): computed by the transition model, aka prior.
        """
        recurrent_out, recurrent_state = self.recurrent_model(
            torch.cat((stochastic_state, action), -1), recurrent_state
        )
        predicted_state_mean_std, predicted_stochastic_state = self._transition(recurrent_out)
        state_mean_std, stochastic_state = self._representation(recurrent_state, embedded_obs)
        return state_mean_std, recurrent_state, stochastic_state, predicted_state_mean_std, predicted_stochastic_state
