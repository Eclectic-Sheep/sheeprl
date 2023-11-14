from typing import Optional, Tuple

import torch

from sheeprl.models.models import MLP, NatureCNN


class MuzeroAgent(torch.nn.Module):
    """
    Muzero agent.

    Args:
        representation: torch.nn.Module
            Representation function, map the observation to a latent space (hidden_state).
            Used to start the imagination process during MCTS.
        prediction: torch.nn.Module
            Prediction function, map the hidden state to policy logits and value.
            Used to predict the policy and value during MCTS, based on the latent space representation of observations.
        dynamics: torch.nn.Module
            Dynamics function, map an action and a hidden state to the imagined reward and next hidden state.
            Used to imagine the response (evolution) of the environment during MCTS.
    """

    def __init__(
        self,
        representation: Optional[torch.nn.Module] = None,
        prediction: Optional[torch.nn.Module] = None,
        dynamics: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.representation: torch.nn.Module = representation
        self.prediction: torch.nn.Module = prediction
        self.dynamics: torch.nn.Module = dynamics
        self.training_steps = 0

    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the initial hidden state and the initial prediction (policy and value) from the observation.
        To run to initialize MCTS.
        """
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state.unsqueeze(0), policy_logits, value

    def recurrent_inference(
        self, action: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the next hidden state and the next prediction (policy and value)
        from the action and the previous hidden state.
        To run at each simulation of MCTS.
        """
        reward, next_hidden_state = self.dynamics(action, hidden_state)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

    def forward(self, action, hidden_state):
        return self.recurrent_inference(action, hidden_state)

    @torch.no_grad()
    def gradient_norm(self):
        """Compute the norm of the parameters' gradients."""
        total_norm = 0
        p_with_grads = [p for p in self.parameters() if p.grad is not None]
        if not p_with_grads:
            raise RuntimeError("No parameters have gradients. Run the backward method first.")
        for p in p_with_grads:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm**0.5


class GruMlpDynamics(torch.nn.Module):
    def __init__(self, embedding_size=256, mlp_hidden_sizes=(64, 64), full_support_size=601):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=embedding_size)
        self.mlp = MLP(
            input_dims=embedding_size,
            hidden_sizes=mlp_hidden_sizes,
            activation=torch.nn.ELU,
            output_dim=full_support_size,
        )

    def forward(self, x, h0):
        y, h1 = self.gru(x, h0)
        y = self.mlp(y)
        return y, h1


class MlpDynamics(torch.nn.Module):
    def __init__(
        self,
        num_actions,
        embedding_size=256,
        state_hidden_sizes=(64, 64, 16),
        rew_hidden_sizes=(64, 64, 16),
        state_activation_fn=torch.nn.ELU,
        rew_activation_fn=torch.nn.ELU,
        full_support_size=601,
    ):
        super().__init__()
        self.hstate = MLP(
            input_dims=embedding_size + int(num_actions),
            hidden_sizes=state_hidden_sizes,
            output_dim=embedding_size,
            activation=state_activation_fn,
        )
        self.mlp = MLP(
            input_dims=embedding_size + int(num_actions),
            hidden_sizes=rew_hidden_sizes,
            activation=rew_activation_fn,
            output_dim=full_support_size,
        )
        self.num_actions = num_actions

    def forward(self, x, h0):
        one_hot_action = torch.nn.functional.one_hot(x.long(), num_classes=self.num_actions).float().transpose(0, 1)
        cat_input = torch.cat([one_hot_action, h0], dim=-1)
        h1 = self.hstate(cat_input)
        y = self.mlp(cat_input)
        return y, h1


class Predictor(torch.nn.Module):
    def __init__(
        self,
        embedding_size=256,
        policy_hidden_sizes=(64, 64, 16),
        value_hidden_sizes=(64, 64, 16),
        num_actions=4,
        full_support_size=601,
    ):
        super().__init__()
        self.mlp_actor = MLP(
            input_dims=embedding_size, hidden_sizes=policy_hidden_sizes, activation=torch.nn.ELU, output_dim=num_actions
        )

        self.mlp_value = MLP(
            input_dims=embedding_size,
            hidden_sizes=value_hidden_sizes,
            activation=torch.nn.ELU,
            output_dim=full_support_size,
        )

    def forward(self, x):
        policy = self.mlp_actor(x)
        value = self.mlp_value(x)
        return policy, value


class RecurrentMuzero(MuzeroAgent):
    def __init__(self, hidden_state_size=256, num_actions=4):
        super().__init__()
        self.representation = NatureCNN(in_channels=3, features_dim=hidden_state_size)
        self.dynamics: torch.nn.Module = GruMlpDynamics(mlp_hidden_sizes=hidden_state_size)
        self.prediction: torch.nn.Module = Predictor(policy_hidden_sizes=hidden_state_size, num_actions=num_actions)


if __name__ == "__main__":
    batch_size = 32
    sequence_len = 5
    agent = RecurrentMuzero()
    # Player:
    print("Player:")
    observation = torch.rand(1, 3, 64, 64)
    hidden_state, policy_logits, value = agent.initial_inference(observation)
    print("Initial inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(value.shape)

    action = torch.rand(1, 1, 1)
    hidden_state, policy_logits, reward, value = agent.recurrent_inference(action, hidden_state)
    print("Recurrent inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(reward.shape)
    print(value.shape)

    ## Trainer:
    print("Trainer:")
    observation = torch.rand(batch_size, 3, 64, 64)
    hidden_state, policy_logits, value = agent.initial_inference(observation)
    print("Initial inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(value.shape)

    action = torch.rand(1, batch_size, 1)
    hidden_state, policy_logits, reward, value = agent.recurrent_inference(action, hidden_state)
    print("Recurrent inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(reward.shape)
    print(value.shape)
    # action2 = torch.randint(0, 4, (1, 1)).to(torch.float32)
    # last_hidden_state, reward, policy_logits, value = agent.recurrent_inference(action2, next_hidden_state)
    # print(last_hidden_state.shape)
    # print(reward.shape)
    # print(policy_logits.shape)
    # print(value.shape)
