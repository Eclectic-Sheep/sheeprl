from dataclasses import dataclass
from typing import List, Optional

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class PPOArgs(StandardArgs):
    share_data: bool = Arg(default=False, help="Toggle sharing data between processes")
    per_rank_batch_size: int = Arg(default=64, help="the batch size for each rank")
    total_steps: int = Arg(default=2**16, help="total timesteps of the experiments")
    rollout_steps: int = Arg(default=128, help="the number of steps to run in each environment per policy rollout")
    capture_video: bool = Arg(
        default=False, help="whether to capture videos of the agent performances (check out `videos` folder)"
    )
    mask_vel: bool = Arg(default=False, help="whether to mask velocities")
    lr: float = Arg(default=1e-3, help="the learning rate of the optimizer")
    anneal_lr: bool = Arg(default=False, help="Toggle learning rate annealing for policy and value networks")
    gamma: float = Arg(default=0.99, help="the discount factor gamma")
    gae_lambda: float = Arg(default=0.95, help="the lambda for the general advantage estimation")
    update_epochs: int = Arg(default=10, help="the K epochs to update the policy")
    loss_reduction: str = Arg(
        default="mean", metadata={"choices": ("mean", "sum", "none")}, help="Which reduction to use"
    )
    normalize_advantages: bool = Arg(default=False, help="Toggles advantages normalization")
    clip_coef: float = Arg(default=0.2, help="the surrogate clipping coefficient")
    anneal_clip_coef: bool = Arg(default=False, help="whether to linearly anneal the clip coefficient to zero")
    clip_vloss: bool = Arg(
        default=False, help="Toggles whether or not to use a clipped loss for the value function, as per the paper."
    )
    ent_coef: float = Arg(default=0.0, help="coefficient of the entropy")
    anneal_ent_coef: bool = Arg(default=False, help="whether to linearly anneal the entropy coefficient to zero")
    vf_coef: float = Arg(default=1.0, help="coefficient of the value function")
    max_grad_norm: float = Arg(default=0.0, help="the maximum norm for the gradient clipping")
    actor_hidden_size: int = Arg(default=64, help="the dimension of the hidden sizes of the actor network")
    critic_hidden_size: int = Arg(default=64, help="the dimension of the hidden sizes of the critic network")

    dense_units: int = Arg(default=64, help="the number of units in dense layers, must be greater than zero")
    mlp_layers: int = Arg(
        default=2, help="the number of MLP layers for every model: actor, critic, continue and reward"
    )
    cnn_channels_multiplier: int = Arg(default=1, help="cnn width multiplication factor, must be greater than zero")
    dense_act: str = Arg(
        default="Tanh",
        help="the activation function for the dense layers, one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity (case sensitive, without 'nn.')",
    )
    cnn_act: str = Arg(
        default="Tanh",
        help="the activation function for the convolutional layers, one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity (case sensitive, without 'nn.')",
    )
    layer_norm: bool = Arg(
        default=False, help="whether to apply nn.LayerNorm after every Linear/Conv2D/ConvTranspose2D"
    )
    grayscale_obs: bool = Arg(default=False, help="whether or not to the observations are grayscale")
    cnn_keys: Optional[List[str]] = Arg(
        default=None, help="a list of observation keys to be processed by the CNN encoder"
    )
    mlp_keys: Optional[List[str]] = Arg(
        default=None, help="a list of observation keys to be processed by the MLP encoder"
    )
    eps: float = Arg(default=1e-4)
    max_episode_steps: int = Arg(default=-1, help="the maximum amount of steps in an episode")
    cnn_features_dim: int = Arg(default=512, help="the features dimension after the CNNEncoder")
    mlp_features_dim: int = Arg(default=64, help="the features dimension after the MLPEncoder")
    atari_noop_max: int = Arg(default=30, help="the maximum number of noop in Atari envs on reset")

    diambra_action_space: str = Arg(
        default="discrete", help="the type of action space: one in [discrete, multi_discrete]"
    )
    diambra_attack_but_combination: bool = Arg(
        default=True, help="whether or not to enable the attack button combination in the action space"
    )
    diambra_noop_max: int = Arg(default=0, help="the maximum number of noop actions after the reset")
    diambra_actions_stack: int = Arg(default=1, help="the number of actions to stack in the observations")
