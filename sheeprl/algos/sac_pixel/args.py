from dataclasses import dataclass
from typing import List, Optional

from sheeprl.algos.sac.args import SACArgs
from sheeprl.utils.parser import Arg


@dataclass
class SACPixelContinuousArgs(SACArgs):
    env_id: str = Arg(default="CarRacing-v2", help="the id of the environment")
    num_envs: int = Arg(default=1, help="the number of parallel game environments")
    action_repeat: int = Arg(default=1, help="how many actions to repeat. Must be greater than 0.")
    frame_stack: int = Arg(default=3, help="how many frames to stack. 0 to disable.")
    screen_size: int = Arg(
        default=64, help="the dimension of the image rendered by the environment (screen_size x screen_size)"
    )
    learning_starts: int = Arg(default=1000, help="timestep to start learning")
    features_dim: int = Arg(default=64, help="the features dimension after the convolutional layer.")
    hidden_dim: int = Arg(default=1024, help="the dimension of the mlps for both the actor and the critic models")
    per_rank_batch_size: int = Arg(default=128, help="the batch size of sample from the reply memory for every rank")
    alpha: float = Arg(default=0.1, help="entropy regularization coefficient")
    q_lr: float = Arg(default=1e-3, help="the learning rate of the critic network optimizer")
    alpha_lr: float = Arg(default=1e-4, help="the learning rate of the policy network optimizer")
    policy_lr: float = Arg(default=1e-3, help="the learning rate of the entropy coefficient optimizer")
    encoder_lr: float = Arg(default=1e-3, help="learning rate for the encoder, used during the reconstruction loss")
    decoder_lr: float = Arg(default=1e-3, help="learning rate for the decoder, used during the reconstruction loss")
    decoder_wd: float = Arg(default=1e-7, help="weight decay for the decoder optimizer")
    decoder_l2_lambda: float = Arg(
        default=1e-6, help="weight to assign to the L2 penalization on the reconstruction loss"
    )
    decoder_update_freq: int = Arg(default=1, help="the update frequency of the decoder")
    actor_network_frequency: int = Arg(default=2, help="the frequency of updates for the actor nerworks")
    target_network_frequency: int = Arg(default=2, help="the frequency of updates for the target nerworks")
    tau: float = Arg(default=0.01, help="target smoothing coefficient for the critic ema")
    encoder_tau: float = Arg(default=0.05, help="target smoothing coefficient for the critic encoder ema")
    actor_hidden_size: int = Arg(default=1024, help="the dimension of the hidden sizes of the actor network")
    critic_hidden_size: int = Arg(default=1024, help="the dimension of the hidden sizes of the critic network")

    cnn_channels_multiplier: int = Arg(default=16, help="cnn width multiplication factor, must be greater than zero")
    dense_units: int = Arg(default=64, help="the number of units in dense layers, must be greater than zero")
    mlp_layers: int = Arg(
        default=2, help="the number of MLP layers for every model: actor, critic, continue and reward"
    )
    dense_act: str = Arg(
        default="ReLU",
        help="the activation function for the dense layers, one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity (case sensitive, without 'nn.')",
    )
    cnn_act: str = Arg(
        default="ReLU",
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
