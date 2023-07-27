from dataclasses import dataclass
from typing import List, Optional

from sheeprl.algos.dreamer_v2.args import DreamerV2Args
from sheeprl.utils.parser import Arg


@dataclass
class DreamerV3Args(DreamerV2Args):
    env_id: str = Arg(default="dmc_walker_walk", help="the id of the environment")

    # Experiment settings
    share_data: bool = Arg(default=False, help="Toggle sharing data between processes")
    per_rank_batch_size: int = Arg(default=16, help="the batch size for each rank")
    per_rank_sequence_length: int = Arg(default=50, help="the sequence length for each rank")
    total_steps: int = Arg(default=int(5e6), help="total timesteps of the experiments")
    capture_video: bool = Arg(
        default=False, help="whether to capture videos of the agent performances (check out `videos` folder)"
    )
    buffer_size: int = Arg(default=int(5e6), help="the size of the buffer")
    learning_starts: int = Arg(default=int(5e3), help="timestep to start learning")
    pretrain_steps: int = Arg(default=100, help="the number of pretrain steps")
    gradient_steps: int = Arg(default=1, help="the number of gradient steps per each environment interaction")
    train_every: int = Arg(default=5, help="the number of steps between one training and another")
    checkpoint_every: int = Arg(default=-1, help="how often to make the checkpoint, -1 to deactivate the checkpoint")
    checkpoint_buffer: bool = Arg(default=False, help="whether or not to save the buffer during the checkpoint")
    checkpoint_path: Optional[str] = Arg(default=None, help="the path of the checkpoint from which you want to restart")
    buffer_type: str = Arg(
        default="sequential",
        help="which buffer to use: `sequential` or `episode`. The `episode` "
        "buffer will save an entire episode, while the sequential will save every step.",
    )
    prioritize_ends: bool = Arg(default=False, help="whether to sample episodes prioritizing the end of them.")

    # Agent settings
    world_lr: float = Arg(default=1e-4, help="the learning rate of the optimizer of the world model")
    actor_lr: float = Arg(default=3e-5, help="the learning rate of the optimizer of the actor")
    critic_lr: float = Arg(default=3e-5, help="the learning rate of the optimizer of the critic")
    horizon: int = Arg(default=15, help="the number of imagination step")
    gamma: float = Arg(default=(1 - 1 / 333), help="the discount factor gamma")
    lmbda: float = Arg(default=0.95, help="the lambda for the TD lambda values")
    use_continues: bool = Arg(default=True, help="wheter or not to use the continue predictor")
    stochastic_size: int = Arg(default=32, help="the dimension of the stochastic state")
    discrete_size: int = Arg(default=32, help="the dimension of the discrete state")
    hidden_size: int = Arg(default=768, help="the hidden size for the transition and representation model")
    recurrent_state_size: int = Arg(default=2048, help="the dimension of the recurrent state")
    kl_dynamic: float = Arg(default=0.5, help="the regularizer for the KL dynamic loss")
    kl_representation: float = Arg(default=0.1, help="the regularizer for the KL representation loss")
    kl_free_nats: float = Arg(default=1.0, help="the minimum value for the kl divergence")
    kl_regularizer: float = Arg(default=1.0, help="the scale factor for the kl divergence")
    continue_scale_factor: float = Arg(default=1.0, help="the scale factor for the continue loss")
    actor_ent_coef: float = Arg(default=3e-4, help="the entropy coefficient for the actor loss")
    actor_init_std: float = Arg(
        default=0.0, help="the amout to sum to the input of the function of the standard deviation of the actions"
    )
    actor_min_std: float = Arg(default=0.1, help="the minimum standard deviation for the actions")
    actor_distribution: str = Arg(
        default="auto",
        help="the actor distribution. One can chose between `auto`, `discrete` (one-hot categorical), "
        "`normal`, `tanh_normal` and `trunc_normal`. If `auto`, then the distribution will be a one-hot categorical if "
        "the action space is discrete, otherwise it will be a truncated normal distribution.",
    )
    world_clip_gradients: float = Arg(default=1000.0, help="how much to clip the gradient norms")
    actor_clip_gradients: float = Arg(default=100.0, help="how much to clip the gradient norms")
    critic_clip_gradients: float = Arg(default=100.0, help="how much to clip the gradient norms")
    dense_units: int = Arg(default=768, help="the number of units in dense layers, must be greater than zero")
    mlp_layers: int = Arg(
        default=4, help="the number of MLP layers for every model: actor, critic, continue and reward"
    )
    cnn_channels_multiplier: int = Arg(default=64, help="cnn width multiplication factor, must be greater than zero")
    dense_act: str = Arg(
        default="SiLU",
        help="the activation function for the dense layers, one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity (case sensitive, without 'nn.')",
    )
    cnn_act: str = Arg(
        default="SiLU",
        help="the activation function for the convolutional layers, one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity (case sensitive, without 'nn.')",
    )
    critic_target_network_update_freq: int = Arg(default=1, help="the frequency to update the target critic network")
    layer_norm: bool = Arg(default=True, help="whether to apply nn.LayerNorm after every Linear/Conv2D/ConvTranspose2D")
    critic_tau: float = Arg(
        default=0.02,
        help="tau value to be used for the EMA critic update: `critic_param * tau + (1 - tau) * target_critic_param`",
    )
    unimix: float = Arg(
        default=0.01, help="whether to use a mix of uniform and categorical for the stochastic state distribution"
    )

    # Environment settings
    expl_amount: float = Arg(default=0.0, help="the exploration amout to add to the actions")
    expl_decay: bool = Arg(default=False, help="whether or not to decrement the exploration amount")
    expl_min: float = Arg(default=0.0, help="the minimum value for the exploration amout")
    max_step_expl_decay: int = Arg(default=0, help="the maximum number of decay steps")
    action_repeat: int = Arg(default=1, help="the number of times an action is repeated")
    max_episode_steps: int = Arg(
        default=-1, help="the maximum duration in terms of number of steps of an episode, -1 to disable"
    )
    atari_noop_max: int = Arg(
        default=0,
        help="for No-op reset in Atari environment, the max number no-ops actions are taken at reset, to turn off, set to 0",
    )
    clip_rewards: bool = Arg(default=False, help="whether or not to clip rewards using tanh")
    grayscale_obs: bool = Arg(default=False, help="whether or not to the observations are grayscale")
    cnn_keys: Optional[List[str]] = Arg(
        default=None, help="a list of observation keys to be processed by the CNN encoder"
    )
    mlp_keys: Optional[List[str]] = Arg(
        default=None, help="a list of observation keys to be processed by the MLP encoder"
    )
    mine_min_pitch: int = Arg(default=-60, help="The minimum value of pitch in Minecraft environmnets.")
    mine_max_pitch: int = Arg(default=60, help="The maximum value of pitch in Minecraft environmnets.")
    mine_start_position: Optional[List[str]] = Arg(
        default=None, help="The starting position of the agent in Minecraft environment. (x, y, z, pitch, yaw)"
    )
    minerl_dense: bool = Arg(default=False, help="whether or not the task has dense reward")
    minerl_extreme: bool = Arg(default=False, help="whether or not the task is extreme")
    mine_break_speed: int = Arg(default=100, help="the break speed multiplier of Minecraft environments")
    mine_sticky_attack: int = Arg(default=30, help="the sticky value for the attack action")
    mine_sticky_jump: int = Arg(default=10, help="the sticky value for the jump action")

    moments_decay: float = Arg(default=0.99, help="")
    moment_max: float = Arg(default=1.0, help="")
    moments_perclo: float = Arg(default=0.05, help="")
    moments_perchi: float = Arg(default=0.95, help="")

    bins: int = Arg(default=255, help="")
