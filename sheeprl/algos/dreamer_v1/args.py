from dataclasses import dataclass
from typing import Optional

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class DreamerV1Args(StandardArgs):
    # Experiment settings
    share_data: bool = Arg(default=False, help="Toggle sharing data between processes")
    per_rank_batch_size: int = Arg(default=50, help="the batch size for each rank")
    per_rank_sequence_length: int = Arg(default=50, help="the sequence length for each rank")
    total_steps: int = Arg(default=int(5e6), help="total timesteps of the experiments")
    capture_video: bool = Arg(
        default=False, help="whether to capture videos of the agent performances (check out `videos` folder)"
    )
    buffer_size: int = Arg(default=int(5e6), help="the size of the buffer")
    learning_starts: int = Arg(default=int(5e3), help="timestep to start learning")
    gradient_steps: int = Arg(default=100, help="the number of gradient steps per each environment interaction")
    train_every: int = Arg(default=1000, help="the number of steps between one training and another")
    checkpoint_every: int = Arg(default=-1, help="how often to make the checkpoint, -1 to deactivate the checkpoint")
    checkpoint_buffer: bool = Arg(default=False, help="whether or not to save the buffer during the checkpoint")
    checkpoint_path: Optional[str] = Arg(default=None, help="the path of the checkpoint from which you want to restart")

    # Agent settings
    world_lr: float = Arg(default=6e-4, help="the learning rate of the optimizer of the world model")
    actor_lr: float = Arg(default=8e-5, help="the learning rate of the optimizer of the actor")
    critic_lr: float = Arg(default=8e-5, help="the learning rate of the optimizer of the critic")
    horizon: int = Arg(default=15, help="the number of imagination step")
    gamma: float = Arg(default=0.99, help="the discount factor gamma")
    lmbda: float = Arg(default=0.95, help="the lambda for the TD lambda values")
    use_continues: bool = Arg(default=False, help="wheter or not to use the continue predictor")
    stochastic_size: int = Arg(default=30, help="the dimension of the stochastic state")
    recurrent_state_size: int = Arg(default=200, help="the dimension of the recurrent state")
    kl_free_nats: float = Arg(default=3.0, help="the minimum value for the kl divergence")
    kl_regularizer: float = Arg(default=1.0, help="the scale factor for the kl divergence")
    continue_scale_factor: float = Arg(default=10.0, help="the scale factor for the continue loss")
    min_std: float = Arg(
        default=0.1, help="the minimum value of the standard deviation for the stochastic state distribution"
    )
    actor_mean_scale: float = Arg(default=5.0, help="the scale factor for the standard deviation of the actions")
    actor_init_std: float = Arg(
        default=5.0, help="the amout to sum to the input of the function of the standard deviation of the actions"
    )
    actor_min_std: float = Arg(default=1e-4, help="the minimum standard deviation for the actions")
    clip_gradients: float = Arg(default=100.0, help="how much to clip the gradient norms")
    dense_units: int = Arg(default=400, help="the number of units in dense layers, must be greater than zero")
    num_layers: int = Arg(
        default=4,
        help="the number of MLP layers for every model: actor, critic, reward and possibly the continue model",
    )
    cnn_channels_multiplier: int = Arg(default=32, help="cnn width multiplication factor, must be greater than zero")
    dense_act: str = Arg(
        default="ELU",
        help="the activation function for the dense layers, one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity (case sensitive, without 'nn.')",
    )
    cnn_act: str = Arg(
        default="ReLU",
        help="the activation function for the convolutional layers, one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity (case sensitive, without 'nn.')",
    )

    # Environment settings
    expl_amount: float = Arg(default=0.3, help="the exploration amout to add to the actions")
    expl_decay: bool = Arg(default=False, help="whether or not to decrement the exploration amount")
    expl_min: float = Arg(default=0.0, help="the minimum value for the exploration amout")
    max_step_expl_decay: int = Arg(default=200000, help="the maximum number of decay steps")
    envs_batch_size: int = Arg(
        default=2, help="the number of environments to be batched during a single Dreamer_v1 epoch"
    )
    action_repeat: int = Arg(default=2, help="the number of times an action is repeated")
    max_episode_steps: int = Arg(
        default=1000, help="the maximum duration in terms of number of steps of an episode, -1 to disable"
    )
    atari_noop_max: int = Arg(
        default=30,
        help="for No-op reset in Atari environment, the max number no-ops actions are taken at reset, to turn off, set to 0",
    )
    clip_rewards: bool = Arg(default=False, help="whether or not to clip rewards using tanh")
    grayscale_obs: bool = Arg(default=False, help="whether or not to the observations are grayscale")
