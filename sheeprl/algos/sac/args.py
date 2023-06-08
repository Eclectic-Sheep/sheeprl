from dataclasses import dataclass

from sheeprl.algos.args import StandardArgs
from sheeprl.utils.parser import Arg


@dataclass
class SACArgs(StandardArgs):
    env_id: str = Arg(default="LunarLanderContinuous-v2", help="the id of the environment")
    total_steps: int = Arg(default=1e6, help="total timesteps of the experiments")
    capture_video: bool = Arg(
        default=False, help="whether to capture videos of the agent performances (check out `videos` folder)"
    )
    buffer_size: int = Arg(default=int(1e6), help="the replay memory buffer size")
    gamma: float = Arg(default=0.99, help="the discount factor gamma")
    tau: float = Arg(default=0.005, help="target smoothing coefficient")
    alpha: float = Arg(default=1.0, help="entropy regularization coefficient")
    per_rank_batch_size: int = Arg(default=256, help="the batch size of sample from the reply memory for every rank")
    learning_starts: int = Arg(default=100, help="timestep to start learning")
    num_critics: int = Arg(default=2, help="the number of critics")
    q_lr: float = Arg(default=3e-4, help="the learning rate of the critic network optimizer")
    alpha_lr: float = Arg(default=3e-4, help="the learning rate of the policy network optimizer")
    policy_lr: float = Arg(default=3e-4, help="the learning rate of the entropy coefficient optimizer")
    target_network_frequency: int = Arg(default=1, help="the frequency of updates for the target nerworks")
    gradient_steps: int = Arg(default=1, help="the number of gradient steps per each environment interaction")
    checkpoint_every: int = Arg(default=-1, help="how often to make the checkpoint, -1 to deactivate the checkpoint")
    checkpoint_buffer: bool = Arg(default=False, help="whether or not to save the buffer during the checkpoint")
    sample_next_obs: bool = Arg(
        default=False, help="whether or not to sample the next observations from the gathered observations"
    )
    actor_hidden_sizes: int = Arg(default=256, help="the dimension of the hidden sizes of the actor network")
    critic_hidden_sizes: int = Arg(default=256, help="the dimension of the hidden sizes of the critic network")
