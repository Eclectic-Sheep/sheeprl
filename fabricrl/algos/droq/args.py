import argparse
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default", help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=True`",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LunarLander-v2", help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=4, help="the number of parallel game environments")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Entropy regularization coefficient.")
    parser.add_argument("--batch-size", type=int, default=256, help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3, help="timestep to start learning")
    parser.add_argument(
        "--policy-lr", type=float, default=3e-2, help="the learning rate of the policy network optimizer"
    )
    parser.add_argument("--q-lr", type=float, default=3e-2, help="the learning rate of the Q network network optimizer")
    parser.add_argument(
        "--alpha-lr", type=float, default=3e-2, help="the learning rate of the entropy coefficient parameter optimizer"
    )
    parser.add_argument(
        "--gradient-steps", type=int, default=20, help="the number of gradient steps per each environment interaction"
    )
    args = parser.parse_args()
    return args
