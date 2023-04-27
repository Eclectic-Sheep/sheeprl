import argparse
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default", help="the name of this experiment")

    # PyTorch arguments
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, GPU training will be used. "
        "This affects also the distributed backend used (NCCL (gpu) vs GLOO (cpu))",
    )
    parser.add_argument(
        "--player-on-gpu",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If toggled, player will run on GPU (used only by `train_fabric_decoupled.py` script). "
        "This affects also the distributed backend used (NCCL (gpu) vs GLOO (cpu))",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=True`",
    )

    # Distributed arguments
    parser.add_argument("--num-envs", type=int, default=4, help="the number of parallel game environments")
    parser.add_argument(
        "--share-data",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle sharing data between processes",
    )
    parser.add_argument("--per-rank-batch-size", type=int, default=64, help="the batch size for each rank")

    # Environment arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2**16, help="total timesteps of the experiments")
    parser.add_argument(
        "--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout"
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )
    parser.add_argument(
        "--mask-vel",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to mask velocities",
    )

    # PPO arguments
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="the learning rate of the optimizer")
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation"
    )
    parser.add_argument("--update-epochs", type=int, default=10, help="the K epochs to update the policy")
    parser.add_argument(
        "--envs-batch-size",
        type=int,
        default=2,
        help="the number of environments to be batched during a single PPO epoch",
    )
    parser.add_argument(
        "--activation-function",
        type=str,
        default="relu",
        choices=["relu", "tanh"],
        help="The activation function of the model",
    )
    parser.add_argument(
        "--ortho-init",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles the orthogonal initialization of the model",
    )
    parser.add_argument(
        "--normalize-advantages",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument("--ent-coef", type=float, default=0.0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    args = parser.parse_args()
    return args
