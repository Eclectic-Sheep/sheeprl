"""The gradient steps of an off-policy algorithm with a given replay ratio.

At the end of every iteration, the training loop (`sheeprl.core.loop.run`) does the gradient steps given by
`TrainSchedule.gradient_steps`: none until `algo.learning_starts` policy steps have filled the replay buffer, then
`algo.replay_ratio` gradient steps per policy step (`sheeprl.utils.utils.Ratio`).
"""

from sheeprl.core import TrainSchedule
from sheeprl.utils.utils import dotdict

if __name__ == "__main__":
    num_envs = 1
    world_size = 1
    replay_ratio = 0.0625
    learning_starts = 128
    per_rank_batch_size = 16
    per_rank_sequence_length = 64
    # The steps replayed by one gradient step of one process
    replayed_steps = per_rank_batch_size * per_rank_sequence_length
    gradient_steps = 0
    total_policy_steps = 2**10
    cfg = dotdict(
        {
            "dry_run": False,
            "env": {"num_envs": num_envs},
            "algo": {
                "total_steps": total_policy_steps,
                "learning_starts": learning_starts,
                "replay_ratio": replay_ratio,
                "per_rank_pretrain_steps": 0,
            },
            "metric": {"log_level": 0},
            "checkpoint": {"every": total_policy_steps},
        }
    )
    # The off-policy algorithms play one step in every environment per iteration
    schedule = TrainSchedule(cfg, world_size, steps_per_iteration=1, off_policy=True)
    for iteration in schedule.iterations():
        per_rank_repeats = schedule.gradient_steps(iteration)
        if per_rank_repeats > 0:
            print(
                f"Training the agent with {per_rank_repeats} repeats on every rank "
                f"({per_rank_repeats * world_size} global repeats) at policy step "
                f"{iteration * schedule.policy_steps_per_iter}"
            )
        gradient_steps += per_rank_repeats * world_size
    print("Replay ratio", replay_ratio)
    print("Hafner train ratio", replay_ratio * replayed_steps)
    # The `Params/replay_ratio` logged at the end of the training: the policy steps include the `learning_starts` ones
    print("Final ratio", gradient_steps / total_policy_steps)
