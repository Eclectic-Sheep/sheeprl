from sheeprl.utils.utils import Ratio

if __name__ == "__main__":
    num_envs = 1
    world_size = 1
    replay_ratio = 0.0625
    per_rank_batch_size = 16
    per_rank_sequence_length = 64
    replayed_steps = world_size * per_rank_batch_size * per_rank_sequence_length
    train_steps = 0
    gradient_steps = 0
    total_policy_steps = 2**10
    r = Ratio(ratio=replay_ratio, pretrain_steps=0)
    policy_steps = num_envs * world_size
    printed = False
    for i in range(0, total_policy_steps, policy_steps):
        if i >= 128:
            per_rank_repeats = r(i / world_size)
            if per_rank_repeats > 0:  # and not printed:
                print(
                    f"Training the agent with {per_rank_repeats} repeats on every rank "
                    f"({per_rank_repeats * world_size} global repeats) at global iteration {i}"
                )
                printed = True
            gradient_steps += per_rank_repeats * world_size
    print("Replay ratio", replay_ratio)
    print("Hafner train ratio", replay_ratio * replayed_steps)
    print("Final ratio", gradient_steps / total_policy_steps)
