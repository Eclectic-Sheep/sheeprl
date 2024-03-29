import warnings


class Ratio:
    """Directly taken from Hafner et al. (2023) implementation:
    https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26
    """

    def __init__(self, ratio: float, pretrain_steps: int = 0):
        if pretrain_steps < 0:
            raise ValueError(f"'pretrain_steps' must be non-negative, got {pretrain_steps}")
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._pretrain_steps = pretrain_steps
        self._ratio = ratio
        self._prev = None

    def __call__(self, step) -> int:
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            if self._pretrain_steps > 0:
                if step < self._pretrain_steps:
                    warnings.warn(
                        "The number of pretrain steps is greater than the number of current steps. This could lead to "
                        f"a higher ratio than the one specified ({self._ratio}). Setting the 'pretrain_steps' equal to "
                        "the number of current steps."
                    )
                    self._pretrain_steps = step
                return round(self._pretrain_steps * self._ratio)
            else:
                return 1
        repeats = round((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


if __name__ == "__main__":
    num_envs = 1
    world_size = 1
    replay_ratio = 0.5
    per_rank_batch_size = 16
    per_rank_sequence_length = 64
    replayed_steps = world_size * per_rank_batch_size * per_rank_sequence_length
    train_steps = 0
    gradient_steps = 0
    total_policy_steps = 2**10
    r = Ratio(ratio=replay_ratio, pretrain_steps=256)
    policy_steps = num_envs * world_size
    printed = False
    for i in range(0, total_policy_steps, policy_steps):
        if i >= 128:
            per_rank_repeats = r(i / world_size)
            if per_rank_repeats > 0 and not printed:
                print(
                    f"Training the agent with {per_rank_repeats} repeats on every rank "
                    f"({per_rank_repeats * world_size} global repeats) at global iteration {i}"
                )
                printed = True
            gradient_steps += per_rank_repeats * world_size
    print("Replay ratio", replay_ratio)
    print("Hafner train ratio", replay_ratio * replayed_steps)
    print("Final ratio", gradient_steps / total_policy_steps)
