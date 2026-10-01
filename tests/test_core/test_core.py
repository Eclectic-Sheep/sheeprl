import copy
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from lightning import Fabric
from torch import Tensor, nn

# `setup_module` is not imported by name: pytest would run it as the setup of this test module
from sheeprl import core
from sheeprl.core import EnvStep, TrainSchedule, TrainState, load_replay_buffer, update
from sheeprl.core.algorithm import load_module_state_dict
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.utils import dotdict


@pytest.fixture()
def fabric():
    return Fabric(accelerator="cpu", devices=1, precision="32-true")


def schedule_cfg(total_steps=100, dry_run=False, learning_starts=10, replay_ratio=1.0, run_benchmarks=False):
    return dotdict(
        {
            "env": {"num_envs": 2},
            "algo": {
                "total_steps": total_steps,
                # Read only by the off-policy algorithms
                "learning_starts": learning_starts,
                "replay_ratio": replay_ratio,
                "per_rank_pretrain_steps": 0,
            },
            "dry_run": dry_run,
            "run_benchmarks": run_benchmarks,
            "metric": {"log_level": 0, "log_every": 16},
            "checkpoint": {"every": 0},
        }
    )


def test_train_schedule_counts_iterations_in_policy_steps_of_all_processes():
    schedule = TrainSchedule(schedule_cfg(total_steps=100), world_size=2, steps_per_iteration=4)
    assert schedule.policy_steps_per_rank == 8
    assert schedule.policy_steps_per_iter == 16
    assert schedule.total_iters == 6
    assert list(schedule.iterations()) == [1, 2, 3, 4, 5, 6]
    assert (schedule.policy_step, schedule.gradient_step) == (0, 0)
    # On-policy: no random actions, and the algorithm decides the number of gradient steps
    assert not schedule.off_policy
    assert not schedule.warmup(policy_step=0)
    assert schedule.gradient_steps(iteration=1) is None


def test_train_schedule_resumes_after_the_checkpointed_iteration():
    # The checkpoint of iteration 3, saved by 2 processes
    checkpoint = {"iter_num": 3 * 2, "per_rank_gradient_steps": 30}
    schedule = TrainSchedule(schedule_cfg(total_steps=100), world_size=2, steps_per_iteration=4, checkpoint=checkpoint)
    assert list(schedule.iterations()) == [4, 5, 6]
    assert schedule.policy_step == 3 * 16
    assert schedule.gradient_step == 30


def test_train_schedule_dry_run_does_one_iteration():
    schedule = TrainSchedule(schedule_cfg(total_steps=1000, dry_run=True), world_size=1, steps_per_iteration=4)
    assert list(schedule.iterations()) == [1]


def test_off_policy_schedule_plays_random_actions_then_follows_the_replay_ratio():
    # 2 envs, 1 process: 2 policy steps per iteration; `learning_starts=10` gives 5 iterations of random actions
    schedule = TrainSchedule(schedule_cfg(learning_starts=10), world_size=1, steps_per_iteration=1, off_policy=True)
    assert schedule.off_policy
    assert [schedule.warmup(policy_step=2 * (i - 1)) for i in range(1, 8)] == [True] * 5 + [False] * 2
    # The training starts in the last iteration of random actions, with the steps of that iteration (ratio 1)
    assert [schedule.gradient_steps(i) for i in range(1, 9)] == [0, 0, 0, 0, 2, 2, 2, 2]


def test_off_policy_schedule_with_a_fractional_replay_ratio():
    schedule = TrainSchedule(
        schedule_cfg(learning_starts=0, replay_ratio=0.25), world_size=1, steps_per_iteration=1, off_policy=True
    )
    assert not schedule.warmup(policy_step=0)
    # One gradient step every 4 policy steps, i.e. every other iteration, from the 4 steps after the first iteration
    assert [schedule.gradient_steps(i) for i in range(1, 7)] == [0, 0, 1, 0, 1, 0]


def test_off_policy_schedule_splits_the_replay_ratio_among_the_processes():
    # 2 processes of 2 envs: every process does half of the gradient steps of the 4 policy steps of an iteration
    schedule = TrainSchedule(schedule_cfg(learning_starts=0), world_size=2, steps_per_iteration=1, off_policy=True)
    assert [schedule.gradient_steps(i) for i in range(1, 4)] == [2, 2, 2]


def test_off_policy_schedule_benchmarks_and_dry_run():
    benchmark = schedule_cfg(learning_starts=0, replay_ratio=3.0, run_benchmarks=True)
    schedule = TrainSchedule(benchmark, world_size=1, steps_per_iteration=1, off_policy=True)
    assert [schedule.gradient_steps(i) for i in range(1, 4)] == [1, 1, 1]
    # A dry run plays one iteration, without random actions
    dry_run = schedule_cfg(dry_run=True, learning_starts=1000)
    schedule = TrainSchedule(dry_run, world_size=1, steps_per_iteration=1, off_policy=True)
    assert list(schedule.iterations()) == [1]
    assert not schedule.warmup(policy_step=0)
    assert schedule.gradient_steps(1) == 2


def test_off_policy_schedule_resumes_the_replay_ratio():
    # The checkpoint of iteration 8 of the first test: the ratio counted 8 policy steps (iterations 5 to 8)
    schedule = TrainSchedule(schedule_cfg(learning_starts=10), world_size=1, steps_per_iteration=1, off_policy=True)
    for i in range(1, 9):
        schedule.gradient_steps(i)
    checkpoint = {"iter_num": 8, "per_rank_gradient_steps": 8, "ratio": schedule.ratio.state_dict()}
    resumed = TrainSchedule(
        schedule_cfg(learning_starts=10), world_size=1, steps_per_iteration=1, checkpoint=checkpoint, off_policy=True
    )
    assert resumed.ratio.state_dict() == schedule.ratio.state_dict()
    assert resumed.gradient_step == 8
    # As before the port, a resumed run plays random actions again for `learning_starts` steps, and doesn't train
    # until its ratio catches up (known issue #42)
    assert list(resumed.iterations())[:2] == [9, 10]
    assert resumed.warmup(policy_step=16)
    assert resumed.gradient_steps(9) == 0


def test_load_replay_buffer_takes_the_buffer_of_the_process(fabric):
    buffer = ReplayBuffer(4, n_envs=1)
    saved = [ReplayBuffer(4, n_envs=1)]
    assert load_replay_buffer(fabric, saved, buffer) is saved[0]
    # A single saved buffer is the one of every process
    assert load_replay_buffer(fabric, saved[0], buffer) is saved[0]
    with pytest.raises(RuntimeError, match="2 replay buffer"):
        load_replay_buffer(fabric, [ReplayBuffer(4, n_envs=1), ReplayBuffer(4, n_envs=1)], buffer)


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


@dataclass
class ToyState(TrainState):
    model: Parent
    optimizer: torch.optim.Optimizer
    coef: Tensor
    counter: int


def toy_state(fabric, seed):
    torch.manual_seed(seed)
    model = Parent()
    # The children are wrapped by Fabric, as the algorithms do
    model.encoder = core.setup_module(fabric, model.encoder)
    model.head = core.setup_module(fabric, model.head)
    optimizer = fabric.setup_optimizers(torch.optim.Adam(model.parameters(), lr=1e-2))
    return ToyState(model=model, optimizer=optimizer, coef=torch.tensor(0.5), counter=3)


def test_train_state_round_trip_with_wrapped_modules(fabric):
    state = toy_state(fabric, seed=0)
    update(fabric, state.model(torch.randn(5, 3)).square().mean(), state.optimizer)
    state.coef.fill_(0.25)
    state.counter = 7
    saved = copy.deepcopy(state.state_dict())
    # The keys are those of the unwrapped modules, as in the checkpoints of the old training loops
    assert set(saved["model"]) == {"encoder.weight", "encoder.bias", "head.weight", "head.bias"}

    restored = toy_state(fabric, seed=1)
    restored.load_state_dict(saved)
    for name, tensor in restored.model.state_dict().items():
        assert torch.equal(tensor, saved["model"][name])
    assert restored.optimizer.state_dict()["state"][0]["step"] == 1
    assert restored.coef.item() == 0.25
    assert restored.counter == 7

    # The restored optimizer continues exactly like the original one
    x = torch.randn(5, 3)
    update(fabric, state.model(x).square().mean(), state.optimizer)
    update(fabric, restored.model(x).square().mean(), restored.optimizer)
    for a, b in zip(state.model.parameters(), restored.model.parameters()):
        assert torch.equal(a, b)


def test_train_state_warns_about_fields_missing_from_the_checkpoint(fabric):
    state = toy_state(fabric, seed=0)
    saved = state.state_dict()
    del saved["coef"]
    with pytest.warns(UserWarning, match="coef"):
        state.load_state_dict(saved)


def test_load_module_state_dict_rejects_wrong_keys(fabric):
    model = toy_state(fabric, seed=0).model
    with pytest.raises(RuntimeError, match="missing keys"):
        load_module_state_dict(model, {"encoder.weight": torch.zeros(4, 3)})


def test_update_matches_backward_and_step(fabric):
    torch.manual_seed(0)
    model = core.setup_module(fabric, Parent())
    reference = copy.deepcopy(model)
    optimizer = fabric.setup_optimizers(torch.optim.Adam(model.parameters(), lr=1e-2))
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-2)
    for _ in range(3):
        x = torch.randn(5, 3)
        update(fabric, model(x).square().mean(), optimizer, max_grad_norm=0.1)

        reference_optimizer.zero_grad()
        reference(x).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        reference_optimizer.step()
    for a, b in zip(model.parameters(), reference.parameters()):
        assert torch.equal(a, b)


def test_update_computes_only_the_gradients_of_the_optimizer_weights(fabric):
    model = core.setup_module(fabric, Parent())
    # Only the head is optimized: the backward pass must not compute the gradients of the encoder
    optimizer = fabric.setup_optimizers(torch.optim.SGD(model.head.parameters(), lr=1.0))
    encoder_weight = model.encoder.weight.detach().clone()
    head_weight = model.head.weight.detach().clone()
    update(fabric, model(torch.randn(5, 3)).square().mean(), optimizer)
    assert model.encoder.weight.grad is None
    assert torch.equal(model.encoder.weight, encoder_weight)
    assert not torch.equal(model.head.weight, head_weight)


def test_update_clips_the_gradient_norm(fabric):
    model = core.setup_module(fabric, Parent())
    optimizer = fabric.setup_optimizers(torch.optim.SGD(model.parameters(), lr=0.0))
    update(fabric, 1000 * model(torch.randn(5, 3)).square().mean(), optimizer, max_grad_norm=0.5)
    total_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert total_norm.item() == pytest.approx(0.5, rel=1e-5)


def test_env_step_final_obs_reads_the_last_observations_of_the_ended_episodes():
    final_obs = np.array([None, {"state": np.full(2, 7.0)}, {"state": np.full(2, 9.0)}], dtype=object)
    step = EnvStep(
        obs={},
        next_obs={"state": np.zeros((3, 2))},
        rewards=np.zeros(3),
        terminated=np.array([False, True, False]),
        truncated=np.array([False, False, True]),
        info={"final_obs": final_obs},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(step.final_obs([1, 2], ["state"])["state"], np.array([[7.0, 7.0], [9.0, 9.0]]))
