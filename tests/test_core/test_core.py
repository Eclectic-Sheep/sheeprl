import copy
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from lightning import Fabric
from torch import Tensor, nn

# `setup_module` is not imported by name: pytest would run it as the setup of this test module
from sheeprl import core
from sheeprl.core import EnvStep, Episode, GymEnvironment, TrainSchedule, TrainState, load_replay_buffer, update
from sheeprl.core.algorithm import load_module_state_dict
from sheeprl.core.cadence import Cadence
from sheeprl.core.loop import phase_timer
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.data.samplers import TransitionSampler
from sheeprl.data.store import ReplayStore
from sheeprl.utils.env import get_vector_env_cls
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import dotdict


@pytest.fixture()
def fabric():
    return Fabric(accelerator="cpu", devices=1, precision="32-true")


def schedule_cfg(
    total_steps=100,
    dry_run=False,
    learning_starts=10,
    replay_ratio=1.0,
    run_benchmarks=False,
    buffer_checkpoint=True,
    pretrain_steps=0,
    sequence_length=None,
):
    return dotdict(
        {
            "env": {"num_envs": 2},
            "buffer": {"checkpoint": buffer_checkpoint},
            "algo": {
                "total_steps": total_steps,
                # Read only by the off-policy algorithms
                "learning_starts": learning_starts,
                "replay_ratio": replay_ratio,
                "per_rank_pretrain_steps": pretrain_steps,
                # Read only by the off-policy algorithms that sample sequences (Dreamer)
                "per_rank_sequence_length": sequence_length,
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


def test_off_policy_schedule_pretrains_at_the_first_training():
    # The first training did `pretrain_steps * replay_ratio` gradient steps, with `pretrain_steps` capped to the policy
    # steps of an iteration: never more than the replay ratio asks (0 with the defaults of DreamerV2)
    schedule = TrainSchedule(
        schedule_cfg(learning_starts=10, replay_ratio=0.25, pretrain_steps=100),
        world_size=1,
        steps_per_iteration=1,
        off_policy=True,
    )
    assert [schedule.gradient_steps(i) for i in range(1, 10)] == [0, 0, 0, 0, 100, 0, 1, 0, 1]
    # Not in a dry run, which does one gradient step
    dry_run = schedule_cfg(dry_run=True, replay_ratio=0.5, pretrain_steps=100)
    schedule = TrainSchedule(dry_run, world_size=1, steps_per_iteration=1, off_policy=True)
    assert schedule.gradient_steps(1) == 1


def test_off_policy_schedule_resumed_after_the_first_training_does_not_pretrain_again():
    _, checkpoint = checkpoint_of_iteration(8, pretrain_steps=100)
    resumed = TrainSchedule(
        schedule_cfg(pretrain_steps=100), world_size=1, steps_per_iteration=1, checkpoint=checkpoint, off_policy=True
    )
    assert [resumed.gradient_steps(i) for i in range(9, 12)] == [2, 2, 2]
    # Without its buffer, it trains as a new run once it has filled a new one: it pretrains on it
    _, checkpoint = checkpoint_of_iteration(8, pretrain_steps=100, buffer_checkpoint=False)
    resumed = TrainSchedule(
        schedule_cfg(pretrain_steps=100, buffer_checkpoint=False),
        world_size=1,
        steps_per_iteration=1,
        checkpoint=checkpoint,
        off_policy=True,
    )
    assert [resumed.gradient_steps(i) for i in range(9, 16)] == [0, 0, 0, 0, 102, 2, 2]


def test_off_policy_schedule_waits_for_a_sequence_of_every_environment():
    # The training started after `learning_starts` policy steps also when the environments had played fewer steps than
    # the sequences it samples: the sampling crashed. 2 envs: `learning_starts=4` gives 2 iterations of random actions
    with pytest.warns(UserWarning, match="sequences of `algo.per_rank_sequence_length=5`"):
        schedule = TrainSchedule(
            schedule_cfg(learning_starts=4, sequence_length=5), world_size=1, steps_per_iteration=1, off_policy=True
        )
    assert [schedule.warmup(policy_step=2 * (i - 1)) for i in range(1, 6)] == [True, True, False, False, False]
    assert [schedule.gradient_steps(i) for i in range(1, 8)] == [0, 0, 0, 0, 2, 2, 2]
    # When `learning_starts` comes later, nothing changes
    schedule = TrainSchedule(
        schedule_cfg(learning_starts=10, sequence_length=5), world_size=1, steps_per_iteration=1, off_policy=True
    )
    assert [schedule.gradient_steps(i) for i in range(1, 7)] == [0, 0, 0, 0, 2, 2]


def test_off_policy_dry_run_plays_until_its_first_training():
    # A dry run played one iteration: with sequences longer than one step, the sampling crashed
    schedule = TrainSchedule(
        schedule_cfg(dry_run=True, sequence_length=3), world_size=1, steps_per_iteration=1, off_policy=True
    )
    assert list(schedule.iterations()) == [1, 2, 3]
    assert [schedule.gradient_steps(i) for i in range(1, 4)] == [0, 0, 2]


def checkpoint_of_iteration(iteration, **cfg):
    """The schedule of an off-policy run (2 envs, 1 process, `learning_starts=10`, ratio 1) up to `iteration`, and the
    checkpoint saved at its end."""
    schedule = TrainSchedule(schedule_cfg(**cfg), world_size=1, steps_per_iteration=1, off_policy=True)
    steps = [schedule.gradient_steps(i) for i in range(1, iteration + 1)]
    checkpoint = {"iter_num": iteration, "per_rank_gradient_steps": sum(steps), "ratio": schedule.ratio.state_dict()}
    return schedule, checkpoint


def test_off_policy_schedule_resumes_as_the_run_it_resumes():
    # The checkpoint of iteration 8 of the first test: the ratio counted 8 policy steps (iterations 5 to 8)
    schedule, checkpoint = checkpoint_of_iteration(8)
    resumed = TrainSchedule(schedule_cfg(), world_size=1, steps_per_iteration=1, checkpoint=checkpoint, off_policy=True)
    assert resumed.ratio.state_dict() == schedule.ratio.state_dict()
    assert resumed.gradient_step == 8
    assert list(resumed.iterations())[:2] == [9, 10]
    # With its replay buffer, it neither plays random actions nor waits to train: it goes on as the run would have
    assert not resumed.warmup(policy_step=16)
    assert [resumed.gradient_steps(i) for i in range(9, 16)] == [schedule.gradient_steps(i) for i in range(9, 16)]


def test_off_policy_schedule_resumed_during_the_random_actions_plays_the_rest_of_them():
    schedule, checkpoint = checkpoint_of_iteration(3)
    resumed = TrainSchedule(schedule_cfg(), world_size=1, steps_per_iteration=1, checkpoint=checkpoint, off_policy=True)
    assert [resumed.warmup(policy_step=2 * (i - 1)) for i in range(4, 8)] == [True, True, False, False]
    assert [resumed.gradient_steps(i) for i in range(4, 10)] == [schedule.gradient_steps(i) for i in range(4, 10)]


def test_off_policy_schedule_resumed_without_its_buffer_fills_a_new_one_with_its_policy():
    _, checkpoint = checkpoint_of_iteration(8, buffer_checkpoint=False)
    resumed = TrainSchedule(
        schedule_cfg(buffer_checkpoint=False),
        world_size=1,
        steps_per_iteration=1,
        checkpoint=checkpoint,
        off_policy=True,
    )
    # No random actions: the policy fills the new buffer for `learning_starts` steps (5 iterations), then the training
    # starts as in a new run
    assert not any(resumed.warmup(policy_step=2 * (i - 1)) for i in range(9, 15))
    assert [resumed.gradient_steps(i) for i in range(9, 16)] == [0, 0, 0, 0, 2, 2, 2]


def test_the_training_speed_counts_the_gradient_steps_of_all_the_processes(monkeypatch):
    # It counted the iterations that trained, whatever their gradient steps: with 4 gradient steps per iteration it
    # was a quarter of the gradient steps per second
    logged = {}
    fabric = SimpleNamespace(world_size=2, log=lambda name, value, step: logged.__setitem__(name, value))
    cfg = dotdict({"metric": {"log_level": 1, "log_every": 1}})
    cadence = Cadence(fabric, cfg, "unused", aggregator=None)
    # One iteration of 4 gradient steps in every process
    for _ in range(4):
        cadence.accumulate({})
    monkeypatch.setattr(timer, "disabled", False)
    monkeypatch.setattr(timer, "compute", classmethod(lambda cls: {"Time/train_time": 2.0}))
    monkeypatch.setattr(timer, "reset", classmethod(lambda cls: None))
    cadence.log(policy_step=8, iteration=1, schedule=SimpleNamespace(off_policy=False, total_iters=10))
    assert logged["Time/sps_train"] == 4 * 2 / 2.0


def test_the_interaction_speed_of_a_resumed_run_counts_only_its_own_steps(monkeypatch):
    # The first speed after a resume divided the policy steps since the last log of the resumed run, also the ones
    # played before its checkpoint, by the time of the steps played after the resume
    logged = {}
    fabric = SimpleNamespace(world_size=2, log=lambda name, value, step: logged.__setitem__(name, value))
    cfg = dotdict({"metric": {"log_level": 1, "log_every": 5000}, "env": {"action_repeat": 2}})
    # Last log at 0, checkpoint at 3000: the resumed run starts from 3000
    checkpoint = {"last_log": 0, "last_checkpoint": 3000}
    cadence = Cadence(fabric, cfg, "unused", aggregator=None, checkpoint=checkpoint, policy_step=3000)
    monkeypatch.setattr(timer, "disabled", False)
    monkeypatch.setattr(timer, "compute", classmethod(lambda cls: {"Time/env_interaction_time": 4.0}))
    monkeypatch.setattr(timer, "reset", classmethod(lambda cls: None))
    cadence.log(policy_step=5000, iteration=1, schedule=SimpleNamespace(off_policy=False, total_iters=10))
    # 2000 policy steps (of the 2 processes) of 2 environment steps in 4 seconds: the speed of the run, as
    # `Time/sps_train` (it was divided by the number of processes)
    assert logged["Time/sps_env_interaction"] == 2000 * 2 / 4.0


def test_the_phases_are_timed_per_process(monkeypatch):
    # The training was timed with `metric.sync_on_compute`: with it, the training times of all the processes were
    # summed, and `Time/sps_train` divided by their number
    monkeypatch.setattr(timer, "timers", {})
    monkeypatch.setattr(timer, "disabled", False)
    for name in ("Time/train_time", "Time/env_interaction_time"):
        with phase_timer(name):
            pass
        assert timer.timers[name].sync_on_compute is False


def test_load_replay_buffer_takes_the_buffer_of_the_process(fabric):
    store = lambda: ReplayStore(ReplayBuffer(4, n_envs=1), TransitionSampler(seed=0))
    saved = [store()]
    assert load_replay_buffer(fabric, saved, store()).storage is saved[0].storage
    # A single saved buffer is the one of every process
    assert load_replay_buffer(fabric, saved[0], store()).storage is saved[0].storage
    with pytest.raises(RuntimeError, match="2 replay buffer"):
        load_replay_buffer(fabric, [store(), store()], store())


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


def test_update_returns_the_gradient_norm_before_clipping(fabric):
    model = core.setup_module(fabric, Parent())
    optimizer = fabric.setup_optimizers(torch.optim.SGD(model.parameters(), lr=0.0))
    x = torch.randn(5, 3)
    assert update(fabric, model(x).square().mean(), optimizer) is None
    grad_norm = update(fabric, 1000 * model(x).square().mean(), optimizer, max_grad_norm=0.5)
    reference = copy.deepcopy(model)
    reference.zero_grad(set_to_none=True)
    (1000 * reference(x).square().mean()).backward()
    expected = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in reference.parameters()]))
    assert grad_norm.item() == pytest.approx(expected.item(), rel=1e-5)
    # A non-finite norm raises only when asked to
    loss = model(x).square().mean() * float("inf")
    with pytest.raises(RuntimeError, match="non-finite"):
        update(fabric, loss, optimizer, max_grad_norm=0.5)
    assert not torch.isfinite(
        update(fabric, model(x).square().mean() * float("inf"), optimizer, max_grad_norm=0.5, error_if_nonfinite=False)
    )


def test_env_step_stack_final_obs_stacks_the_last_observations_of_the_ended_episodes():
    final_obs = np.array([None, {"state": np.full(2, 7.0)}, {"state": np.full(2, 9.0)}], dtype=object)
    step = EnvStep(
        obs={},
        next_obs={"state": np.zeros((3, 2))},
        rewards=np.zeros(3),
        terminated=np.array([False, True, False]),
        truncated=np.array([False, False, True]),
        final_obs=final_obs,
        restarted=np.zeros(3, dtype=bool),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(step.stack_final_obs([1, 2], ["state"])["state"], np.array([[7.0, 7.0], [9.0, 9.0]]))


def test_every_process_returns_from_the_training_once_all_of_them_trained(monkeypatch, tmp_path):
    # The process of rank 0 returned while the other ones still trained: a test removed the files that another process
    # of DreamerV3.5 was writing (its buffer), and Lightning killed the worker of the tests when that process failed
    import sys

    from lightning.fabric import Fabric

    from sheeprl.algos.ppo import ppo
    from sheeprl.cli import run

    events = []
    barrier, train_step = Fabric.barrier, ppo.PPO.train_step

    def recording_barrier(self, *args, **kwargs):
        events.append("barrier")
        return barrier(self, *args, **kwargs)

    def recording_train_step(self, *args, **kwargs):
        events.append("train")
        return train_step(self, *args, **kwargs)

    monkeypatch.setattr(Fabric, "barrier", recording_barrier)
    monkeypatch.setattr(ppo.PPO, "train_step", recording_train_step)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LT_DEVICES", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sheeprl.py",
            "hydra/job_logging=disabled",
            "hydra/hydra_logging=disabled",
            "exp=ppo",
            "dry_run=True",
            "env.num_envs=2",
            "env.sync_env=True",
            "env.capture_video=False",
            "fabric.accelerator=cpu",
            "metric.log_level=0",
            "checkpoint.save_last=False",
            "algo.run_test=False",
            "algo.rollout_steps=4",
            "algo.per_rank_batch_size=4",
        ],
    )
    run()
    assert "train" in events and events[-1] == "barrier"


def test_a_rollout_is_a_store_whose_epoch_sampler_draws_the_minibatches(tmp_path):
    # The steps of the rollout in its ReplayBuffer, read whole on the device; the minibatches of every epoch of its
    # update cover all the elements of what the training computes from them
    from sheeprl.core import rollout_store
    from sheeprl.data.samplers import EpochSampler

    cfg = dotdict(
        {
            "seed": 0,
            "env": {"num_envs": 2},
            "buffer": {"memmap": False, "from_numpy": False, "share_data": False},
            "algo": {"per_rank_batch_size": 3, "cnn_keys": {"encoder": []}, "mlp_keys": {"encoder": ["state"]}},
        }
    )
    fabric = Fabric(accelerator="cpu", devices=1)
    rollout = rollout_store(fabric, cfg, str(tmp_path), size=4)
    assert isinstance(rollout.storage, ReplayBuffer) and isinstance(rollout.sampler, EpochSampler)
    for t in range(4):
        rollout.add({"state": np.full((1, 2, 1), t, np.float32)})
    data = rollout.read()
    assert data["state"].shape == (4, 2, 1)
    flat = {"state": data["state"].flatten(0, 1)}
    torch.manual_seed(0)
    minibatches = list(rollout.minibatches(flat, epochs=2))
    assert [len(b["state"]) for b in minibatches] == [3, 3, 2] * 2
    for epoch in (minibatches[:3], minibatches[3:]):
        assert sorted(torch.cat([b["state"] for b in epoch]).flatten().tolist()) == sorted(
            flat["state"].flatten().tolist()
        )


class TwoStepEnv(gym.Env):
    """Episodes of two steps, each with reward 1; the observation is the step of the episode."""

    observation_space = gym.spaces.Dict({"state": gym.spaces.Box(0, 2, (1,), np.float32)})
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return {"state": np.zeros(1, np.float32)}, {}

    def step(self, action):
        self.t += 1
        return {"state": np.full(1, self.t, np.float32)}, 1.0, self.t == 2, False, {}


def test_gym_environment_reports_the_ended_episodes_in_the_env_step_fields():
    envs = get_vector_env_cls(True)([lambda: gym.wrappers.RecordEpisodeStatistics(TwoStepEnv())] * 2)
    env = GymEnvironment(envs, seed=0)
    env.reset()
    first = env.step(np.zeros(2, dtype=np.int64))
    assert list(first.final_obs) == [None, None] and first.episodes == () and not first.restarted.any()
    second = env.step(np.zeros(2, dtype=np.int64))
    # The episodes end and the environments start the next ones in the same step
    assert np.array_equal(second.next_obs["state"], np.zeros((2, 1))) and np.array_equal(
        env.obs["state"], np.zeros((2, 1))
    )
    assert np.array_equal(second.stack_final_obs([0, 1], ["state"])["state"], np.full((2, 1), 2.0))
    assert second.episodes == (Episode(0, 2.0, 2), Episode(1, 2.0, 2))
    assert not second.restarted.any()
    env.close()
