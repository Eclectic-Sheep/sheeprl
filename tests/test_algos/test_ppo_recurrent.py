"""The agent of PPO-recurrent: orthogonal initialization, continuous actions, the player."""

import tempfile
from math import sqrt
from types import SimpleNamespace
from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn

from sheeprl.algos.ppo_recurrent import ppo_recurrent
from sheeprl.algos.ppo_recurrent.agent import RecurrentPPOAgent, RecurrentPPOPolicy
from sheeprl.utils import compile as compile_utils
from sheeprl.utils.utils import dotdict
from tests.test_algos.compiled import assert_same_step, no_host_reads, same_random_numbers

HIDDEN_SIZE = 16


def build_agent(ortho_init: bool = False, is_continuous: bool = False) -> RecurrentPPOAgent:
    torch.manual_seed(0)
    networks = {"dense_units": 32, "mlp_layers": 2, "dense_act": "torch.nn.Tanh", "layer_norm": False}
    no_mlp = {"apply": False, "bias": True, "activation": "torch.nn.Tanh", "layer_norm": False, "dense_units": 8}
    return RecurrentPPOAgent(
        actions_dim=[2] if is_continuous else [3],
        obs_space=gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (8,), np.float32)}),
        encoder_cfg=dotdict({**networks, "cnn_features_dim": 64, "mlp_features_dim": 32, "ortho_init": ortho_init}),
        rnn_cfg=dotdict({"lstm": {"hidden_size": HIDDEN_SIZE}, "pre_rnn_mlp": no_mlp, "post_rnn_mlp": no_mlp}),
        actor_cfg=dotdict({**networks, "ortho_init": ortho_init}),
        critic_cfg=dotdict({**networks, "ortho_init": ortho_init}),
        cnn_keys=[],
        mlp_keys=["state"],
        is_continuous=is_continuous,
        distribution_cfg=dotdict({"type": "auto"}),
    )


def player_of(agent: RecurrentPPOAgent) -> RecurrentPPOPolicy:
    return RecurrentPPOPolicy(
        agent.feature_extractor, agent.rnn, agent.actor, agent.critic, HIDDEN_SIZE, agent.actions_dim
    )


def linear_layers(module: nn.Module) -> List[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def assert_orthogonal(layer: nn.Linear, gain: float) -> None:
    singular_values = torch.linalg.svdvals(layer.weight.detach())
    torch.testing.assert_close(singular_values, torch.full_like(singular_values, gain))
    assert torch.all(layer.bias == 0)


def test_ortho_init():
    # The flags inherited from PPO's configuration were not read
    agent = build_agent(ortho_init=True)
    for layer in linear_layers(agent.feature_extractor):
        assert_orthogonal(layer, 1.0)
    *actor_hidden, actor_head = linear_layers(agent.actor)
    *critic_hidden, critic_output = linear_layers(agent.critic)
    for layer in actor_hidden + critic_hidden:
        assert_orthogonal(layer, sqrt(2))
    assert_orthogonal(actor_head, 0.01)
    assert_orthogonal(critic_output, 1.0)
    default = build_agent(ortho_init=False)
    for layer in [layer for layer in linear_layers(default) if min(layer.weight.shape) > 1]:
        singular_values = torch.linalg.svdvals(layer.weight.detach())
        assert not torch.allclose(singular_values, singular_values[0].expand_as(singular_values))


def test_continuous_actions():
    # The log-probabilities of the continuous actions were computed from `None` (the player) or from the tuple of
    # actions (the agent): both crashed
    agent = build_agent(is_continuous=True)
    player = player_of(agent)
    obs = {"state": torch.randn(1, 4, 8)}
    prev_actions = torch.zeros(1, 4, 2)
    states = (torch.zeros(1, 4, HIDDEN_SIZE), torch.zeros(1, 4, HIDDEN_SIZE))
    with torch.no_grad():
        actions, logprobs, _, _ = player(obs, prev_actions=prev_actions, prev_states=states)
        _, agent_logprobs, entropies, _, _ = agent(obs, prev_actions=prev_actions, prev_states=states, actions=actions)
    assert actions[0].shape == (1, 4, 2) and logprobs.shape == (1, 4, 1) and entropies.shape == (1, 4, 1)
    torch.testing.assert_close(agent_logprobs, logprobs)


def config(overrides: List[str]) -> dotdict:
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        return dotdict(OmegaConf.to_container(compose(config_name="config", overrides=overrides), resolve=True))


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"))],
)
def test_the_player_follows_the_updates_of_the_agent(accelerator):
    # The player was a copy of the agent with the weights tied: on CUDA its LSTM moved them into a new buffer at its
    # first forward (`flatten_parameters`), and the player played with the initial weights for the whole training
    from lightning import Fabric

    from sheeprl.algos.ppo_recurrent.agent import build_agent as build_agents

    cfg = config(
        ["exp=ppo_recurrent", "env.num_envs=2", "algo.mlp_keys.encoder=[state]", "algo.rnn.lstm.hidden_size=8"]
    )
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (8,), np.float32)})
    fabric = Fabric(accelerator=accelerator, devices=1)
    agent, player = build_agents(fabric, [3], False, cfg, obs_space)
    device = fabric.device
    with torch.no_grad():
        player(
            {"state": torch.randn(1, 2, 8, device=device)},
            prev_actions=torch.zeros(1, 2, 3, device=device),
            prev_states=(torch.zeros(1, 2, 8, device=device), torch.zeros(1, 2, 8, device=device)),
        )
        for p in agent.parameters():
            p.add_(1.0)
    for module in ("feature_extractor", "rnn", "actor", "critic"):
        agent_params = list(getattr(agent, module).parameters())
        player_params = list(getattr(player, module).parameters())
        assert len(agent_params) == len(player_params) > 0
        for agent_p, player_p in zip(agent_params, player_params):
            assert torch.equal(agent_p, player_p), module


def small_ppo_recurrent(overrides=(), accelerator="cpu"):
    """PPO-recurrent with small models, built as by the training, and a minibatch of 3 sequences of 4 steps, the last
    ones padded."""
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=ppo_recurrent",
                "algo.mlp_keys.encoder=[state]",
                "algo.cnn_keys.encoder=[]",
                "algo.rnn.lstm.hidden_size=8",
                "algo.dense_units=8",
                "algo.normalize_advantages=True",
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    fabric = Fabric(accelerator=accelerator, devices=1)
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)})
    torch.manual_seed(0)
    algorithm = ppo_recurrent.PPORecurrent(fabric, cfg)
    state, _ = algorithm.build(obs_space, gym.spaces.Discrete(3), SimpleNamespace(total_iters=4), tempfile.mkdtemp())
    generator = torch.Generator().manual_seed(1)
    T, B = 4, 3
    actions = torch.nn.functional.one_hot(torch.randint(0, 3, (T, B), generator=generator), 3).float()
    batch = {
        "state": torch.randn(T, B, 5, generator=generator),
        "actions": actions,
        "prev_actions": torch.cat((torch.zeros(1, B, 3), actions[:-1])),
        "prev_hx": torch.randn(T, B, 8, generator=generator),
        "prev_cx": torch.randn(T, B, 8, generator=generator),
        # The sequences of 4, 2 and 3 steps
        "mask": torch.arange(T)[:, None] < torch.tensor([4, 2, 3]),
    }
    for k in ("logprobs", "values", "returns", "advantages"):
        batch[k] = torch.randn(T, B, 1, generator=generator)
    return cfg, algorithm, state, {k: v.to(fabric.device) for k, v in batch.items()}


def test_the_ppo_recurrent_loss_compiles_into_a_single_graph_without_synchronizations(monkeypatch):
    # The steps of the sequences were selected by indexing, and the LSTM packed the sequences with their lengths read
    # on the host: shapes that depend on the data, which broke the graph and synchronized with the device
    cfg, algorithm, state, batch = small_ppo_recurrent()
    no_host_reads(monkeypatch)
    torch.compile(ppo_recurrent.ppo_recurrent_loss, backend="eager", fullgraph=True)(
        state.agent,
        {"state": batch["state"]},
        batch["prev_actions"],
        (batch["prev_hx"][:1], batch["prev_cx"][:1]),
        batch["actions"],
        batch["mask"].unsqueeze(-1),
        batch["logprobs"],
        batch["values"],
        batch["returns"],
        batch["advantages"],
        algorithm.clip_coef,
        algorithm.ent_coef,
        actions_dim=(3,),
        normalize_advantages=True,
        vf_coef=cfg.algo.vf_coef,
        clip_vloss=True,
        entropy_reduction=cfg.algo.loss_reduction,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled losses are compared on the GPU")
def test_the_compiled_ppo_recurrent_loss_is_the_eager_loss(monkeypatch):
    # The same weights, the same minibatch and the same random numbers: the same losses and gradients of a gradient
    # step, with and without `torch.compile` (and its CUDA graphs)
    same_random_numbers(monkeypatch)
    monkeypatch.setattr(compile_utils, "_COMPILED", {})
    results = []
    for enabled in (False, True):
        _, algorithm, state, batch = small_ppo_recurrent([f"algo.compile.enabled={enabled}"], accelerator="cuda")
        torch.manual_seed(1)
        metrics = algorithm.train_step(state, batch, 0)
        losses = [(name, value.detach().clone().reshape(-1)) for name, value in metrics.items()]
        grads = [[None if p.grad is None else p.grad.detach().clone() for p in state.agent.parameters()]]
        results.append((losses, grads))
    assert_same_step(*results)


def test_a_minibatch_padded_with_sequences_out_of_the_mask_has_the_same_loss():
    # Compiled, the minibatches are padded with copies of their first sequence out of the mask (`batches`): the same
    # losses and gradients as the minibatch alone
    results = []
    for padded in (False, True):
        _, algorithm, state, batch = small_ppo_recurrent()
        if padded:
            size = batch["mask"].shape[1]
            idxes = list(range(size)) + [0] * (ppo_recurrent.SEQUENCES_MULTIPLE - size)
            batch = {k: v[:, idxes] for k, v in batch.items()}
            batch["mask"][:, size:] = False
        metrics = algorithm.train_step(state, batch, 0)
        losses = [(name, value.detach().clone().reshape(-1)) for name, value in metrics.items()]
        grads = [[None if p.grad is None else p.grad.detach().clone() for p in state.agent.parameters()]]
        results.append((losses, grads))
    assert_same_step(*results)


def test_the_compiled_minibatches_of_a_rollout_have_one_size(monkeypatch):
    # The number of sequences changes with the episodes that end in the rollout, and so did the size of the
    # minibatches, the last one of every epoch smaller: compiled with CUDA graphs, every new size was a new recording.
    # Compiled, all the minibatches of a rollout have the size rounded up to a multiple of `SEQUENCES_MULTIPLE`, padded
    # with sequences out of the mask
    import os
    import shutil
    import sys
    from unittest import mock

    from sheeprl.cli import run

    monkeypatch.setattr(ppo_recurrent, "compiled", lambda fn, fabric, cfg, **kwargs: fn)
    rollouts = []
    batches = ppo_recurrent.PPORecurrent.batches

    def recording_batches(self, *args, **kwargs):
        rollouts.append([])
        for batch in batches(self, *args, **kwargs):
            rollouts[-1].append(batch["mask"].clone())
            yield batch

    monkeypatch.setattr(ppo_recurrent.PPORecurrent, "batches", recording_batches)
    root_dir = "pytest_ppo_recurrent_padded_minibatches"
    argv = [
        "sheeprl.py",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
        "exp=ppo_recurrent",
        "env.num_envs=4",
        "env.sync_env=True",
        "env.capture_video=False",
        "fabric.accelerator=cpu",
        "metric.log_level=0",
        "checkpoint.save_last=False",
        "algo.run_test=False",
        "algo.compile.enabled=True",
        # The losses are not compiled (`compiled` above), nor the player: on the CPU Inductor needs a C++ compiler
        "algo.compile.policy=False",
        "algo.rollout_steps=64",
        "algo.per_rank_sequence_length=8",
        "algo.per_rank_num_batches=3",
        "algo.update_epochs=2",
        "algo.total_steps=768",
        f"root_dir={root_dir}",
    ]
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", argv):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    assert len(rollouts) == 3
    sizes = []
    for masks in rollouts:
        (shape,) = {mask.shape for mask in masks}
        assert shape[1] % ppo_recurrent.SEQUENCES_MULTIPLE == 0
        valid = [int(mask.any(0).sum()) for mask in masks]
        for mask, n in zip(masks, valid):
            # The sequences of the minibatch, then the padding
            assert mask[:, :n].any(0).all() and not mask[:, n:].any()
        sizes.append(sorted(set(valid)))
    # The minibatches of a rollout had different sizes (the last one of every epoch smaller), padded to one
    assert all(len(s) > 1 for s in sizes)


@pytest.mark.parametrize("reset_on_done", [False, True])
def test_the_refreshed_recurrent_states_are_the_ones_of_the_player_with_its_weights(reset_on_done):
    # `algo.refresh_recurrent_states` unrolls the recurrent states of the rollout again with the current weights: with
    # the weights that played it, the states the player stored, reset after the end of the episodes as it does
    from sheeprl.algos.ppo_recurrent.agent import build_agent as build_agents

    cfg = config(
        ["exp=ppo_recurrent", "env.num_envs=3", "algo.mlp_keys.encoder=[state]", "algo.rnn.lstm.hidden_size=8"]
    )
    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (5,), np.float32)})
    torch.manual_seed(0)
    agent, player = build_agents(Fabric(accelerator="cpu", devices=1), [3], False, cfg, obs_space)
    T, N = 12, 3
    generator = torch.Generator().manual_seed(1)
    obs = torch.randn(T, N, 5, generator=generator)
    dones = (torch.rand(T, N, 1, generator=generator) < 0.2).float()
    states = (torch.randn(1, N, 8, generator=generator), torch.randn(1, N, 8, generator=generator))
    prev_actions = torch.zeros(1, N, 3)
    stored = {"prev_hx": [], "prev_cx": [], "prev_actions": []}
    with torch.no_grad():
        for t in range(T):
            stored["prev_hx"].append(states[0])
            stored["prev_cx"].append(states[1])
            stored["prev_actions"].append(prev_actions)
            actions, _, _, states = player({"state": obs[t : t + 1]}, prev_actions=prev_actions, prev_states=states)
            actions = torch.cat(actions, dim=-1)
            # As the rollout player
            prev_actions = (1 - dones[t : t + 1]) * actions
            if reset_on_done:
                states = tuple((1 - dones[t : t + 1]) * s for s in states)
    stored = {k: torch.cat(v) for k, v in stored.items()}
    prev_hx, prev_cx = ppo_recurrent.recurrent_states(
        agent,
        {"state": obs},
        stored["prev_actions"],
        dones,
        (stored["prev_hx"][:1], stored["prev_cx"][:1]),
        reset_on_done,
    )
    torch.testing.assert_close(prev_hx, stored["prev_hx"])
    torch.testing.assert_close(prev_cx, stored["prev_cx"])


def test_a_ppo_recurrent_training_refreshes_the_recurrent_states(monkeypatch):
    # With `algo.refresh_recurrent_states`, every epoch after the first one unrolls the states of the rollout again;
    # the rollout stores the previous actions in double precision, which the LSTM refused
    import os
    import shutil
    import sys
    from unittest import mock

    from sheeprl.cli import run

    refreshes = []
    recurrent_states = ppo_recurrent.recurrent_states

    def recording_recurrent_states(*args, **kwargs):
        refreshes.append(recurrent_states(*args, **kwargs))
        return refreshes[-1]

    monkeypatch.setattr(ppo_recurrent, "recurrent_states", recording_recurrent_states)
    root_dir = "pytest_ppo_recurrent_refreshed_states"
    argv = [
        "sheeprl.py",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
        "exp=ppo_recurrent",
        "dry_run=True",
        "env.num_envs=2",
        "env.sync_env=True",
        "env.capture_video=False",
        "fabric.accelerator=cpu",
        "metric.log_level=0",
        "checkpoint.save_last=False",
        "algo.run_test=False",
        "algo.refresh_recurrent_states=True",
        "algo.rollout_steps=8",
        "algo.per_rank_sequence_length=4",
        "algo.update_epochs=3",
        f"root_dir={root_dir}",
    ]
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", argv):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)
    # The epochs after the first one, from the states of the 8 steps of the 2 environments
    assert len(refreshes) == 2
    assert all(tuple(s.shape) == (8, 2, 64) for states in refreshes for s in states)
