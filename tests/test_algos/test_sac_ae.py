"""SAC-AE: the actor uses the convolutions of the critic (also when loaded from an older checkpoint), the decoder and
the loss of the temperature are the ones of the official implementation (https://github.com/denisyarats/pytorch_sac_ae)
and the images have the configured size."""

import math
import os
import shutil
import sys
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from lightning import Fabric
from omegaconf import OmegaConf
from torch import nn

from sheeprl import ROOT_DIR
from sheeprl.algos.sac_ae import agent as sac_ae_agent
from sheeprl.algos.sac_ae.agent import CNNDecoder, CNNEncoder, build_agent, tie_actor_optimizer
from sheeprl.algos.sac_ae.loss import entropy_loss
from sheeprl.utils.utils import dotdict

ACTIONS = gym.spaces.Box(-1, 1, (2,), np.float32)


def small_sac_ae(overrides=(), screen_size=64):
    with initialize_config_module(config_module="sheeprl.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "exp=sac_ae",
                "env=dummy",
                f"env.screen_size={screen_size}",
                "algo.cnn_keys.encoder=[rgb]",
                "algo.mlp_keys.encoder=[]",
                "algo.mlp_keys.decoder=[]",
                "algo.hidden_size=8",
                "algo.encoder.features_dim=8",
                *overrides,
            ],
        )
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True))
    obs_space = gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, (9, screen_size, screen_size), np.uint8)})
    return cfg, obs_space


def test_the_actor_uses_the_convolutions_of_the_critic():
    # The actor had its own convolutions, never trained (it detaches them): assigning the ones of the critic to the
    # read-only property `model` registered them as an unused module
    cfg, obs_space = small_sac_ae()
    agent, *_ = build_agent(Fabric(accelerator="cpu", devices=1), cfg, obs_space, ACTIONS)
    actor_encoder = agent.actor.module.encoder.cnn_encoder
    critic_encoder = agent.critic.module.encoder.cnn_encoder
    assert actor_encoder.model is critic_encoder.model
    assert "model" not in actor_encoder._modules
    # Its own fully-connected layer, as in the official implementation
    assert actor_encoder.fc is not critic_encoder.fc
    obs = {"rgb": torch.rand(3, 9, 64, 64)}
    before = agent.actor.module.encoder(obs).detach().clone()
    with torch.no_grad():
        critic_encoder.model[0].weight.add_(1.0)
    assert not torch.allclose(agent.actor.module.encoder(obs), before)


def test_an_agent_saved_with_the_untied_convolutions_gets_the_ones_of_the_critic():
    # The state of an agent saved before: the actor with its own convolutions, and the ones of the critic as `model`
    cfg, obs_space = small_sac_ae()
    fabric = Fabric(accelerator="cpu", devices=1)
    agent, *_ = build_agent(fabric, cfg, obs_space, ACTIONS)
    state = {}
    for k, v in agent.state_dict().items():
        if k.startswith("_actor.") and ".cnn_encoder._model." in k:
            # The convolutions of the actor, then the ones of the critic
            state[k] = torch.randn_like(v)
            state[k.replace("._model.", ".model.")] = agent.state_dict()["_critic." + k[len("_actor.") :]]
        else:
            state[k] = v
    old_actor_params = [k for k in state if k.startswith("_actor.") and not k.endswith(("action_scale", "action_bias"))]
    loaded, *_ = build_agent(fabric, cfg, obs_space, ACTIONS, agent_state=state)
    loaded_actor = loaded.actor.module.encoder.cnn_encoder
    loaded_critic = loaded.critic.module.encoder.cnn_encoder
    assert loaded_actor.model is loaded_critic.model
    for k, v in loaded_critic.model.state_dict().items():
        torch.testing.assert_close(v, state[f"_critic.encoder.cnn_encoder._model.{k}"])

    # The state of its optimizer: one entry per weight of the old actor, with the state of the trained ones (the
    # convolutions of the actor are detached: they have none)
    trained = [i for i, k in enumerate(old_actor_params) if ".cnn_encoder." not in k or ".fc." in k]
    optimizer_state = {
        "state": {i: {"step": torch.tensor(float(i))} for i in trained},
        "param_groups": [{"lr": 1e-3, "params": list(range(len(old_actor_params)))}],
    }
    optimizer = torch.optim.Adam(loaded.actor.parameters(), lr=1e-3)
    optimizer.load_state_dict(tie_actor_optimizer(optimizer_state, state))
    new_names = [n for n, _ in loaded.actor.module.named_parameters()]
    for i, p in enumerate(loaded.actor.parameters()):
        if i in optimizer.state_dict()["state"]:
            assert optimizer.state[p]["step"].item() == old_actor_params.index("_actor." + new_names[i])
    assert len(optimizer.state) == len(trained)


def test_the_decoder_is_initialized_as_the_official_implementation():
    # It had the default initialization of PyTorch: the official one is orthogonal for the linear layers and
    # delta-orthogonal (orthogonal at the center of the kernels, zero elsewhere) for the transposed convolutions
    torch.manual_seed(0)
    cfg, obs_space = small_sac_ae(["algo.encoder.features_dim=50"])
    _, _, decoder, _ = build_agent(Fabric(accelerator="cpu", devices=1), cfg, obs_space, ACTIONS)
    cnn_decoder = decoder.module.cnn_decoder
    gain = nn.init.calculate_gain("relu")
    for deconv in [cnn_decoder.model[i] for i in (0, 2, 4)] + [cnn_decoder.to_obs]:
        weight = deconv.weight.detach()
        center = weight[:, :, 1, 1].clone()
        weight[:, :, 1, 1] = 0
        assert torch.all(weight == 0) and torch.all(deconv.bias == 0)
        rows, columns = center.shape
        product = center @ center.T if rows <= columns else center.T @ center
        torch.testing.assert_close(product, gain**2 * torch.eye(min(rows, columns)), atol=1e-4, rtol=0)
    fc = cnn_decoder.fc.model[0]
    torch.testing.assert_close(fc.weight.T @ fc.weight, torch.eye(50), atol=1e-4, rtol=0)
    assert torch.all(fc.bias == 0)


def test_the_loss_of_the_temperature_is_the_one_of_the_official_implementation():
    # SAC-AE minimizes the temperature, not its logarithm, times the gap of the entropy: the gradient of its logarithm
    # is scaled by the temperature
    log_alpha = torch.tensor([math.log(0.1)], requires_grad=True)
    logprobs = torch.tensor([[-1.0], [0.5], [2.0]])
    loss = entropy_loss(log_alpha, logprobs, torch.tensor(-2.0))
    expected = 0.1 * (-logprobs + 2.0).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(log_alpha.grad, expected.reshape(1))


@pytest.mark.parametrize("screen_size", [64, 84])
def test_the_decoder_reconstructs_images_of_the_size_of_the_observations(screen_size):
    encoder = CNNEncoder(9, 8, ["rgb"], screen_size=screen_size)
    decoder = CNNDecoder(encoder.conv_output_shape, 8, ["rgb"], [9], screen_size=screen_size)
    reconstructed = decoder(encoder({"rgb": torch.rand(2, 9, screen_size, screen_size)}))["rgb"]
    assert reconstructed.shape == (2, 9, screen_size, screen_size)


def test_the_decoder_rejects_the_odd_sizes():
    encoder = CNNEncoder(9, 8, ["rgb"], screen_size=63)
    with pytest.raises(ValueError, match="must be even"):
        CNNDecoder(encoder.conv_output_shape, 8, ["rgb"], [9], screen_size=63)


SAC_AE_ARGS = [
    "hydra/job_logging=disabled",
    "hydra/hydra_logging=disabled",
    "exp=sac_ae",
    "env=dummy",
    "env.id=continuous_dummy",
    "env.num_envs=1",
    "env.sync_env=True",
    "env.capture_video=False",
    "fabric.accelerator=cpu",
    "fabric.devices=1",
    "metric.log_level=0",
    "checkpoint.save_last=False",
    "algo.run_test=False",
    "algo.cnn_keys.encoder=[rgb]",
    "algo.mlp_keys.encoder=[]",
    "algo.mlp_keys.decoder=[]",
    "algo.hidden_size=8",
    "buffer.memmap=False",
]


def run_sac_ae(args, root_dir):
    from sheeprl.cli import run

    argv = [os.path.join(ROOT_DIR, "__main__.py"), *SAC_AE_ARGS, *args, f"root_dir={root_dir}"]
    try:
        with mock.patch.dict(os.environ, {"LT_DEVICES": "1"}), mock.patch.object(sys, "argv", argv):
            run()
    finally:
        shutil.rmtree(os.path.join("logs", "runs", root_dir), ignore_errors=True)


def test_the_images_have_the_configured_size():
    # The run used images of 64x64 pixels whatever `env.screen_size`: the official implementation uses 84x84
    sizes = []
    build = sac_ae_agent.build_agent

    def recording_build_agent(fabric, cfg, obs_space, *args, **kwargs):
        sizes.append(obs_space["rgb"].shape[-1])
        return build(fabric, cfg, obs_space, *args, **kwargs)

    with mock.patch("sheeprl.algos.sac_ae.sac_ae.build_agent", recording_build_agent):
        run_sac_ae(["dry_run=True", "algo.per_rank_batch_size=1", "buffer.size=2"], "pytest_sac_ae_screen_size")
    assert sizes == [84]


def test_the_batches_of_the_pretraining_are_sampled_16_at_a_time():
    # The batches of all the gradient steps of an iteration were sampled at once: the images of the 1000 steps of the
    # pretraining took 65 GB on the device
    from sheeprl.data.buffers import ReplayBuffer

    sizes = []
    sample_tensors = ReplayBuffer.sample_tensors

    def recording_sample_tensors(self, batch_size, *args, **kwargs):
        sizes.append(batch_size)
        return sample_tensors(self, batch_size, *args, **kwargs)

    with mock.patch.object(ReplayBuffer, "sample_tensors", recording_sample_tensors):
        run_sac_ae(
            [
                "env.screen_size=64",
                "algo.per_rank_batch_size=2",
                "algo.learning_starts=2",
                "algo.per_rank_pretrain_steps=20",
                "algo.replay_ratio=1",
                "algo.total_steps=3",
                "buffer.size=8",
            ],
            "pytest_sac_ae_pretraining",
        )
    # 20 steps of pretraining and 1 of the ratio at the first training, 1 at the next one
    assert sizes == [16 * 2, 5 * 2, 1 * 2]
