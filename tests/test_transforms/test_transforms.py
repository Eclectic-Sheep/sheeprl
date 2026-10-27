import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from lightning import Fabric

from sheeprl.transforms import (
    Cast,
    Clip,
    Dequantize,
    ExpandDims,
    Flatten,
    FrameStack,
    Image,
    OneHot,
    Pipeline,
    Scale,
    ScaleByReturnStd,
    Sign,
    Standardize,
    Symlog,
    Tanh,
    Transform,
    check_transform,
)
from tests.test_transforms import oracles

U8 = np.uint8

CATALOGUE = {
    "image_rgb_resized": (lambda: Image(16), gym.spaces.Box(0, 255, (3, 32, 32), U8)),
    "image_channel_last_gray": (lambda: Image(16, grayscale=True), gym.spaces.Box(0, 255, (48, 48, 3), U8)),
    "image_2d_float": (lambda: Image(8), gym.spaces.Box(0, 1, (20, 20), np.float32)),
    "frame_stack": (lambda: FrameStack(3, dilation=2), gym.spaces.Box(0, 255, (3, 8, 8), U8)),
    "frame_stack_refill": (lambda: FrameStack(2, refill_on=["round_done"]), gym.spaces.Box(0, 255, (1, 8, 8), U8)),
    "flatten_frame": (lambda: Flatten(), gym.spaces.Box(0, 1, (5, 5, 5), np.float32)),
    "flatten_batch": (lambda: Flatten(where="batch"), gym.spaces.Box(0, 1, (5, 5), np.float32)),
    "expand_dims": (lambda: ExpandDims(0), gym.spaces.Box(0, 1, (5, 5), np.float32)),
    "cast": (lambda: Cast("float32"), gym.spaces.Box(0, 255, (4,), U8)),
    "one_hot": (lambda: OneHot(), gym.spaces.Discrete(5, start=2)),
    "standardize": (lambda: Standardize(clip=5.0), gym.spaces.Box(-5, 5, (3,), np.float32)),
    "scale_batch": (lambda: Scale(), gym.spaces.Box(0, 255, (3, 4, 4), U8)),
    "dequantize": (lambda: Dequantize(bits=5), gym.spaces.Box(0, 255, (3, 4, 4), U8)),
    "scale_frame_no_shift": (lambda: Scale(shift=0, where="frame"), gym.spaces.Box(0, 255, (3, 4, 4), U8)),
    "scale_envs": (lambda: Scale(divisor=10, where="envs"), gym.spaces.Box(-10, 10, (2,), np.float32)),
    "symlog": (lambda: Symlog(), gym.spaces.Box(-100, 100, (4,), np.float32)),
    "scale_by_return_std": (lambda: ScaleByReturnStd(gamma=0.9, clip=5.0), gym.spaces.Box(-10, 10, (), np.float32)),
    "tanh": (lambda: Tanh(), gym.spaces.Box(-10, 10, (), np.float64)),
    "sign": (lambda: Sign(), gym.spaces.Box(-10, 10, (), np.float32)),
    "clip_batch": (lambda: Clip(-1.0, 2.0, where="batch"), gym.spaces.Box(-10, 10, (3,), np.float32)),
}


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_every_transform_of_the_catalogue_keeps_the_promises_of_its_space(name):
    factory, space = CATALOGUE[name]
    transform = factory()
    check_transform(transform, space, compile=transform.where == "batch")


IMAGE_SHAPES = [(20, 24), (1, 20, 24), (3, 20, 24), (20, 24, 3), (5, 5, 5), (3, 16, 16)]


@pytest.mark.parametrize("shape", IMAGE_SHAPES)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("grayscale", [False, True])
@pytest.mark.parametrize("size", [16, 8])
def test_image_is_the_transform_of_make_env_bit_for_bit(shape, dtype, grayscale, size):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, shape).astype(U8) if dtype == np.uint8 else rng.random(shape, dtype=np.float32)
    try:
        expected = oracles.transform_obs({"k": x.copy()}, ["k"], size, grayscale)["k"]
    except cv2_error():
        # What the transform of `make_env` couldn't convert, the space refuses
        with pytest.raises(ValueError):
            Image(size, grayscale).space(gym.spaces.Box(0, 255 if dtype == np.uint8 else 1, shape, dtype))
        return
    transform = Image(size, grayscale)
    space = transform.space(gym.spaces.Box(0, 255 if dtype == np.uint8 else 1, shape, dtype))
    out = transform.frame(x.copy())
    assert out.dtype == expected.dtype and out.shape == expected.shape
    assert np.array_equal(out, expected)
    # The space is the one of the data, also where `make_env` declared `(3, size, size)` uint8 whatever the data
    assert space.shape == out.shape and space.dtype == out.dtype


def cv2_error():
    import cv2

    return cv2.error


def test_image_rejects_what_is_not_an_image():
    with pytest.raises(ValueError, match="2D or 3D"):
        Image(8).space(gym.spaces.Box(0, 1, (4,), np.float32))
    with pytest.raises(TypeError, match="Box"):
        Image(8).space(gym.spaces.Discrete(3))


class FramesEnv(gym.Env):
    """Random frames, with the round, stage and game ends of DIAMBRA in `info`."""

    observation_space = gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, (3, 4, 4), U8)})
    action_space = gym.spaces.Discrete(2)

    def __init__(self, ends):
        self.ends = ends
        self.rng = np.random.default_rng(0)
        self.t = 0

    def _frame(self):
        return {"rgb": self.rng.integers(0, 256, (3, 4, 4)).astype(U8)}

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._frame(), {}

    def step(self, action):
        self.t += 1
        info = {"env_domain": "DIAMBRA", "round_done": self.t in self.ends, "stage_done": False, "game_done": False}
        return self._frame(), 0.0, self.t == 12, False, info


@pytest.mark.parametrize("n,dilation", [(1, 1), (3, 1), (2, 3)])
def test_frame_stack_is_the_wrapper_with_the_frames_as_channels(n, dilation):
    # Round ends at steps 4 and 9 (the episode goes on) and at step 12 (the episode ends: no refill)
    ends = (4, 9, 12)
    wrapper = oracles.FrameStackWrapper(FramesEnv(ends), n, ["rgb"], dilation)
    env = FramesEnv(ends)
    transform = FrameStack(n, dilation, refill_on=["round_done", "stage_done", "game_done"])
    transform.space(env.observation_space["rgb"])
    for episode in range(2):
        expected, _ = wrapper.reset()
        obs, info = env.reset()
        transform.reset()
        out = transform.frame(obs["rgb"], {**info, "terminated": False, "truncated": False})
        assert np.array_equal(out, expected["rgb"].reshape(-1, 4, 4))
        done = False
        while not done:
            expected, _, done, _, _ = wrapper.step(0)
            obs, _, terminated, truncated, info = env.step(0)
            out = transform.frame(obs["rgb"], {**info, "terminated": terminated, "truncated": truncated})
            assert out.shape == (n * 3, 4, 4)
            assert np.array_equal(out, expected["rgb"].reshape(-1, 4, 4))


def test_standardize_is_the_obs_normalizer_bit_for_bit():
    space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float64)
    oracle = oracles.ObsNormalizer(gym.spaces.Dict({"state": space}), ["state"], clip=5.0, epsilon=1e-8)
    transform = Standardize(clip=5.0, epsilon=1e-8)
    transform.space(space)
    rng = np.random.default_rng(0)
    for step in range(20):
        rows = rng.normal(3.0, 4.0, (2, 3))
        # Every third step, the last observations of ended episodes: normalized without being added
        update = step % 3 != 2
        if update:
            oracle.update({"state": rows})
        expected = oracle.normalize({"state": rows})["state"]
        out = transform.envs(rows, update=update)
        assert out.dtype == np.float32 and np.array_equal(out, expected)
    assert np.array_equal(transform.stats.mean.numpy(), oracle.stats["state"].mean)
    assert np.array_equal(transform.stats.var.numpy(), oracle.stats["state"].var)


def test_standardize_restores_its_statistics():
    space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
    trained, restored = Standardize(), Standardize()
    trained.space(space)
    restored.space(space)
    rng = np.random.default_rng(0)
    for _ in range(5):
        trained.envs(rng.normal(size=(4, 3)))
    state = trained.state_dict()
    # Only the statistics: the ones of the last sync and the ones added since are not saved
    assert sorted(state) == ["stats.count", "stats.mean", "stats.var"]
    restored.load_state_dict(state)
    rows = rng.normal(size=(4, 3))
    assert np.array_equal(restored.envs(rows, update=False), trained.envs(rows, update=False))
    # The loaded statistics are the synced ones, with nothing added since
    assert torch.equal(restored.stats.synced_mean, trained.stats.mean)
    assert float(restored.stats.added_count) == 0


def process_rows(rank):
    rng = np.random.default_rng(rank)
    return [[rng.normal(rank, 1 + rank, size=(n, 2)) for n in sizes] for sizes in ([1, 3], [2 + rank])]


def standardize_on_two_processes(fabric):
    transform = Standardize()
    transform.space(gym.spaces.Box(-np.inf, np.inf, (2,), np.float64))
    for rollout in process_rows(fabric.global_rank):
        for rows in rollout:
            transform.envs(rows)
        transform.sync(fabric)
    stats = transform.stats
    moments = torch.cat([stats.mean, stats.var, stats.count[None]])
    gathered = fabric.all_gather(moments)
    samples = np.concatenate([rows for rank in range(2) for rollout in process_rows(rank) for rows in rollout])
    return (
        torch.equal(gathered[0], gathered[1])
        and np.allclose(stats.mean.numpy(), samples.mean(axis=0), rtol=1e-4)
        and np.allclose(stats.var.numpy(), samples.var(axis=0), rtol=1e-4)
        and float(stats.count) == pytest.approx(len(samples) + 1e-4)
    )


@pytest.mark.skipif(sys.platform == "win32", reason="2 processes")
def test_standardize_syncs_the_statistics_of_all_the_processes():
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    assert fabric.launch(standardize_on_two_processes)


class TrainOnly(Transform):
    """Adds 1, in the training batches only."""

    where = "batch"

    def __init__(self):
        super().__init__(phases=("train",))

    def batch(self, x):
        return x + 1

    def inverse(self, x):
        return x - 1


def test_a_pipeline_runs_its_stages_in_order_and_reports_their_spaces():
    space = gym.spaces.Box(0, 255, (3, 32, 32), U8)
    pipeline = Pipeline([Image(16), FrameStack(2), Standardize(), Scale(), TrainOnly()], space)
    assert pipeline.frame_space.shape == (6, 16, 16) and pipeline.frame_space.dtype == U8
    assert pipeline.envs_space.shape == (6, 16, 16) and pipeline.envs_space.dtype == np.float32
    assert pipeline.batch_space.shape == (6, 16, 16)
    # Built from the space of the store, it holds only the batch stage
    store_pipeline = Pipeline([Image(16), Scale(), TrainOnly()], gym.spaces.Box(0, 255, (3, 16, 16), U8), "batch")
    assert store_pipeline.is_identity("frame") and len(store_pipeline.transforms("batch")) == 2
    x = torch.full((2, 3, 16, 16), 255, dtype=torch.uint8)
    assert torch.equal(store_pipeline.batch(x, phase="act"), torch.full((2, 3, 16, 16), 0.5))
    assert torch.equal(store_pipeline.batch(x, phase="train"), torch.full((2, 3, 16, 16), 1.5))
    assert torch.equal(store_pipeline.inverse(store_pipeline.batch(x, phase="train")), x.float())


def test_a_pipeline_rejects_transforms_out_of_the_order_of_the_placements():
    with pytest.raises(ValueError, match="order of their placements"):
        Pipeline([Scale(), Image(16)], gym.spaces.Box(0, 255, (3, 32, 32), U8))
    with pytest.raises(ValueError, match="order of their placements"):
        Pipeline([Standardize(), Flatten()], gym.spaces.Box(0, 1, (2, 2), np.float32))


def test_a_transform_runs_only_at_the_placements_it_supports():
    assert Scale(where="frame").where == "frame"
    with pytest.raises(ValueError, match="runs at frame"):
        Image(16, where="batch")
    with pytest.raises(ValueError, match="phases"):
        Scale(phases=("test",))


def test_a_pipeline_is_built_from_the_configuration():
    cfg = [
        {"_target_": "sheeprl.transforms.Image", "size": 8, "grayscale": True},
        {"_target_": "sheeprl.transforms.Scale", "divisor": 255, "shift": 0.5},
    ]
    pipeline = Pipeline.from_config(cfg, gym.spaces.Box(0, 255, (3, 16, 16), U8))
    assert pipeline.frame_space.shape == (1, 8, 8)
    assert [type(t).__name__ for t in pipeline.transforms("batch")] == ["Scale"]
    with pytest.raises(TypeError, match="Transform"):
        Pipeline.from_config([{"_target_": "torch.nn.Identity"}], gym.spaces.Box(0, 1, (2,), np.float32))


def test_dequantize_keeps_the_top_bits_of_the_pixels_and_adds_less_than_one_level_of_noise():
    torch.manual_seed(0)
    pixels = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
    out = Dequantize(bits=5).batch(pixels)
    assert out.dtype == torch.float32 and out.min() >= -0.5 and out.max() < 0.5
    # 32 levels: the level of every pixel is its 5 top bits, the noise is below one level
    assert torch.equal(torch.floor((out + 0.5) * 32), torch.floor(pixels / 8))
    assert not torch.equal(out, Dequantize(bits=5).batch(pixels))
