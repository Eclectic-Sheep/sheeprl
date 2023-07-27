import copy
import os
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from math import prod
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.accelerators import CUDAAccelerator, TPUAccelerator
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.plugins.collectives.collective import CollectibleGroup
from lightning.fabric.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.fabric.wrappers import _FabricModule
from tensordict import TensorDict, make_tensordict
from tensordict.tensordict import TensorDictBase
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.sac.loss import critic_loss, entropy_loss, policy_loss
from sheeprl.algos.sac_pixel.agent import (
    CNNDecoder,
    CNNEncoder,
    MLPDecoder,
    MLPEncoder,
    SACPixelAgent,
    SACPixelContinuousActor,
    SACPixelCritic,
    SACPixelQFunction,
)
from sheeprl.algos.sac_pixel.args import SACPixelContinuousArgs
from sheeprl.algos.sac_pixel.utils import preprocess_obs, test_sac_pixel
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.models.models import MultiDecoder, MultiEncoder
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import make_dict_env


def train(
    fabric: Fabric,
    agent: SACPixelAgent,
    encoder: Union[MultiEncoder, _FabricModule],
    decoder: Union[MultiDecoder, _FabricModule],
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    encoder_optimizer: Optimizer,
    decoder_optimizer: Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    global_step: int,
    args: SACPixelContinuousArgs,
    group: Optional[CollectibleGroup] = None,
):
    cnn_keys = encoder.cnn_keys
    mlp_keys = encoder.mlp_keys
    data = data.to(fabric.device)
    normalized_obs = {}
    normalized_next_obs = {}
    for k in cnn_keys + mlp_keys:
        if k in cnn_keys:
            normalized_obs[k] = data[k] / 255.0
            normalized_next_obs[k] = data[k] / 255.0
        else:
            normalized_obs[k] = data[k]
            normalized_next_obs[k] = data[f"next_{k}"]

    # Update the soft-critic
    next_target_qf_value = agent.get_next_target_q_values(
        normalized_next_obs, data["rewards"], data["dones"], args.gamma
    )
    qf_values = agent.get_q_values(normalized_obs, data["actions"])
    qf_loss = critic_loss(qf_values, next_target_qf_value, agent.num_critics)
    qf_optimizer.zero_grad(set_to_none=True)
    fabric.backward(qf_loss)
    qf_optimizer.step()
    aggregator.update("Loss/value_loss", qf_loss)

    # Update the target networks with EMA
    if global_step % args.target_network_frequency == 0:
        agent.critic_target_ema()
        agent.critic_encoder_target_ema()

    # Update the actor
    if global_step % args.actor_network_frequency == 0:
        actions, logprobs = agent.get_actions_and_log_probs(normalized_obs, detach_encoder_features=True)
        qf_values = agent.get_q_values(normalized_obs, actions, detach_encoder_features=True)
        min_qf_values = torch.min(qf_values, dim=-1, keepdim=True)[0]
        actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
        actor_optimizer.zero_grad(set_to_none=True)
        fabric.backward(actor_loss)
        actor_optimizer.step()
        aggregator.update("Loss/policy_loss", actor_loss)

        # Update the entropy value
        alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
        alpha_optimizer.zero_grad(set_to_none=True)
        fabric.backward(alpha_loss)
        agent.log_alpha.grad = fabric.all_reduce(agent.log_alpha.grad, group=group)
        alpha_optimizer.step()
        aggregator.update("Loss/alpha_loss", alpha_loss)

    # Update the decoder
    if global_step % args.decoder_update_freq == 0:
        hidden = encoder(normalized_obs)
        reconstruction = decoder(hidden)
        reconstruction_loss = 0
        for k in cnn_keys + mlp_keys:
            target = preprocess_obs(data[k], bits=5) if k in cnn_keys else data[k]
            reconstruction_loss += (
                F.mse_loss(target, reconstruction[k])  # Reconstruction
                + args.decoder_l2_lambda * (0.5 * hidden.pow(2).sum(1)).mean()  # L2 penalty on the hidden state
            )
        encoder_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        fabric.backward(reconstruction_loss)
        encoder_optimizer.step()
        decoder_optimizer.step()
        aggregator.update("Loss/reconstruction_loss", reconstruction_loss)


@register_algorithm()
def main():
    parser = HfArgumentParser(SACPixelContinuousArgs)
    args: SACPixelContinuousArgs = parser.parse_args_into_dataclasses()[0]
    # These arguments cannot be changed
    args.sample_next_obs = False
    args.screen_size = 64

    # Initialize Fabric
    devices = os.environ.get("LT_DEVICES", None)
    strategy = os.environ.get("LT_STRATEGY", None)
    is_tpu_available = TPUAccelerator.is_available()
    if strategy is not None:
        warnings.warn(
            "You are running the SAC-Pixel-Continuous algorithm through the Lightning CLI and you have specified a strategy: "
            f"`lightning run model --strategy={strategy}`. This algorithm is run with the "
            "`lightning.fabric.strategies.DDPStrategy` strategy, unless a TPU is available."
        )
        os.environ.pop("LT_STRATEGY")
    if is_tpu_available:
        strategy = "auto"
    else:
        strategy = DDPStrategy(find_unused_parameters=True)
        if devices == "1":
            strategy = SingleDeviceStrategy(device="cuda:0" if CUDAAccelerator.is_available() else "cpu")
    fabric = Fabric(strategy=strategy, callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    world_collective = TorchCollective()
    if fabric.world_size > 1:
        world_collective.setup()
        world_collective.create_group()
    if rank == 0:
        root_dir = (
            args.root_dir
            if args.root_dir is not None
            else os.path.join("logs", "sac_pixel_continuous", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        )
        logger = TensorBoardLogger(root_dir=root_dir, name=run_name)
        fabric._loggers = [logger]
        log_dir = logger.log_dir
        fabric.logger.log_hyperparams(asdict(args))
        if fabric.world_size > 1:
            world_collective.broadcast_object_list([log_dir], src=0)

        # Save args as dict automatically
        args.log_dir = log_dir
    else:
        data = [None]
        world_collective.broadcast_object_list(data, src=0)
        log_dir = data[0]
        os.makedirs(log_dir, exist_ok=True)

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if args.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_dict_env(
                args.env_id,
                args.seed + rank * args.num_envs + i,
                rank,
                args,
                logger.log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(args.num_envs)
        ]
    )
    cnn_keys = []
    mlp_keys = []
    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        cnn_keys = []
        for k, v in envs.single_observation_space.spaces.items():
            if args.cnn_keys and (
                k in args.cnn_keys or (len(args.cnn_keys) == 1 and args.cnn_keys[0].lower() == "all")
            ):
                if len(v.shape) in {3, 4}:
                    cnn_keys.append(k)
                else:
                    fabric.print(
                        f"Found a CNN key which is not an image: `{k}` of shape {v.shape}. "
                        "Try to transform the observation from the environment into a 3D image"
                    )
        for k, v in envs.single_observation_space.spaces.items():
            if args.mlp_keys and (
                k in args.mlp_keys or (len(args.mlp_keys) == 1 and args.mlp_keys[0].lower() == "all")
            ):
                if len(v.shape) == 1:
                    mlp_keys.append(k)
                else:
                    fabric.print(
                        f"Found an MLP key which is not a vector: `{k}` of shape {v.shape}. "
                        "Try to flatten the observation from the environment"
                    )
    else:
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {envs.single_observation_space}")
    if cnn_keys == [] and mlp_keys == []:
        raise RuntimeError(f"There must be at least one valid observation.")
    fabric.print("CNN keys:", cnn_keys)
    fabric.print("MLP keys:", mlp_keys)

    # Define the agent and the optimizer and setup them with Fabric
    act_dim = prod(envs.single_action_space.shape)
    target_entropy = -act_dim

    # Define the encoder and decoder and setup them with fabric.
    # Then we will set the critic encoder and actor decoder as the unwrapped encoder module:
    # we do not need it wrapped with the strategy inside actor and critic
    if args.cnn_channels_multiplier <= 0:
        raise ValueError(f"cnn_channels_multiplier must be greater than zero, given {args.cnn_channels_multiplier}")
    if args.dense_units <= 0:
        raise ValueError(f"dense_units must be greater than zero, given {args.dense_units}")
    try:
        dense_act = getattr(nn, args.dense_act)
    except:
        raise ValueError(
            f"Invalid value for mlp_act, given {args.dense_act}, must be one of https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        )

    cnn_channels = [prod(envs.single_observation_space[k].shape[:-2]) for k in cnn_keys]
    mlp_splits = [envs.single_observation_space[k].shape[0] for k in mlp_keys]
    cnn_encoder = (
        CNNEncoder(
            in_channels=sum(cnn_channels),
            features_dim=args.features_dim,
            keys=cnn_keys,
            screen_size=args.screen_size,
            cnn_channels_multiplier=args.cnn_channels_multiplier,
        )
        if cnn_keys is not None and len(cnn_keys) > 0
        else None
    )
    mlp_encoder = (
        MLPEncoder(
            sum(mlp_splits),
            mlp_keys,
            args.dense_units,
            args.mlp_layers,
            dense_act,
            args.layer_norm,
        )
        if mlp_keys is not None and len(mlp_keys) > 0
        else None
    )

    encoder = MultiEncoder(cnn_encoder, mlp_encoder, fabric.device)
    cnn_decoder = (
        CNNDecoder(
            cnn_encoder.conv_output_shape,
            features_dim=encoder.output_dim,
            keys=cnn_keys,
            channels=cnn_channels,
            screen_size=args.screen_size,
            cnn_channels_multiplier=args.cnn_channels_multiplier,
        )
        if cnn_keys is not None and len(cnn_keys) > 0
        else None
    )
    mlp_decoder = (
        MLPDecoder(
            encoder.output_dim,
            mlp_splits,
            mlp_keys,
            args.dense_units,
            args.mlp_layers,
            dense_act,
            args.layer_norm,
        )
        if mlp_keys is not None and len(mlp_keys) > 0
        else None
    )
    decoder = MultiDecoder(cnn_decoder, mlp_decoder, fabric.device)

    encoder = fabric.setup_module(encoder)
    decoder = fabric.setup_module(decoder)

    # Setup actor and critic. Those will initialize with orthogonal weights
    # both the actor and critic
    actor = SACPixelContinuousActor(
        encoder=copy.deepcopy(encoder.module),
        action_dim=act_dim,
        hidden_size=args.actor_hidden_size,
        action_low=envs.single_action_space.low,
        action_high=envs.single_action_space.high,
    )
    qfs = [
        SACPixelQFunction(
            input_dim=encoder.output_dim, action_dim=act_dim, hidden_size=args.critic_hidden_size, output_dim=1
        )
        for _ in range(args.num_critics)
    ]
    critic = SACPixelCritic(encoder=encoder.module, qfs=qfs)
    actor = fabric.setup_module(actor)
    critic = fabric.setup_module(critic)

    # The agent will tied convolutional weights between the encoder actor and critic
    agent = SACPixelAgent(
        actor,
        critic,
        target_entropy,
        alpha=args.alpha,
        tau=args.tau,
        encoder_tau=args.encoder_tau,
        device=fabric.device,
    )

    # Optimizers
    qf_optimizer, actor_optimizer, alpha_optimizer, encoder_optimizer, decoder_optimizer = fabric.setup_optimizers(
        Adam(agent.critic.parameters(), lr=args.q_lr),
        Adam(agent.actor.parameters(), lr=args.policy_lr),
        Adam([agent.log_alpha], lr=args.alpha_lr, betas=(0.5, 0.999)),
        Adam(encoder.parameters(), lr=args.encoder_lr),
        Adam(decoder.parameters(), lr=args.decoder_lr, weight_decay=args.decoder_wd),
    )

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(),
                "Game/ep_len_avg": MeanMetric(),
                "Time/step_per_second": MeanMetric(),
                "Loss/value_loss": MeanMetric(),
                "Loss/policy_loss": MeanMetric(),
                "Loss/alpha_loss": MeanMetric(),
                "Loss/reconstruction_loss": MeanMetric(),
            }
        )

    # Local data
    buffer_size = args.buffer_size // int(args.num_envs * fabric.world_size) if not args.dry_run else 1
    rb = ReplayBuffer(
        buffer_size, args.num_envs, device=fabric.device if args.memmap_buffer else "cpu", memmap=args.memmap_buffer
    )
    step_data = TensorDict({}, batch_size=[args.num_envs], device=fabric.device if args.memmap_buffer else "cpu")

    # Global variables
    start_time = time.time()
    num_updates = int(args.total_steps // (args.num_envs * fabric.world_size)) if not args.dry_run else 1
    args.learning_starts = args.learning_starts // int(args.num_envs * fabric.world_size) if not args.dry_run else 0

    # Get the first environment observation and start the optimization
    with device:
        o = envs.reset(seed=args.seed)[0]  # [N_envs, N_obs]
        obs = {}
        for k in o.keys():
            if k in mlp_keys + cnn_keys:
                with fabric.device:
                    torch_obs = torch.from_numpy(o[k])
                    if k in cnn_keys:
                        torch_obs = torch_obs.view(args.num_envs, -1, *torch_obs.shape[-2:])
                    step_data[k] = torch_obs
                obs[k] = torch_obs

    for global_step in range(1, num_updates + 1):
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                normalized_obs = {k: v / 255 if k in cnn_keys else torch_obs.float() for k, v in obs.items()}
                actions, _ = actor.module(normalized_obs)
                actions = actions.cpu().numpy()
        o, rewards, dones, truncated, infos = envs.step(actions)
        dones = np.logical_or(dones, truncated)

        if "final_info" in infos:
            for i, agent_final_info in enumerate(infos["final_info"]):
                if agent_final_info is not None and "episode" in agent_final_info:
                    fabric.print(
                        f"Rank-0: global_step={global_step}, reward_env_{i}={agent_final_info['episode']['r'][0]}"
                    )
                    aggregator.update("Rewards/rew_avg", agent_final_info["episode"]["r"][0])
                    aggregator.update("Game/ep_len_avg", agent_final_info["episode"]["l"][0])

        # Save the real next observation
        real_next_obs = copy.deepcopy(o)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v

        next_obs = {}
        for k in real_next_obs.keys():  # [N_envs, N_obs]
            step_data[k] = obs[k]
            next_obs[k] = torch.from_numpy(o[k]).view(args.num_envs, *o[k].shape[1:])
            if not args.sample_next_obs:
                step_data[f"next_{k}"] = torch.from_numpy(real_next_obs[k])
            if k in cnn_keys:
                next_obs[k] = next_obs[k].view(args.num_envs, -1, *next_obs[k].shape[-2:])
                step_data[f"next_{k}"] = step_data[f"next_{k}"].view(
                    args.num_envs, -1, *step_data[f"next_{k}"].shape[-2:]
                )
        actions = torch.from_numpy(actions).view(args.num_envs, -1).float()
        rewards = torch.from_numpy(rewards).view(args.num_envs, -1).float()
        dones = torch.from_numpy(dones).view(args.num_envs, -1).float()

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["rewards"] = rewards
        rb.add(step_data.unsqueeze(0))

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if global_step >= args.learning_starts - 1:
            training_steps = args.learning_starts if global_step == args.learning_starts - 1 else 1
            for _ in range(training_steps):
                # We sample one time to reduce the communications between processes
                sample = rb.sample(
                    args.gradient_steps * args.per_rank_batch_size, sample_next_obs=args.sample_next_obs
                )  # [G*B, 1]
                gathered_data = fabric.all_gather(sample.to_dict())  # [G*B, World, 1]
                gathered_data = make_tensordict(gathered_data).view(-1)  # [G*B*World]
                if fabric.world_size > 1:
                    dist_sampler: DistributedSampler = DistributedSampler(
                        range(len(gathered_data)),
                        num_replicas=fabric.world_size,
                        rank=fabric.global_rank,
                        shuffle=True,
                        seed=args.seed,
                        drop_last=False,
                    )
                    sampler: BatchSampler = BatchSampler(
                        sampler=dist_sampler, batch_size=args.per_rank_batch_size, drop_last=False
                    )
                else:
                    sampler = BatchSampler(
                        sampler=range(len(gathered_data)), batch_size=args.per_rank_batch_size, drop_last=False
                    )
                for batch_idxes in sampler:
                    train(
                        fabric,
                        agent,
                        encoder,
                        decoder,
                        actor_optimizer,
                        qf_optimizer,
                        alpha_optimizer,
                        encoder_optimizer,
                        decoder_optimizer,
                        gathered_data[batch_idxes],
                        aggregator,
                        global_step,
                        args,
                    )
        aggregator.update("Time/step_per_second", int(global_step / (time.time() - start_time)))
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint model
        if (args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0) or args.dry_run:
            state = {
                "agent": agent.state_dict(),
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "args": asdict(args),
                "global_step": global_step * fabric.world_size,
                "batch_size": args.per_rank_batch_size * fabric.world_size,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{global_step}_{fabric.global_rank}.ckpt")
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if args.checkpoint_buffer else None,
            )

    envs.close()
    if fabric.is_global_zero:
        test_env = make_dict_env(args.env_id, args.seed, 0, args, fabric.logger.log_dir, "test", vector_env_idx=0)()
        test_sac_pixel(actor.module, test_env, fabric, args, normalize=True)


if __name__ == "__main__":
    main()
