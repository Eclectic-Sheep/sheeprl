from __future__ import annotations

import copy
import os
import shutil
from collections.abc import Sequence
from operator import itemgetter
from typing import Any

import gymnasium
import hydra
import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch import Tensor
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler
from torchmetrics import SumMetric

from sheeprl.algos.mopo.agent import build_agent
from sheeprl.algos.mopo.loss import world_model_loss
from sheeprl.algos.mopo.utils import test
from sheeprl.algos.sac.sac import train as train_sac
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.d4rl import TERMINATION_FUNCTIONS
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import dotdict, save_configs


def train_ensembles(
    fabric: Fabric,
    ensembles: _FabricModule,
    ensembles_optimizer: _FabricOptimizer,
    batch_size: int,
    training_offline_rb: ReplayBuffer,
    weight_decays: Sequence[float],
    cfg: dotdict[str, Any],
    max_epochs: float = float("inf"),
    max_epochs_since_update: int = 5,
    validation_offline_rb: ReplayBuffer | None = None,
    aggregator: MetricAggregator | None = None,
    log_dir: str = "./",
    state: dict[str, Any] | None = None,
):
    """World Model training."""
    # Setup training
    epoch = (state or {}).get("wm_epoch", 0)
    epochs_since_update = (state or {}).get("epochs_since_update", 0)

    if cfg.buffer.share_data:
        train_idxes = [torch.arange(training_offline_rb.buffer_size) for _ in range(ensembles.num_ensembles)]
        sampler = DistributedSampler(
            train_idxes[0],
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=cfg.seed,
        )
    else:
        train_idxes = [torch.randperm(training_offline_rb.buffer_size) for _ in range(ensembles.num_ensembles)]
        sampler = RandomSampler(train_idxes[0])
    sampler = BatchSampler(sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False)

    train_obs, train_next_obs, train_actions, train_rewards = itemgetter(
        "observations", "next_observations", "actions", "rewards"
    )(training_offline_rb.to_tensor(dtype=torch.float32, device=fabric.device))
    train_obs: Tensor = train_obs.squeeze(1)
    train_next_obs: Tensor = train_next_obs.squeeze(1)
    train_actions: Tensor = train_actions.squeeze(1)
    train_rewards: Tensor = train_rewards.squeeze(1)
    ensembles.scaler.fit(torch.cat((train_obs, train_actions), dim=-1))

    # Setpu validation
    best_val_losses = (state or {}).get("best_val_losses", torch.ones(ensembles.num_ensembles) * torch.inf)
    best_ensembles_states = (state or {}).get(
        "best_ensembles_states", [copy.deepcopy(model.state_dict()) for model in ensembles._models]
    )
    if validation_offline_rb is not None:
        validation_dataset = {
            k: v.squeeze(1)
            for k, v in validation_offline_rb.to_tensor(dtype=torch.float32, device=fabric.device).items()
        }
    else:
        val_idxes = torch.multinomial(torch.ones(training_offline_rb.buffer_size), num_samples=1000)
        validation_dataset = {
            k: v[val_idxes] for k, v in training_offline_rb.to_tensor(dtype=torch.float32, device=fabric.device).items()
        }
    val_obs = validation_dataset["observations"]
    val_actions = validation_dataset["actions"]
    val_target_obs = validation_dataset["next_observations"] - val_obs
    val_target_rewards = validation_dataset["rewards"]
    val_targets = torch.cat((val_target_rewards, val_target_obs), dim=-1)

    while epochs_since_update < max_epochs_since_update and epoch < max_epochs:
        print(f"Starting Epoch: {epoch}")
        if not cfg.buffer.share_data:
            train_idxes = [torch.randperm(training_offline_rb.buffer_size) for _ in range(ensembles.num_ensembles)]
        for i, batch_idxes in enumerate(sampler):
            batch = {
                "rewards": torch.stack([train_rewards[idxes[batch_idxes]] for idxes in train_idxes], dim=0).to(
                    device=fabric.device, dtype=torch.float32
                ),
                "actions": torch.stack([train_actions[idxes[batch_idxes]] for idxes in train_idxes], dim=0).to(
                    device=fabric.device, dtype=torch.float32
                ),
                "observations": torch.stack([train_obs[idxes[batch_idxes]] for idxes in train_idxes], dim=0).to(
                    device=fabric.device, dtype=torch.float32
                ),
                "next_observations": torch.stack(
                    [train_next_obs[idxes[batch_idxes]] for idxes in train_idxes], dim=0
                ).to(device=fabric.device, dtype=torch.float32),
            }
            targets = torch.cat((batch["rewards"], batch["next_observations"] - batch["observations"]), dim=-1)
            mean, logvar = ensembles(batch["observations"], batch["actions"])
            ensembles_optimizer.zero_grad(set_to_none=True)
            loss = world_model_loss(
                mean,
                logvar,
                targets,
                weight_decays=weight_decays,
                ensembles=ensembles,
                min_logvar=ensembles.min_logvar,
                max_logvar=ensembles.max_logvar,
            )
            fabric.backward(loss)
            grads = 0
            for p in ensembles.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grads += param_norm.item() ** 2
            grads = grads ** (1.0 / 2)
            ensembles_optimizer.step()

            if aggregator and not aggregator.disabled:
                aggregator.update("Grads/world_model", grads)
                aggregator.update("Loss/train_world_model", loss)
        epoch += 1

        validation_losses = ensembles.validate(val_obs, val_actions, val_targets)
        if aggregator and not aggregator.disabled:
            aggregator.update("Loss/validation_world_model", torch.mean(validation_losses))
        with torch.inference_mode():
            updated = False
            if best_val_losses.isinf().any():
                best_val_losses = validation_losses
                best_ensembles_states = [copy.deepcopy(model.state_dict()) for model in ensembles._models]
            for i, (val_loss, best) in enumerate(zip(validation_losses, best_val_losses, strict=True)):
                if (best - val_loss) / best > 0.01:
                    updated = True
                    best_val_losses[i] = val_loss
                    best_ensembles_states[i] = copy.deepcopy(ensembles._models[i].state_dict())
            epochs_since_update = epochs_since_update * int(not updated) + int(not updated)

        if aggregator and not aggregator.disabled:
            metrics_dict = aggregator.compute()
            fabric.log_dict(metrics_dict, epoch)
            aggregator.reset()

        # Checkpoint Model
        state = {
            "ensembles": ensembles.state_dict(),
            "ensembles_optimizer": ensembles_optimizer.state_dict(),
            "wm_epoch": epoch,
            "epochs_since_update": epochs_since_update,
            "best_val_losses": best_val_losses.cpu(),
            "best_ensembles_states": best_ensembles_states,
        }
        ckpt_path = os.path.join(log_dir, "checkpoint", f"wm_ckpt_{epoch}_{fabric.global_rank}.ckpt")
        fabric.call(
            "on_checkpoint_coupled",
            fabric=fabric,
            ckpt_path=ckpt_path,
            state=state,
            replay_buffer=None,
        )

    # Load best ensembles
    [
        model.load_state_dict(best_state)
        for model, best_state in zip(ensembles._models, best_ensembles_states, strict=True)
    ]

    # Final evaluation
    validation_losses = ensembles.validate(val_obs, val_actions, val_targets, True)
    fabric.log("Loss/validation_world_model", torch.mean(validation_losses), epoch)


@register_algorithm()
def main(fabric: Fabric, cfg: dotdict[str, Any]):
    """MOPO main function."""
    device = fabric.device

    state = None
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # Create Logger
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    env: gymnasium.Wrapper = hydra.utils.instantiate(cfg.env.wrapper)
    if cfg.env.capture_video and fabric.global_rank == 0 and log_dir is not None:
        env = gymnasium.experimental.wrappers.RecordVideoV0(
            env, os.path.join(log_dir, "train_videos"), disable_logger=True
        )
        env.metadata["render_fps"] = getattr(env, "frames_per_sec", 30)

    obs_space = env.observation_space
    if not (isinstance(obs_space, gymnasium.spaces.Dict)):
        raise RuntimeError(f"Invalid observation space, only Dict are allowed. Got {type(obs_space)}")
    action_space = env.action_space

    training_offline_rb, validation_offline_rb = env.get_dataset(
        validation_split=cfg.env.validation_split, seed=cfg.seed
    )
    rb = ReplayBuffer(
        buffer_size=cfg.buffer.size,
        n_envs=1,
        obs_keys=cfg.algo.mlp_keys.encoder,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint and state is not None and "rb" in state:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], ReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

    ensembles, sac_agent, sac_player = build_agent(fabric, cfg, obs_space, action_space)

    # Optimizers
    ensembles_optimizer = torch.optim.Adam(
        ensembles.parameters(), lr=cfg.algo.ensembles.optimizer.lr, eps=cfg.algo.ensembles.optimizer.eps
    )
    qf_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
        cfg.algo.critic.optimizer,
        params=sac_agent.qfs.parameters(),
        _convert_="all",
    )
    actor_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer,
        params=sac_agent.actor.parameters(),
        _convert_="all",
    )
    alpha_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
        cfg.algo.alpha.optimizer,
        params=[sac_agent.log_alpha],
        _convert_="all",
    )

    if cfg.checkpoint.resume_from:
        ensembles_optimizer.load_state_dict((state or {}).get("ensembles_optimizer", ensembles_optimizer.state_dict()))
        qf_optimizer.load_state_dict((state or {}).get("qf_optimizer", qf_optimizer.state_dict()))
        actor_optimizer.load_state_dict((state or {}).get("actor_optimizer", actor_optimizer.state_dict()))
        alpha_optimizer.load_state_dict((state or {}).get("alpha_optimizer", alpha_optimizer.state_dict()))

    ensembles_optimizer, qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        ensembles_optimizer, qf_optimizer, actor_optimizer, alpha_optimizer
    )  # type: ignore[attr-defined]

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Create a metric aggregator to log the metrics
    aggregator: MetricAggregator | None = None
    if not MetricAggregator.disabled:
        aggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Train ensembles
    if state is None or "wm_epoch" in state:
        train_ensembles(
            fabric=fabric,
            ensembles=ensembles,
            ensembles_optimizer=ensembles_optimizer,
            batch_size=cfg.algo.ensembles.batch_size,
            training_offline_rb=training_offline_rb,
            weight_decays=cfg.algo.ensembles.optimizer.weight_decays,
            max_epochs=cfg.algo.ensembles.max_epochs or float("inf"),
            max_epochs_since_update=cfg.algo.ensembles.max_epochs_since_update,
            validation_offline_rb=validation_offline_rb,
            aggregator=aggregator,
            log_dir=log_dir,
            state=state,
        )
        # Save Last `cfg.checkpoint.keep_last` world model chekpoints
        if os.path.isdir(os.path.join(log_dir, "checkpoint")):
            shutil.copytree(os.path.join(log_dir, "checkpoint"), os.path.join(log_dir, "checkpoint", "wm"))

    offline_rb, _ = env.get_dataset(validation_split=0, seed=cfg.seed)
    h = cfg.algo.h
    batch_size = cfg.algo.per_rank_batch_size
    num_epochs = cfg.algo.num_epochs
    epoch_length = cfg.algo.total_steps // num_epochs
    start_epoch = (state or {}).get("sac_epoch", 0)
    offline_rb_size = int(batch_size * 0.05)
    rollout_rb_size = batch_size - offline_rb_size
    rollout_batch_size = int(cfg.algo.rollout_batch_size)
    for epoch in range(start_epoch, num_epochs):
        # Rollout
        with torch.inference_mode():
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # for _ in range(rollout_batch_size // 10_000):
                initial_states = offline_rb.sample_tensors(
                    rollout_batch_size, n_samples=1, dtype=torch.float32, device=device
                )
                terminated = initial_states["terminated"]
                truncated = initial_states["truncated"]
                non_dones_mask = torch.logical_or(terminated, truncated).squeeze() == 0.0
                initial_states = {k: v.squeeze(0)[non_dones_mask] for k, v in initial_states.items()}
                obs = initial_states["observations"]
                for j in range(h):
                    actions: Tensor = sac_player(obs)
                    next_obs, rewards = ensembles.predict(obs, actions)
                    terminated = TERMINATION_FUNCTIONS[cfg.env.id](obs, actions, next_obs)
                    truncated = torch.zeros_like(terminated) + float(j == h - 1)
                    buffer_data = {
                        "observations": obs[:, None, ...].cpu().numpy(),
                        "actions": actions[:, None, ...].cpu().numpy(),
                        "rewards": rewards[:, None, ...].cpu().numpy(),
                        "next_observations": next_obs[:, None, ...].cpu().numpy(),
                        "terminated": terminated[:, None, ...].cpu().numpy(),
                        "truncated": truncated[:, None, ...].cpu().numpy(),
                    }
                    rb.add(buffer_data)
                    non_dones_mask = torch.logical_or(terminated, truncated).squeeze() == 0.0
                    if not non_dones_mask.any():
                        break
                    obs = next_obs[non_dones_mask]

        # Train SAC
        for _ in range(epoch_length):
            offline_data = offline_rb.sample_tensors(offline_rb_size, n_samples=1, dtype=torch.float32, device=device)
            rollout_data = rb.sample_tensors(rollout_rb_size, n_samples=1, dtype=torch.float32, device=device)
            data = {k: torch.cat((offline_data[k].squeeze(0), rollout_data[k].squeeze(0)), dim=0) for k in offline_data}
            with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                train_sac(
                    fabric=fabric,
                    agent=sac_agent,
                    actor_optimizer=actor_optimizer,
                    qf_optimizer=qf_optimizer,
                    alpha_optimizer=alpha_optimizer,
                    data=data,
                    aggregator=aggregator,
                    update=2,
                    cfg=cfg,
                    policy_steps_per_iter=1,
                )

        # Sync distributed timers
        if not timer.disabled:
            timer_metrics = timer.compute()
            if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                fabric.log(
                    "Time/sps_train",
                    epoch_length / timer_metrics["Time/train_time"],
                    epoch,
                )
            if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                fabric.log(
                    "Time/sps_env_interaction",
                    h / timer_metrics["Time/env_interaction_time"],
                    epoch,
                )
            timer.reset()

        # Checkpoint model
        state = {
            "ensembles": ensembles.state_dict(),
            "agent": sac_agent.state_dict(),
            "ensembles_optimizer": ensembles_optimizer.state_dict(),
            "qf_optimizer": qf_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "alpha_optimizer": alpha_optimizer.state_dict(),
            "sac_epoch": epoch + 1,
        }
        ckpt_path = os.path.join(log_dir, "checkpoint", f"sac_ckpt_{epoch}_{fabric.global_rank}.ckpt")
        fabric.call(
            "on_checkpoint_coupled",
            fabric=fabric,
            ckpt_path=ckpt_path,
            state=state,
            replay_buffer=rb if cfg.buffer.checkpoint else None,
        )

        # Evaluation
        if fabric.is_global_zero and cfg.algo.run_test:
            test(sac_player, fabric, env, cfg, epoch)

        # Log metrics
        if aggregator and not aggregator.disabled:
            metrics_dict = aggregator.compute()
            fabric.log_dict(metrics_dict, epoch)
            aggregator.reset()

    env.close()

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"ensembles": ensembles, "agent": sac_agent}
        register_model(fabric, log_models, cfg, models_to_log)
