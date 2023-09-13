import os
import time

import gymnasium as gym
import hydra
import torch
from lightning import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.plugins.collectives import TorchCollective
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchmetrics import MeanMetric

from sheeprl.algos.muzero.agent import MlpDynamics, MuzeroAgent, Predictor
from sheeprl.algos.muzero.loss import policy_loss, reward_loss, value_loss
from sheeprl.algos.muzero.utils import Node, test, visit_softmax_temperature
from sheeprl.data.buffers import EpisodeBuffer  # Trajectory, TrajectoryReplayBuffer
from sheeprl.models.models import MLP
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import create_tensorboard_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import nstep_returns


def apply_temperature(logits, temperature):
    """Returns `logits / temperature`, supporting also temperature=0.

    Taken from https://github.com/deepmind/mctx/blob/main/mctx/_src/policies.py#L409
    """
    # The max subtraction prevents +inf after dividing by a small temperature.
    logits = logits.float() - torch.max(logits, keepdims=True, axis=-1)[0]
    tiny = torch.finfo(logits.dtype).tiny
    return logits / max(tiny, temperature)


@register_algorithm()
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Set logger only on rank-0 but share the logger directory: since we don't know
    # what is happening during the `fabric.save()` method, at least we assure that all
    # ranks save under the same named folder.
    # As a plus, rank-0 sets the time uniquely for everyone
    world_collective = TorchCollective()
    if fabric.world_size > 1:
        world_collective.setup()
        world_collective.create_group()

    logger, log_dir = create_tensorboard_logger(fabric, cfg, "muzero")
    if fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Environment setup
    env = make_env(cfg.env.id, seed=cfg.seed + rank, idx=rank, capture_video=False)()
    assert isinstance(env.action_space, gym.spaces.Discrete), "Only discrete action space is supported"

    # Create the model
    embedding_size = 8
    full_support_size = 2 * cfg.algo.support_size + 1
    # TODO hydralize everything
    agent = MuzeroAgent(
        representation=MLP(
            input_dims=env.observation_space.shape,
            hidden_sizes=tuple(),
            output_dim=embedding_size,
            activation=torch.nn.ELU,
        ),
        dynamics=MlpDynamics(
            num_actions=env.action_space.n, embedding_size=embedding_size, full_support_size=full_support_size
        ),
        prediction=Predictor(
            embedding_size=embedding_size, num_actions=env.action_space.n, full_support_size=full_support_size
        ),
    )
    optimizer = hydra.utils.instantiate(cfg.algo.optimizer, params=agent.parameters())
    agent = fabric.setup_module(agent)
    optimizer = fabric.setup_optimizers(optimizer)

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(),
                "Game/ep_len_avg": MeanMetric(),
                "Time/step_per_second": MeanMetric(),
                "Loss/value_loss": MeanMetric(),
                "Loss/policy_loss": MeanMetric(),
                "Loss/reward_loss": MeanMetric(),
                "Loss/total_loss": MeanMetric(),
                "Gradient/gradient_norm": MeanMetric(),
                "Info/policy_entropy": MeanMetric(),
            }
        )

    # Local data
    buffer_size = cfg.buffer.size // int(fabric.world_size) if not cfg.dry_run else 1
    rb = EpisodeBuffer(
        buffer_size=buffer_size, sequence_length=cfg.chunk_sequence_len, device=device, memmap=cfg.buffer.memmap
    )

    # Global variables
    start_time = time.perf_counter()
    num_updates = int(cfg.total_steps // int(fabric.world_size)) if not cfg.dry_run else 1
    cfg.learning_starts = cfg.learning_starts // int(fabric.world_size) if not cfg.dry_run else 0

    env_steps = 0
    num_collected_trajectories = 1
    update_steps = 1
    while num_collected_trajectories <= num_updates:
        with torch.no_grad():
            # reset the episode at every update
            with device:
                # Get the first environment observation and start the optimization
                obs: torch.Tensor = torch.tensor(env.reset(seed=cfg.seed)[0], device=device).reshape(1, -1)
                rew_sum = 0.0

            steps_data = None
            for trajectory_step in range(0, cfg.buffer.max_trajectory_len):
                node = Node(prior=0, image=obs, device=device)

                # start MCTS
                node.mcts(
                    agent,
                    cfg.algo.num_simulations,
                    cfg.algo.gamma,
                    cfg.algo.dirichlet_alpha,
                    cfg.algo.exploration_fraction,
                    support_size=cfg.algo.support_size,
                )

                # Select action based on the visit count distribution and the temperature
                visits_count = torch.tensor([child.visit_count for child in node.children.values()])
                temperature = visit_softmax_temperature(training_steps=agent.training_steps)
                visit_probs = visits_count / cfg.algo.num_simulations
                visit_probs = torch.where(visit_probs > 0, visit_probs, 1 / visit_probs.shape[-1])
                tiny = torch.finfo(visit_probs.dtype).tiny
                visit_logits = torch.log(torch.maximum(visit_probs, torch.tensor(tiny, device=device)))
                logits = apply_temperature(visit_logits, temperature)
                action = torch.distributions.Categorical(logits=logits).sample()

                # Single environment step
                next_obs, reward, done, truncated, info = env.step(action.cpu().numpy().reshape(env.action_space.shape))
                rew_sum += reward

                # Store the current step data
                trajectory_step_data = TensorDict(
                    {
                        "policies": visit_probs.reshape(1, 1, -1),
                        "actions": action.reshape(1, 1, -1),
                        "observations": obs.unsqueeze(0),
                        "rewards": torch.tensor([reward]).reshape(1, 1, -1),
                        "values": node.value().reshape(1, 1, -1),
                        "dones": torch.tensor([done]).reshape(1, 1, -1),
                    },
                    batch_size=(1, 1),
                    device=device,
                )
                if steps_data is None:
                    steps_data = trajectory_step_data
                else:
                    steps_data = torch.cat([steps_data, trajectory_step_data])

                if done or truncated:
                    break
                else:
                    with device:
                        obs = torch.tensor(next_obs).reshape(1, -1)

            aggregator.update("Rewards/rew_avg", rew_sum)
            aggregator.update("Game/ep_len_avg", trajectory_step)
            # print("Finished episode")
            if len(steps_data) >= cfg.chunk_sequence_len:
                steps_data["returns"] = nstep_returns(
                    steps_data["rewards"],
                    steps_data["values"],
                    steps_data["dones"],
                    cfg.algo.nstep_horizon,
                    cfg.algo.gamma,
                )
                steps_data["weights"] = (
                    torch.abs(steps_data["returns"] - steps_data["values"]) ** cfg.algo.priority_alpha
                )
                rb.add(episode=steps_data)
                # update counter only if trajectory is long enough
                num_collected_trajectories += 1
            env_steps += trajectory_step

        if len(rb.buffer) >= cfg.learning_starts:
            print("UPDATING")
            all_data = rb.sample(batch_size=cfg.chunks_per_batch, n_samples=cfg.update_epochs, shuffle=True)

            for epoch_idx in range(cfg.update_epochs):
                # We sample one time to reduce the communications between processes
                data = all_data[epoch_idx]

                target_rewards = data["rewards"]
                target_values = data["returns"]
                target_policies = data["policies"]
                observations: torch.Tensor = data["observations"]  # shape should be (L, N, C, H, W)
                actions = data["actions"]

                hidden_states, policy_0, value_0 = agent.initial_inference(
                    observations[0]
                )  # in shape should be (N, C, H, W)
                # Policy loss
                pg_loss = policy_loss(policy_0, target_policies[0])
                # Value loss
                v_loss = value_loss(value_0, target_values[0])
                # Reward loss
                r_loss = torch.tensor(0.0, device=device)
                entropy = torch.distributions.Categorical(logits=policy_0.detach()).entropy().unsqueeze(0)

                for sequence_idx in range(1, cfg.chunk_sequence_len):
                    hidden_states, rewards, policies, values = agent.recurrent_inference(
                        actions[sequence_idx : sequence_idx + 1].to(dtype=torch.float32), hidden_states
                    )  # action should be (1, N, 1)
                    # Policy loss
                    pg_loss += policy_loss(policies.squeeze(), target_policies[sequence_idx])
                    # Value loss
                    v_loss += value_loss(values.squeeze(), target_values[sequence_idx])
                    # Reward loss
                    r_loss += reward_loss(rewards.squeeze(), target_rewards[sequence_idx])
                    entropy += torch.distributions.Categorical(logits=policies.detach()).entropy()

                # Equation (1) in the paper, the regularization loss is handled by `weight_decay` in the optimizer
                loss = (pg_loss + v_loss + r_loss) / cfg.chunk_sequence_len

                optimizer.zero_grad(set_to_none=True)
                fabric.backward(loss)
                optimizer.step()

                # Update metrics
                aggregator.update("Loss/policy_loss", pg_loss.detach())
                aggregator.update("Loss/value_loss", v_loss.detach())
                aggregator.update("Loss/reward_loss", r_loss.detach())
                aggregator.update("Loss/total_loss", loss.detach())
                aggregator.update("Gradient/gradient_norm", agent.gradient_norm())
                aggregator.update("Info/policy_entropy", entropy.mean() / cfg.chunk_sequence_len)
            update_steps += 1
        aggregator.update("Time/step_per_second", int(update_steps / (time.perf_counter() - start_time)))
        fabric.log_dict(aggregator.compute(), env_steps)
        aggregator.reset()

        if (
            (cfg.checkpoint.every > 0 and update_steps % cfg.checkpoint.every == 0)
            or cfg.dry_run
            or update_steps == num_updates
        ):
            state = {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_steps": update_steps,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{update_steps}_{fabric.global_rank}.ckpt")
            fabric.call("on_checkpoint_coupled", fabric=fabric, ckpt_path=ckpt_path, state=state)

    env.close()

    if fabric.is_global_zero:
        test_env = make_env(
            cfg.env.id,
            None,
            0,
            True,
            fabric.logger.log_dir,
            "test",
            vector_env_idx=0,
        )()
        test(agent, test_env, fabric, cfg)


if __name__ == "__main__":
    main()
