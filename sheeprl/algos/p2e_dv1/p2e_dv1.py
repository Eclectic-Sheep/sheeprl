import os
import pathlib
import time
from dataclasses import asdict
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from lightning.pytorch.utilities.seed import isolate_rng
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn
from torch.distributions import Bernoulli, Independent, Normal
from torch.optim import Adam
from torch.utils.data import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.dreamer_v1.agent import Player, WorldModel
from sheeprl.algos.dreamer_v1.loss import actor_loss, critic_loss, reconstruction_loss
from sheeprl.algos.dreamer_v1.utils import make_env, test
from sheeprl.algos.p2e_dv1.agent import build_models
from sheeprl.algos.p2e_dv1.args import P2EDV1Args
from sheeprl.data.buffers import SequentialReplayBuffer
from sheeprl.models.models import MLP
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.model import cnn_forward
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import compute_lambda_values, init_weights, polynomial_decay

# Decomment the following line if you are using MineDojo on an headless machine
# os.environ["MINEDOJO_HEADLESS"] = "1"


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor_task: _FabricModule,
    critic_task: _FabricModule,
    world_optimizer: _FabricOptimizer,
    actor_task_optimizer: _FabricOptimizer,
    critic_task_optimizer: _FabricOptimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: P2EDV1Args,
    ensembles: _FabricModule,
    ensemble_optimizer: _FabricOptimizer,
    actor_exploration: _FabricModule,
    critic_exploration: _FabricModule,
    actor_exploration_optimizer: _FabricOptimizer,
    critic_exploration_optimizer: _FabricOptimizer,
    is_exploring: True,
) -> None:
    """Runs one-step update of the agent.

    In particular, it updates the agent as specified by Algorithm 1 in
    [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960).

    The algorithm is made by different phases:
    1. Dynamic Learning: see Algorithm 1 in [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
    2. Ensemble Learning: learn the ensemble models as described in [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960).
        The ensemble models give the novelty of the state visited by the agent.
    3. Behaviour Learning Exploration: the agent learns to explore the environment, having as reward only the intrinsic reward, computed from the ensembles.
    4. Behaviour Learning Task (zero-shot): the agent learns to solve the task, the experiences it uses to learn it are the ones collected during the exploration:
        - Imagine trajectories in the latent space from each latent state s_t up to the horizon H: s'_(t+1), ..., s'_(t+H).
        - Predict rewards and values in the imagined trajectories.
        - Compute lambda targets (Eq. 6 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603))
        - Update the actor and the critic

    This method is based on [sheeprl.algos.dreamer_v1.dreamer_v1](sheeprl.algos.dreamer_v1.dreamer_v1) algorithm,
    extending it to implement the [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960).

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        actor_task (_FabricModule): the actor for solving the task.
        critic_task (_FabricModule): the critic for solving the task.
        world_optimizer (_FabricOptimizer): the world optimizer.
        actor_task_optimizer (_FabricOptimizer): the actor optimizer for solving the task.
        critic_task_optimizer (_FabricOptimizer): the critic optimizer for solving the task.
        data (TensorDictBase): the batch of data to use for training.
        aggregator (MetricAggregator): the aggregator to print the metrics.
        args (DreamerV1Args): the configs.
        ensembles (_FabricModule): the ensemble models.
        ensemble_optimizer (_FabricOptimizer): the optimizer of the ensemble models.
        actor_exploration (_FabricModule): the actor for exploration.
        critic_exploration (_FabricModule): the critic for exploration.
        actor_exploration_optimizer (_FabricOptimizer): the optimizer of the actor for exploration.
        critic_exploration_optimizer (_FabricOptimizer): the optimizer of the critic for exploration.
        is_exploring (bool): whether the agent is exploring.
    """
    batch_size = args.per_rank_batch_size
    sequence_length = args.per_rank_sequence_length
    observation_shape = data["observations"].shape[-3:]
    device = fabric.device
    batch_obs = data["observations"] / 255 - 0.5

    # Dynamic Learning
    recurrent_state = torch.zeros(1, batch_size, args.recurrent_state_size, device=device)
    posterior = torch.zeros(1, batch_size, args.stochastic_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, args.recurrent_state_size, device=device)
    posteriors = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    priors = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    posteriors_mean = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    posteriors_std = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    priors_mean = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    priors_std = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    embedded_obs = cnn_forward(world_model.encoder, batch_obs, observation_shape, (-1,))

    for i in range(0, sequence_length):
        recurrent_state, posterior, prior, posterior_mean_std, prior_mean_std = world_model.rssm.dynamic(
            posterior, recurrent_state, data["actions"][i : i + 1], embedded_obs[i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        posteriors[i] = posterior
        posteriors_mean[i] = posterior_mean_std[0]
        posteriors_std[i] = posterior_mean_std[1]
        priors_mean[i] = prior_mean_std[0]
        priors_std[i] = prior_mean_std[1]
        priors[i] = prior
    latent_states = torch.cat((posteriors, recurrent_states), -1)

    decoded_information = cnn_forward(
        world_model.observation_model, latent_states, (latent_states.shape[-1],), observation_shape
    )
    qo = Independent(Normal(decoded_information, 1), len(observation_shape))
    qr = Independent(Normal(world_model.reward_model(latent_states.detach()), 1), 1)
    if args.use_continues and world_model.continue_model:
        qc = Independent(Bernoulli(logits=world_model.continue_model(latent_states.detach()), validate_args=False), 1)
        continue_targets = (1 - data["dones"]) * args.gamma
    else:
        qc = continue_targets = None
    p = Independent(Normal(posteriors_mean, posteriors_std), 1)
    q = Independent(Normal(priors_mean, priors_std), 1)

    world_optimizer.zero_grad(set_to_none=True)
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        qo,
        batch_obs,
        qr,
        data["rewards"],
        p,
        q,
        args.kl_free_nats,
        args.kl_regularizer,
        qc,
        continue_targets,
        args.continue_scale_factor,
    )
    fabric.backward(rec_loss)
    if args.clip_gradients is not None and args.clip_gradients > 0:
        world_grad = fabric.clip_gradients(
            module=world_model, optimizer=world_optimizer, max_norm=args.clip_gradients, error_if_nonfinite=False
        )
        aggregator.update("Grads/world_model", world_grad.detach())
    world_optimizer.step()
    aggregator.update("Loss/reconstruction_loss", rec_loss.detach())
    aggregator.update("Loss/observation_loss", observation_loss.detach())
    aggregator.update("Loss/reward_loss", reward_loss.detach())
    aggregator.update("Loss/state_loss", state_loss.detach())
    aggregator.update("Loss/continue_loss", continue_loss.detach())
    aggregator.update("State/kl", kl.mean().detach())
    aggregator.update("State/p_entropy", p.entropy().mean().detach())
    aggregator.update("State/q_entropy", q.entropy().mean().detach())

    if is_exploring:
        # Ensemble Learning
        loss = 0.0
        ensemble_optimizer.zero_grad(set_to_none=True)
        for ens in ensembles:
            out = ens(torch.cat((posteriors.detach(), recurrent_states.detach(), data["actions"].detach()), -1))[:-1]
            next_obs_embedding_dist = Independent(Normal(out, 1), 1)
            loss -= next_obs_embedding_dist.log_prob(embedded_obs.detach()[1:]).mean()
        loss.backward()
        if args.ensemble_clip_gradients is not None and args.ensemble_clip_gradients > 0:
            ensemble_grad = fabric.clip_gradients(
                module=ens,
                optimizer=ensemble_optimizer,
                max_norm=args.ensemble_clip_gradients,
                error_if_nonfinite=False,
            )
            aggregator.update("Grads/ensemble", ensemble_grad.detach())
        ensemble_optimizer.step()
        aggregator.update(f"Loss/ensemble_loss", loss.detach().cpu())

        # Behaviour Learning Exploration
        imagined_prior = posteriors.detach().reshape(1, -1, args.stochastic_size)
        recurrent_state = recurrent_states.detach().reshape(1, -1, args.recurrent_state_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories = torch.empty(
            args.horizon, batch_size * sequence_length, args.stochastic_size + args.recurrent_state_size, device=device
        )
        # initialize the tensor of imagined actions, they are used to compute the intrinsic reward
        imagined_actions = torch.zeros(
            args.horizon, batch_size * sequence_length, data["actions"].shape[-1], device=device
        )

        # imagine trajectories in the latent space
        for i in range(args.horizon):
            actions = torch.cat(actor_exploration(imagined_latent_state.detach()), dim=-1)
            imagined_actions[i] = actions
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
            imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
            imagined_trajectories[i] = imagined_latent_state
        predicted_values = critic_exploration(imagined_trajectories)

        # Predict intrinsic reward
        next_obs_embedding = torch.zeros(
            len(ensembles),
            args.horizon,
            batch_size * sequence_length,
            embedded_obs.shape[-1],
            device=device,
        )
        for i, ens in enumerate(ensembles):
            next_obs_embedding[i] = ens(torch.cat((imagined_trajectories.detach(), imagined_actions.detach()), -1))

        # next_obs_embedding -> N_ensemble x Horizon x Batch_size*Seq_len x Obs_embedding_size
        intrinsic_reward = next_obs_embedding.var(0).mean(-1, keepdim=True) * args.intrinsic_reward_multiplier
        aggregator.update("Rewards/intrinsic", intrinsic_reward.detach().cpu().mean())

        if args.use_continues and world_model.continue_model:
            predicted_continues = Independent(
                Bernoulli(logits=world_model.continue_model(imagined_trajectories)), 1
            ).mean
        else:
            predicted_continues = torch.ones_like(intrinsic_reward.detach()) * args.gamma

        lambda_values = compute_lambda_values(
            intrinsic_reward,
            predicted_values,
            predicted_continues,
            last_values=predicted_values[-1],
            horizon=args.horizon,
            lmbda=args.lmbda,
        )

        aggregator.update("Values_exploration/predicted_values", predicted_values.detach().cpu().mean())
        aggregator.update("Values_exploration/lambda_values", lambda_values.detach().cpu().mean())

        with torch.no_grad():
            discount = torch.cumprod(
                torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0
            )

        actor_exploration_optimizer.zero_grad(set_to_none=True)
        policy_loss_exploration = actor_loss(discount * lambda_values)
        fabric.backward(policy_loss_exploration)
        if args.clip_gradients is not None and args.clip_gradients > 0:
            actor_exploration_grad = fabric.clip_gradients(
                module=actor_exploration,
                optimizer=actor_exploration_optimizer,
                max_norm=args.clip_gradients,
                error_if_nonfinite=False,
            )
            aggregator.update("Grads/actor_exploration", actor_exploration_grad.detach())
        actor_exploration_optimizer.step()
        aggregator.update("Loss/policy_loss_exploration", policy_loss_exploration.detach())

        qv = Independent(Normal(critic_exploration(imagined_trajectories.detach())[:-1], 1), 1)
        critic_exploration_optimizer.zero_grad(set_to_none=True)
        value_loss_exploration = critic_loss(qv, lambda_values.detach(), discount[..., 0])
        fabric.backward(value_loss_exploration)
        if args.clip_gradients is not None and args.clip_gradients > 0:
            critic_exploration_grad = fabric.clip_gradients(
                module=critic_exploration,
                optimizer=critic_exploration_optimizer,
                max_norm=args.clip_gradients,
                error_if_nonfinite=False,
            )
            aggregator.update("Grads/critic_exploration", critic_exploration_grad.detach())
        critic_exploration_optimizer.step()
        aggregator.update("Loss/value_loss_exploration", value_loss_exploration.detach())

    # reset the world_model gradients, to avoid interferences with task learning
    world_optimizer.zero_grad(set_to_none=True)

    # Behaviour Learning Task
    imagined_prior = posteriors.detach().reshape(1, -1, args.stochastic_size)
    recurrent_state = recurrent_states.detach().reshape(1, -1, args.recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = torch.empty(
        args.horizon, batch_size * sequence_length, args.stochastic_size + args.recurrent_state_size, device=device
    )
    for i in range(args.horizon):
        actions = torch.cat(actor_task(imagined_latent_state.detach()), dim=-1)
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state

    predicted_values = critic_task(imagined_trajectories)
    predicted_rewards = world_model.reward_model(imagined_trajectories)
    if args.use_continues and world_model.continue_model:
        predicted_continues = Independent(Bernoulli(logits=world_model.continue_model(imagined_trajectories)), 1).mean
    else:
        predicted_continues = torch.ones_like(predicted_rewards.detach()) * args.gamma

    lambda_values = compute_lambda_values(
        predicted_rewards,
        predicted_values,
        predicted_continues,
        last_values=predicted_values[-1],
        horizon=args.horizon,
        lmbda=args.lmbda,
    )

    with torch.no_grad():
        discount = torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0)

    actor_task_optimizer.zero_grad(set_to_none=True)
    policy_loss_task = actor_loss(discount * lambda_values)
    fabric.backward(policy_loss_task)
    if args.clip_gradients is not None and args.clip_gradients > 0:
        actor_task_grad = fabric.clip_gradients(
            module=actor_task, optimizer=actor_task_optimizer, max_norm=args.clip_gradients, error_if_nonfinite=False
        )
        aggregator.update("Grads/actor_task", actor_task_grad.detach())
    actor_task_optimizer.step()
    aggregator.update("Loss/policy_loss_task", policy_loss_task.detach())

    qv = Independent(Normal(critic_task(imagined_trajectories.detach())[:-1], 1), 1)
    critic_task_optimizer.zero_grad(set_to_none=True)
    value_loss = critic_loss(qv, lambda_values.detach(), discount[..., 0])
    fabric.backward(value_loss)
    if args.clip_gradients is not None and args.clip_gradients > 0:
        critic_task_grad = fabric.clip_gradients(
            module=critic_task, optimizer=critic_task_optimizer, max_norm=args.clip_gradients, error_if_nonfinite=False
        )
        aggregator.update("Grads/critic_task", critic_task_grad.detach())
    critic_task_optimizer.step()
    aggregator.update("Loss/value_loss_task", value_loss.detach())

    # Reset everything
    actor_exploration_optimizer.zero_grad(set_to_none=True)
    critic_exploration_optimizer.zero_grad(set_to_none=True)
    actor_task_optimizer.zero_grad(set_to_none=True)
    critic_task_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)
    ensemble_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main():
    parser = HfArgumentParser(P2EDV1Args)
    args: P2EDV1Args = parser.parse_args_into_dataclasses()[0]
    args.num_envs = 1
    torch.set_num_threads(1)

    # Initialize Fabric
    fabric = Fabric(callbacks=[CheckpointCallback()])
    if not _is_using_cli():
        fabric.launch()
    rank = fabric.global_rank
    device = fabric.device
    fabric.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.checkpoint_path:
        state = fabric.load(args.checkpoint_path)
        state["args"]["checkpoint_path"] = args.checkpoint_path
        args = P2EDV1Args(**state["args"])
        args.per_rank_batch_size = state["batch_size"] // fabric.world_size
        ckpt_path = pathlib.Path(args.checkpoint_path)

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
            else os.path.join("logs", "p2e_dv1", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        run_name = (
            args.run_name
            if args.run_name is not None
            else f"{args.env_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
        )
        if args.checkpoint_path:
            root_dir = ckpt_path.parent.parent
            run_name = "resume_from_checkpoint"
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

    env: gym.Env = make_env(
        args.env_id,
        args.seed + rank * args.num_envs,
        rank,
        args,
        logger.log_dir if rank == 0 else None,
        "train",
    )

    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
    actions_dim = (
        env.action_space.shape
        if is_continuous
        else (env.action_space.nvec.tolist() if is_multidiscrete else [env.action_space.n])
    )
    observation_shape = env.observation_space.shape
    clip_rewards_fn = lambda r: torch.tanh(r) if args.clip_rewards else r

    world_model, actor_task, critic_task, actor_exploration, critic_exploration = build_models(
        fabric,
        actions_dim,
        observation_shape,
        is_continuous,
        args,
        state["world_model"] if args.checkpoint_path else None,
        state["actor_task"] if args.checkpoint_path else None,
        state["critic_task"] if args.checkpoint_path else None,
        state["actor_exploration"] if args.checkpoint_path else None,
        state["critic_exploration"] if args.checkpoint_path else None,
    )

    # initialize the ensembles with different seeds to be sure they have different weights
    ens_list = []
    with isolate_rng():
        for i in range(args.num_ensembles):
            fabric.seed_everything(args.seed + i)
            ens_list.append(
                MLP(
                    input_dims=int(np.sum(actions_dim) + args.recurrent_state_size + args.stochastic_size),
                    output_dim=world_model.encoder.output_size,
                    hidden_sizes=[args.dense_units] * args.mlp_layers,
                ).apply(init_weights)
            )
    ensembles = nn.ModuleList(ens_list)
    if args.checkpoint_path:
        ensembles.load_state_dict(state["ensembles"])
    fabric.setup_module(ensembles)
    player = Player(
        world_model.encoder.module,
        world_model.rssm.recurrent_model.module,
        world_model.rssm.representation_model.module,
        actor_exploration.module,
        actions_dim,
        args.expl_amount,
        args.num_envs,
        args.stochastic_size,
        args.recurrent_state_size,
        fabric.device,
    )

    # Optimizers
    world_optimizer = Adam(world_model.parameters(), eps=1e-5, lr=args.world_lr, weight_decay=1e-6)
    actor_task_optimizer = Adam(actor_task.parameters(), eps=1e-5, lr=args.actor_lr, weight_decay=1e-6)
    critic_task_optimizer = Adam(critic_task.parameters(), eps=1e-5, lr=args.critic_lr, weight_decay=1e-6)
    actor_exploration_optimizer = Adam(actor_exploration.parameters(), eps=1e-5, lr=args.actor_lr, weight_decay=1e-6)
    critic_exploration_optimizer = Adam(critic_exploration.parameters(), eps=1e-5, lr=args.critic_lr, weight_decay=1e-6)
    ensemble_optimizer = Adam(ensembles.parameters(), eps=args.ensemble_eps, lr=args.ensemble_lr, weight_decay=1e-6)
    if args.checkpoint_path:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_task_optimizer.load_state_dict(state["actor_task_optimizer"])
        critic_task_optimizer.load_state_dict(state["critic_task_optimizer"])
        ensemble_optimizer.load_state_dict(state["ensemble_optimizer"])
        actor_exploration_optimizer.load_state_dict(state["actor_exploration_optimizer"])
        critic_exploration_optimizer.load_state_dict(state["critic_exploration_optimizer"])
    (
        world_optimizer,
        actor_task_optimizer,
        critic_task_optimizer,
        ensemble_optimizer,
        actor_exploration_optimizer,
        critic_exploration_optimizer,
    ) = fabric.setup_optimizers(
        world_optimizer,
        actor_task_optimizer,
        critic_task_optimizer,
        ensemble_optimizer,
        actor_exploration_optimizer,
        critic_exploration_optimizer,
    )

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(sync_on_compute=False),
                "Game/ep_len_avg": MeanMetric(sync_on_compute=False),
                "Time/step_per_second": MeanMetric(sync_on_compute=False),
                "Loss/reconstruction_loss": MeanMetric(sync_on_compute=False),
                "Loss/value_loss_task": MeanMetric(sync_on_compute=False),
                "Loss/policy_loss_task": MeanMetric(sync_on_compute=False),
                "Loss/value_loss_exploration": MeanMetric(sync_on_compute=False),
                "Loss/policy_loss_exploration": MeanMetric(sync_on_compute=False),
                "Loss/observation_loss": MeanMetric(sync_on_compute=False),
                "Loss/reward_loss": MeanMetric(sync_on_compute=False),
                "Loss/state_loss": MeanMetric(sync_on_compute=False),
                "Loss/continue_loss": MeanMetric(sync_on_compute=False),
                "Loss/ensemble_loss": MeanMetric(sync_on_compute=False),
                "State/kl": MeanMetric(sync_on_compute=False),
                "State/p_entropy": MeanMetric(sync_on_compute=False),
                "State/q_entropy": MeanMetric(sync_on_compute=False),
                "Params/exploration_amout": MeanMetric(sync_on_compute=False),
                "Rewards/intrinsic": MeanMetric(sync_on_compute=False),
                "Values_exploration/predicted_values": MeanMetric(sync_on_compute=False),
                "Values_exploration/lambda_values": MeanMetric(sync_on_compute=False),
                "Grads/world_model": MeanMetric(sync_on_compute=False),
                "Grads/actor_task": MeanMetric(sync_on_compute=False),
                "Grads/critic_task": MeanMetric(sync_on_compute=False),
                "Grads/actor_exploration": MeanMetric(sync_on_compute=False),
                "Grads/critic_exploration": MeanMetric(sync_on_compute=False),
                "Grads/ensemble": MeanMetric(sync_on_compute=False),
            }
        )

    # Local data
    buffer_size = (
        args.buffer_size // int(args.num_envs * fabric.world_size * args.action_repeat) if not args.dry_run else 4
    )
    rb = SequentialReplayBuffer(buffer_size, args.num_envs, device="cpu", memmap=args.memmap_buffer)
    if args.checkpoint_path and args.checkpoint_buffer:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], SequentialReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")
    step_data = TensorDict({}, batch_size=[args.num_envs], device="cpu")
    expl_decay_steps = state["expl_decay_steps"] if args.checkpoint_path else 0

    # Global variables
    start_time = time.perf_counter()
    start_step = state["global_step"] // fabric.world_size if args.checkpoint_path else 1
    step_before_training = args.train_every // (fabric.world_size * args.action_repeat) if not args.dry_run else 0
    num_updates = int(args.total_steps // (fabric.world_size * args.action_repeat)) if not args.dry_run else 4
    exploration_updates = (
        int(args.exploration_steps // (fabric.world_size * args.action_repeat)) if not args.dry_run else 4
    )
    exploration_updates = min(num_updates, exploration_updates)
    learning_starts = (args.learning_starts // (fabric.world_size * args.action_repeat)) if not args.dry_run else 3
    if args.checkpoint_path and not args.checkpoint_buffer:
        learning_starts = start_step + args.learning_starts // int(fabric.world_size * args.action_repeat)
    max_step_expl_decay = args.max_step_expl_decay // (args.gradient_steps * fabric.world_size)
    if args.checkpoint_path:
        player.expl_amount = polynomial_decay(
            expl_decay_steps,
            initial=args.expl_amount,
            final=args.expl_min,
            max_decay_steps=max_step_expl_decay,
        )

    # Get the first environment observation and start the optimization
    obs = torch.from_numpy(env.reset(seed=args.seed)[0].copy()).view(
        args.num_envs, *observation_shape
    )  # [N_envs, N_obs]
    step_data["dones"] = torch.zeros(args.num_envs, 1)
    step_data["actions"] = torch.zeros(args.num_envs, np.sum(actions_dim))
    step_data["rewards"] = torch.zeros(args.num_envs, 1)
    step_data["observations"] = obs
    rb.add(step_data[None, ...])
    player.init_states()

    is_exploring = True
    for global_step in range(start_step, num_updates + 1):
        if global_step == exploration_updates:
            is_exploring = False
            player.actor = actor_task.module
            # task test zero-shot
            if fabric.is_global_zero:
                test(player, fabric, args, "zero-shot")

        # Sample an action given the observation received by the environment
        if global_step <= learning_starts and args.checkpoint_path is None:
            real_actions = actions = np.array(env.action_space.sample())
            if not is_continuous:
                actions = np.concatenate(
                    [
                        F.one_hot(torch.tensor(act), act_dim).numpy()
                        for act, act_dim in zip(actions.reshape(len(actions_dim)), actions_dim)
                    ],
                    axis=-1,
                )
        else:
            with torch.no_grad():
                real_actions = actions = player.get_exploration_action(
                    obs[None, ...].to(device) / 255 - 0.5, is_continuous
                )
                actions = torch.cat(actions, -1).cpu().numpy()
                if is_continuous:
                    real_actions = torch.cat(real_actions, -1).cpu().numpy()
                else:
                    real_actions = np.array([real_act.cpu().argmax() for real_act in real_actions])
        next_obs, rewards, dones, truncated, infos = env.step(real_actions.reshape(env.action_space.shape))
        dones = np.logical_or(dones, truncated)

        if (dones or truncated) and "episode" in infos:
            fabric.print(f"Rank-0: global_step={global_step}, reward_env_{0}={infos['episode']['r'][0]}")
            aggregator.update("Rewards/rew_avg", infos["episode"]["r"][0])
            aggregator.update("Game/ep_len_avg", infos["episode"]["l"][0])

        next_obs = torch.from_numpy(next_obs.copy()).view(args.num_envs, *observation_shape)
        actions = torch.from_numpy(actions).view(args.num_envs, -1).float()
        rewards = torch.tensor([rewards]).view(args.num_envs, -1).float()
        dones = torch.tensor([bool(dones)]).view(args.num_envs, -1).float()

        # next_obs becomes the new obs
        obs = next_obs

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["observations"] = obs
        step_data["rewards"] = clip_rewards_fn(rewards)
        rb.add(step_data[None, ...])

        if dones or truncated:
            obs = torch.from_numpy(env.reset(seed=args.seed)[0].copy()).view(
                args.num_envs, *observation_shape
            )  # [N_envs, N_obs]
            step_data["dones"] = torch.zeros(args.num_envs, 1)
            step_data["actions"] = torch.zeros(args.num_envs, np.sum(actions_dim))
            step_data["rewards"] = torch.zeros(args.num_envs, 1)
            step_data["observations"] = obs
            rb.add(step_data[None, ...])
            player.init_states()

        step_before_training -= 1

        # Train the agent
        if global_step >= learning_starts and step_before_training <= 0:
            fabric.barrier()
            local_data = rb.sample(
                args.per_rank_batch_size,
                sequence_length=args.per_rank_sequence_length,
                n_samples=args.gradient_steps,
            ).to(device)
            distributed_sampler = BatchSampler(range(local_data.shape[0]), batch_size=1, drop_last=False)
            for i in distributed_sampler:
                train(
                    fabric,
                    world_model,
                    actor_task,
                    critic_task,
                    world_optimizer,
                    actor_task_optimizer,
                    critic_task_optimizer,
                    local_data[i].view(args.per_rank_sequence_length, args.per_rank_batch_size),
                    aggregator,
                    args,
                    ensembles=ensembles,
                    ensemble_optimizer=ensemble_optimizer,
                    actor_exploration=actor_exploration,
                    critic_exploration=critic_exploration,
                    actor_exploration_optimizer=actor_exploration_optimizer,
                    critic_exploration_optimizer=critic_exploration_optimizer,
                    is_exploring=is_exploring,
                )
            step_before_training = args.train_every // (args.num_envs * fabric.world_size * args.action_repeat)
            if args.expl_decay:
                expl_decay_steps += 1
                player.expl_amount = polynomial_decay(
                    expl_decay_steps,
                    initial=args.expl_amount,
                    final=args.expl_min,
                    max_decay_steps=max_step_expl_decay,
                )
            aggregator.update("Params/exploration_amout", player.expl_amount)
        aggregator.update("Time/step_per_second", int(global_step / (time.perf_counter() - start_time)))
        fabric.log_dict(aggregator.compute(), global_step)
        aggregator.reset()

        # Checkpoint Model
        if (
            (args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0)
            or args.dry_run
            or global_step == num_updates
        ):
            state = {
                "world_model": world_model.state_dict(),
                "actor_task": actor_task.state_dict(),
                "critic_task": critic_task.state_dict(),
                "ensembles": ensembles.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_task_optimizer": actor_task_optimizer.state_dict(),
                "critic_task_optimizer": critic_task_optimizer.state_dict(),
                "ensemble_optimizer": ensemble_optimizer.state_dict(),
                "expl_decay_steps": expl_decay_steps,
                "args": asdict(args),
                "global_step": global_step * fabric.world_size,
                "batch_size": args.per_rank_batch_size * fabric.world_size,
                "actor_exploration": actor_exploration.state_dict(),
                "critic_exploration": critic_exploration.state_dict(),
                "actor_exploration_optimizer": actor_exploration_optimizer.state_dict(),
                "critic_exploration_optimizer": critic_exploration_optimizer.state_dict(),
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{global_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if args.checkpoint_buffer else None,
            )

    env.close()
    # task test few-shot
    if fabric.is_global_zero:
        player.actor = actor_task.module
        test(player, fabric, args, "few-shot")


if __name__ == "__main__":
    main()
