import copy
import os
import pathlib
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.fabric import _is_using_cli
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.plugins.collectives import TorchCollective
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch.distributions import Bernoulli, Independent, Normal
from torch.optim import Adam
from torch.utils.data import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.dreamer_v1.agent import Player, WorldModel, build_models
from sheeprl.algos.dreamer_v1.args import DreamerV1Args
from sheeprl.algos.dreamer_v1.loss import actor_loss, critic_loss, reconstruction_loss
from sheeprl.algos.dreamer_v2.utils import test
from sheeprl.data.buffers import AsyncReplayBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import compute_lambda_values, make_dict_env, polynomial_decay

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    world_optimizer: _FabricOptimizer,
    actor_optimizer: _FabricOptimizer,
    critic_optimizer: _FabricOptimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: DreamerV1Args,
    cnn_keys: Sequence[str],
    mlp_keys: Sequence[str],
) -> None:
    """Runs one-step update of the agent.

    The follwing designations are used:
        - recurrent_state: is what is called ht or deterministic state from Figure 2c in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
        - stochastic_state: is what is called st or stochastic state from Figure 2c in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).
            It can be both posterior or prior.
        - latent state: the concatenation of the stochastic and recurrent states on the last dimension.
        - p: the output of the representation model, from Eq. 9 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - q: the output of the transition model, from Eq. 9 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - qo: the output of the observation model, from Eq. 9 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - qr: the output of the reward model, from Eq. 9 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).
        - qc: the output of the continue model.
        - qv: the output of the value model (critic), from Eq. 2 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    In particular, it updates the agent as specified by Algorithm 1 in
    [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603).

    1. Dynamic Learning:
        - Encoder: encode the observations.
        - Recurrent Model: compute the recurrent state from the previous recurrent state,
            the previous stochastic state, and from the previous actions.
        - Transition Model: predict the posterior state from the recurrent state, i.e., the deterministic state or ht.
        - Representation Model: compute the posterior state from the recurrent state and
            from the embedded observations provided by the environment.
        - Observation Model: reconstructs observations from latent states.
        - Reward Model: estimate rewards from the latent states.
        - Update the models
    2. Behaviour Learning:
        - Imagine trajectories in the latent space from each latent state s_t up to the horizon H: s'_(t+1), ..., s'_(t+H).
        - Predict rewards and values in the imagined trajectories.
        - Compute lambda targets (Eq. 6 in [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603))
        - Update the actor and the critic

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        actor (_FabricModule): the actor model wrapped with Fabric.
        critic (_FabricModule): the critic model wrapped with Fabric.
        world_optimizer (_FabricOptimizer): the world optimizer.
        actor_optimizer (_FabricOptimizer): the actor optimizer.
        critic_optimizer (_FabricOptimizer): the critic optimizer.
        data (TensorDictBase): the batch of data to use for training.
        aggregator (MetricAggregator): the aggregator to print the metrics.
        args (DreamerV1Args): the configs.
    """
    batch_size = args.per_rank_batch_size
    sequence_length = args.per_rank_sequence_length
    device = fabric.device
    batch_obs = {k: data[k] / 255 - 0.5 for k in cnn_keys}
    batch_obs.update({k: data[k] for k in mlp_keys})

    # Dynamic Learning
    # initialize the recurrent_state that must be a tuple of tensors (one for GRU or RNN).
    # the dimension of each vector must be (1, batch_size, recurrent_state_size)
    # the recurrent state is the deterministic state (or ht) from the Figure 2c in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
    recurrent_state = torch.zeros(1, batch_size, args.recurrent_state_size, device=device)

    # initialize the posterior that must be of dimension (batch_size, 1, stochastic_size)
    # the stochastic state is the stochastic state (or st) from the Figure 2c in [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)
    posterior = torch.zeros(1, batch_size, args.stochastic_size, device=device)

    # initialize the tensors for dynamic learning
    # recurrent_states will contain all the recurrent states computed during the dynamic learning phase,
    # and its dimension is (sequence_length, batch_size, recurrent_state_size)
    recurrent_states = torch.empty(sequence_length, batch_size, args.recurrent_state_size, device=device)
    # posteriors will contain all the posterior states computed during the dynamic learning phase,
    # and its dimension is (sequence_length, batch_size, stochastic_size)
    posteriors = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)

    # posteriors_mean and posteriors_std will contain all the actual means and stds of the posterior states respectively,
    # their dimension is (sequence_length, batch_size, stochastic_size)
    posteriors_mean = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    posteriors_std = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)

    # priors_mean and priors_std will contain all the predicted means and stds of the prior states respectively,
    # their dimension is (sequence_length, batch_size, stochastic_size)
    priors_mean = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)
    priors_std = torch.empty(sequence_length, batch_size, args.stochastic_size, device=device)

    embedded_obs = world_model.encoder(batch_obs)

    for i in range(0, sequence_length):
        # one step of dynamic learning, take the posterior state, the recurrent state, the action, and the observation
        # compute the mean and std of both the posterior and prior state, the new recurrent state
        # and the new posterior state
        recurrent_state, posterior, _, posterior_mean_std, prior_state_mean_std = world_model.rssm.dynamic(
            posterior, recurrent_state, data["actions"][i : i + 1], embedded_obs[i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        posteriors[i] = posterior
        posteriors_mean[i] = posterior_mean_std[0]
        posteriors_std[i] = posterior_mean_std[1]
        priors_mean[i] = prior_state_mean_std[0]
        priors_std[i] = prior_state_mean_std[1]

    # concatenate the posterior states with the recurrent states on the last dimension
    # latent_states tensor has dimension (sequence_length, batch_size, recurrent_state_size + stochastic_size)
    latent_states = torch.cat((posteriors, recurrent_states), -1)

    # compute predictions for the observations
    decoded_information: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)
    # compute the distribution of the reconstructed observations
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size observations.shape
    qo = {k: Independent(Normal(rec_obs, 1), len(rec_obs.shape[2:])) for k, rec_obs in decoded_information.items()}

    # compute predictions for the rewards
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the number of rewards
    qr = Independent(Normal(world_model.reward_model(latent_states), 1), 1)

    # compute predictions for terminal steps, if required
    if args.use_continues and world_model.continue_model:
        qc = Independent(Bernoulli(logits=world_model.continue_model(latent_states), validate_args=False), 1)
        continue_targets = (1 - data["dones"]) * args.gamma
    else:
        qc = continue_targets = None

    # compute the distributions of the states (posteriors and priors)
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the stochastic size
    p = Independent(Normal(posteriors_mean, posteriors_std), 1)
    q = Independent(Normal(priors_mean, priors_std), 1)

    # world model optimization step
    world_optimizer.zero_grad(set_to_none=True)
    # compute the overall loss of the world model
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
        world_model_grads = fabric.clip_gradients(
            module=world_model, optimizer=world_optimizer, max_norm=args.clip_gradients
        )
    world_optimizer.step()
    aggregator.update("Grads/world_model", world_model_grads.mean().detach())
    aggregator.update("Loss/reconstruction_loss", rec_loss.detach())
    aggregator.update("Loss/observation_loss", observation_loss.detach())
    aggregator.update("Loss/reward_loss", reward_loss.detach())
    aggregator.update("Loss/state_loss", state_loss.detach())
    aggregator.update("Loss/continue_loss", continue_loss.detach())
    aggregator.update("State/kl", kl.detach())
    aggregator.update("State/p_entropy", p.entropy().mean().detach())
    aggregator.update("State/q_entropy", q.entropy().mean().detach())

    # Behaviour Learning
    # unflatten first 2 dimensions of recurrent and posterior states in order to have all the states on the first dimension.
    # The 1 in the second dimension is needed for the recurrent model in the imagination step,
    # 1 because the agent imagines one state at a time.
    # (1, batch_size * sequence_length, stochastic_size)
    imagined_prior = posteriors.detach().reshape(1, -1, args.stochastic_size)

    # initialize the recurrent state of the recurrent model with the recurrent states computed
    # during the dynamic learning phase, its shape is (1, batch_size * sequence_length, recurrent_state_size).
    recurrent_state = recurrent_states.detach().reshape(1, -1, args.recurrent_state_size)

    # (1, batch_size * sequence_length, determinisitic_size + stochastic_size)
    imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)

    # initialize the tensor of the imagined states
    imagined_trajectories = torch.empty(
        args.horizon, batch_size * sequence_length, args.stochastic_size + args.recurrent_state_size, device=device
    )

    # imagine trajectories in the latent space
    for i in range(args.horizon):
        # actions tensor has dimension (1, batch_size * sequence_length, num_actions)
        actions = torch.cat(actor(imagined_latent_states.detach())[0], dim=-1)

        # imagination step
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)

        # update current state
        imagined_latent_states = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_states

    # predict values and rewards
    # it is necessary an Independent distribution because
    # it is necessary to create (batch_size * sequence_length) independent distributions,
    # each producing a sample of size equal to the number of values/rewards
    predicted_values = Independent(Normal(critic(imagined_trajectories), 1), 1).mean
    predicted_rewards = Independent(Normal(world_model.reward_model(imagined_trajectories), 1), 1).mean

    # predict the probability that the episode will continue in the imagined states
    if args.use_continues and world_model.continue_model:
        predicted_continues = Independent(Bernoulli(logits=world_model.continue_model(imagined_trajectories)), 1).mean
    else:
        predicted_continues = torch.ones_like(predicted_rewards.detach()) * args.gamma

    # compute the lambda_values, by passing as last values the values of the last imagined state
    # the dimensions of the lambda_values tensor are
    # (horizon, batch_size * sequence_length, recurrent_state_size + stochastic_size)
    lambda_values = compute_lambda_values(
        predicted_rewards,
        predicted_values,
        predicted_continues,
        last_values=predicted_values[-1],
        horizon=args.horizon,
        lmbda=args.lmbda,
    )

    # compute the discounts to multiply to the lambda values
    with torch.no_grad():
        # the time steps in Eq. 7 and Eq. 8 of the paper are weighted by the cumulative product of the predicted
        # discount factors, estimated by the continue model, so terms are wighted down based on how likely
        # the imagined trajectory would have ended.
        # Ref. subsection "Learning objectives" of paragraph 3 (Learning Behaviors by Latent Imagination)
        # in [https://doi.org/10.48550/arXiv.1912.01603](https://doi.org/10.48550/arXiv.1912.01603)
        #
        # Suppose the case in which the continue model is not used and gamma = .99
        # predicted_continues.shape = (15, 2500, 1)
        # predicted_continues = [
        #   [ [.99], ..., [.99] ], (2500 columns)
        #   ...
        # ] (15 rows)
        # torch.ones_like(predicted_continues[:1]) = [
        #   [ [1.], ..., [1.] ]
        # ] (1 row and 2500 columns), the discount of the time step 0 is 1.
        # predicted_continues[:-2] = [
        #   [ [.99], ..., [.99] ], (2500 columns)
        #   ...
        # ] (13 rows)
        # torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0) = [
        #   [ [1.], ..., [1.] ], (2500 columns)
        #   [ [.99], ..., [.99] ],
        #   ...,
        #   [ [.99], ..., [.99] ],
        # ] (14 rows), the total number of imagined steps is 15, but one is lost because of the values computation
        # torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0) = [
        #   [ [1.], ..., [1.] ], (2500 columns)
        #   [ [.99], ..., [.99] ],
        #   [ [.9801], ..., [.9801] ],
        #   ...,
        #   [ [.8775], ..., [.8775] ],
        # ] (14 rows)
        discount = torch.cumprod(torch.cat((torch.ones_like(predicted_continues[:1]), predicted_continues[:-2]), 0), 0)

    # actor optimization step
    actor_optimizer.zero_grad(set_to_none=True)
    # compute the policy loss
    policy_loss = actor_loss(discount * lambda_values)
    fabric.backward(policy_loss)
    if args.clip_gradients is not None and args.clip_gradients > 0:
        actor_grads = fabric.clip_gradients(module=actor, optimizer=actor_optimizer, max_norm=args.clip_gradients)
    actor_optimizer.step()
    aggregator.update("Grads/actor", actor_grads.mean().detach())
    aggregator.update("Loss/policy_loss", policy_loss.detach())

    # predict the values distribution only for the first H (horizon) imagined states (to match the dimension with the lambda values),
    # it removes the last imagined state in the trajectory because it is used only for compuing correclty the lambda values
    qv = Independent(Normal(critic(imagined_trajectories.detach())[:-1], 1), 1)

    # critic optimization step
    critic_optimizer.zero_grad(set_to_none=True)
    # compute the value loss
    # the discount has shape (horizon, seuqence_length * batch_size, 1), so,
    # it is necessary to remove the last dimension properly match the shapes
    # for the log prob
    value_loss = critic_loss(qv, lambda_values.detach(), discount[..., 0])
    fabric.backward(value_loss)
    if args.clip_gradients is not None and args.clip_gradients > 0:
        critic_grads = fabric.clip_gradients(module=critic, optimizer=critic_optimizer, max_norm=args.clip_gradients)
    critic_optimizer.step()
    aggregator.update("Grads/critic", critic_grads.mean().detach())
    aggregator.update("Loss/value_loss", value_loss.detach())

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main():
    parser = HfArgumentParser(DreamerV1Args)
    args: DreamerV1Args = parser.parse_args_into_dataclasses()[0]
    args.frame_stack = -1
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
        args = DreamerV1Args(**state["args"])
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
            else os.path.join("logs", "dreamer_v1", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
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

    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = (
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    # observation_shape = observation_space["rgb"].shape
    clip_rewards_fn = lambda r: torch.tanh(r) if args.clip_rewards else r
    cnn_keys = []
    mlp_keys = []
    if isinstance(observation_space, gym.spaces.Dict):
        cnn_keys = []
        for k, v in observation_space.spaces.items():
            if args.cnn_keys and k in args.cnn_keys:
                if len(v.shape) == 3:
                    cnn_keys.append(k)
                else:
                    fabric.print(
                        f"Found a CNN key which is not an image: `{k}` of shape {v.shape}. "
                        "Try to transform the observation from the environment into a 3D image"
                    )
        mlp_keys = []
        for k, v in observation_space.spaces.items():
            if args.mlp_keys and k in args.mlp_keys:
                if len(v.shape) == 1:
                    mlp_keys.append(k)
                else:
                    fabric.print(
                        f"Found an MLP key which is not a vector: `{k}` of shape {v.shape}. "
                        "Try to flatten the observation from the environment"
                    )
    else:
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {type(observation_space)}")
    if cnn_keys == [] and mlp_keys == []:
        raise RuntimeError(
            "You should specify at least one CNN keys or MLP keys from the cli: `--cnn_keys rgb` or `--mlp_keys state` "
        )
    fabric.print("CNN keys:", cnn_keys)
    fabric.print("MLP keys:", mlp_keys)

    world_model, actor, critic = build_models(
        fabric,
        actions_dim,
        is_continuous,
        args,
        observation_space,
        cnn_keys,
        mlp_keys,
        state["world_model"] if args.checkpoint_path else None,
        state["actor"] if args.checkpoint_path else None,
        state["critic"] if args.checkpoint_path else None,
    )
    player = Player(
        world_model.encoder.module,
        world_model.rssm.recurrent_model.module,
        world_model.rssm.representation_model.module,
        actor.module,
        actions_dim,
        args.expl_amount,
        args.num_envs,
        args.stochastic_size,
        args.recurrent_state_size,
        fabric.device,
    )

    # Optimizers
    world_optimizer = Adam(world_model.parameters(), lr=args.world_lr)
    actor_optimizer = Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = Adam(critic.parameters(), lr=args.critic_lr)
    if args.checkpoint_path:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_optimizer, critic_optimizer
    )

    # Metrics
    with device:
        aggregator = MetricAggregator(
            {
                "Rewards/rew_avg": MeanMetric(sync_on_compute=False),
                "Game/ep_len_avg": MeanMetric(sync_on_compute=False),
                "Time/step_per_second": MeanMetric(sync_on_compute=False),
                "Loss/reconstruction_loss": MeanMetric(sync_on_compute=False),
                "Loss/value_loss": MeanMetric(sync_on_compute=False),
                "Loss/policy_loss": MeanMetric(sync_on_compute=False),
                "Loss/observation_loss": MeanMetric(sync_on_compute=False),
                "Loss/reward_loss": MeanMetric(sync_on_compute=False),
                "Loss/state_loss": MeanMetric(sync_on_compute=False),
                "Loss/continue_loss": MeanMetric(sync_on_compute=False),
                "State/p_entropy": MeanMetric(sync_on_compute=False),
                "State/q_entropy": MeanMetric(sync_on_compute=False),
                "State/kl": MeanMetric(sync_on_compute=False),
                "Params/exploration_amout": MeanMetric(sync_on_compute=False),
                "Grads/world_model": MeanMetric(sync_on_compute=False),
                "Grads/actor": MeanMetric(sync_on_compute=False),
                "Grads/critic": MeanMetric(sync_on_compute=False),
            }
        )
    aggregator.to(fabric.device)

    # Local data
    buffer_size = args.buffer_size // int(args.num_envs * fabric.world_size) if not args.dry_run else 2
    rb = AsyncReplayBuffer(
        buffer_size,
        args.num_envs,
        device=fabric.device if args.memmap_buffer else "cpu",
        memmap=args.memmap_buffer,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        sequential=True,
    )
    if args.checkpoint_path and args.checkpoint_buffer:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], AsyncReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")
    step_data = TensorDict({}, batch_size=[args.num_envs], device=fabric.device if args.memmap_buffer else "cpu")
    expl_decay_steps = state["expl_decay_steps"] if args.checkpoint_path else 0

    # Global variables
    start_time = time.perf_counter()
    start_step = state["global_step"] // fabric.world_size if args.checkpoint_path else 1
    single_global_step = int(args.num_envs * fabric.world_size * args.action_repeat)
    step_before_training = args.train_every // single_global_step if not args.dry_run else 0
    num_updates = int(args.total_steps // single_global_step) if not args.dry_run else 1
    learning_starts = (args.learning_starts // single_global_step) if not args.dry_run else 0
    if args.checkpoint_path and not args.checkpoint_buffer:
        learning_starts = start_step + args.learning_starts // single_global_step
    max_step_expl_decay = args.max_step_expl_decay // (args.gradient_steps * fabric.world_size)
    if args.checkpoint_path:
        player.expl_amount = polynomial_decay(
            expl_decay_steps,
            initial=args.expl_amount,
            final=args.expl_min,
            max_decay_steps=max_step_expl_decay,
        )

    # Get the first environment observation and start the optimization
    o = envs.reset(seed=args.seed)[0]
    obs = {}
    for k in o.keys():
        torch_obs = torch.from_numpy(o[k]).view(args.num_envs, *o[k].shape[1:])
        if k in mlp_keys:
            torch_obs = torch_obs.float()
        step_data[k] = torch_obs
        obs[k] = torch_obs
    step_data["dones"] = torch.zeros(args.num_envs, 1)
    step_data["actions"] = torch.zeros(args.num_envs, sum(actions_dim))
    step_data["rewards"] = torch.zeros(args.num_envs, 1)
    rb.add(step_data[None, ...])
    player.init_states()

    for global_step in range(start_step, num_updates + 1):
        # Sample an action given the observation received by the environment
        if global_step <= learning_starts and args.checkpoint_path is None and "minedojo" not in args.env_id:
            real_actions = actions = np.array(envs.action_space.sample())
            if not is_continuous:
                actions = np.concatenate(
                    [
                        F.one_hot(torch.tensor(act), act_dim).numpy()
                        for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                    ],
                    axis=-1,
                )
        else:
            with torch.no_grad():
                preprocessed_obs = {}
                for k, v in obs.items():
                    if k in cnn_keys:
                        preprocessed_obs[k] = v[None, ...].to(device) / 255 - 0.5
                    else:
                        preprocessed_obs[k] = v[None, ...].to(device)
                mask = {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
                if len(mask) == 0:
                    mask = None
                real_actions = actions = player.get_exploration_action(preprocessed_obs, is_continuous, mask)
                actions = torch.cat(actions, -1).cpu().numpy()
                if is_continuous:
                    real_actions = torch.cat(real_actions, -1).cpu().numpy()
                else:
                    real_actions = np.array([real_act.cpu().argmax(dim=-1).numpy() for real_act in real_actions])
        o, rewards, dones, truncated, infos = envs.step(real_actions.reshape(envs.action_space.shape))
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
            next_obs[k] = torch.from_numpy(o[k]).view(args.num_envs, *o[k].shape[1:])
            step_data[k] = torch.from_numpy(real_next_obs[k]).view(args.num_envs, *real_next_obs[k].shape[1:])
            if k in mlp_keys:
                next_obs[k] = next_obs[k].float()
                step_data[k] = step_data[k].float()
        actions = torch.from_numpy(actions).view(args.num_envs, -1).float()
        rewards = torch.from_numpy(rewards).view(args.num_envs, -1).float()
        dones = torch.from_numpy(dones).view(args.num_envs, -1).float()

        # next_obs becomes the new obs
        obs = next_obs

        step_data["dones"] = dones
        step_data["actions"] = actions
        step_data["rewards"] = clip_rewards_fn(rewards)
        rb.add(step_data[None, ...])

        # Reset and save the observation coming from the automatic reset
        dones_idxes = dones.nonzero(as_tuple=True)[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            reset_data = TensorDict({}, batch_size=[reset_envs], device="cpu")
            for k in next_obs.keys():
                reset_data[k] = next_obs[k][dones_idxes]
            reset_data["dones"] = torch.zeros(reset_envs, 1)
            reset_data["actions"] = torch.zeros(reset_envs, np.sum(actions_dim))
            reset_data["rewards"] = torch.zeros(reset_envs, 1)
            rb.add(reset_data[None, ...], dones_idxes)
            # Reset dones so that `is_first` is updated
            for d in dones_idxes:
                step_data["dones"][d] = torch.zeros_like(step_data["dones"][d])
            # Reset internal agent states
            player.init_states(dones_idxes)

        step_before_training -= 1

        # Train the agent
        if global_step > learning_starts and step_before_training <= 0:
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
                    actor,
                    critic,
                    world_optimizer,
                    actor_optimizer,
                    critic_optimizer,
                    local_data[i].view(args.per_rank_sequence_length, args.per_rank_batch_size),
                    aggregator,
                    args,
                    cnn_keys,
                    mlp_keys,
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
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "expl_decay_steps": expl_decay_steps,
                "args": asdict(args),
                "global_step": global_step * fabric.world_size,
                "batch_size": args.per_rank_batch_size * fabric.world_size,
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{global_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if args.checkpoint_buffer else None,
            )

    envs.close()
    if fabric.is_global_zero:
        test(player, fabric, args, cnn_keys, mlp_keys)


if __name__ == "__main__":
    main()
