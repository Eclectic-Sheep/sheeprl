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
from lightning.fabric.wrappers import _FabricModule
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import Tensor
from torch.distributions import Bernoulli, Distribution, Independent, OneHotCategorical
from torch.optim import Adam, Optimizer
from torch.utils.data import BatchSampler
from torchmetrics import MeanMetric

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import compute_lambda_values, make_env, test
from sheeprl.algos.dreamer_v3.agent import PlayerDV3, build_models
from sheeprl.algos.dreamer_v3.args import DreamerV3Args
from sheeprl.algos.dreamer_v3.loss import reconstruction_loss
from sheeprl.algos.dreamer_v3.utils import Moments
from sheeprl.data.buffers import AsyncReplayBuffer, EpisodeBuffer
from sheeprl.utils.callback import CheckpointCallback
from sheeprl.utils.distribution import DiscDist, MSEDist, SymlogDist
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.parser import HfArgumentParser
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.utils import polynomial_decay

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"
torch.set_float32_matmul_precision("medium" or "high")


def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    target_critic: torch.nn.Module,
    world_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    data: TensorDictBase,
    aggregator: MetricAggregator,
    args: DreamerV3Args,
    is_continuous: bool,
    cnn_keys: Sequence[str],
    mlp_keys: Sequence[str],
    actions_dim: Sequence[int],
    moments: Moments,
) -> None:
    """Runs one-step update of the agent.

    The follwing designations are used:
        - recurrent_state: is what is called ht or deterministic state from Figure 2 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - prior: the stochastic state coming out from the transition model, depicted as z-hat_t
        in Figure 2 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - posterior: the stochastic state coming out from the representation model, depicted as z_t
        in Figure 2 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - latent state: the concatenation of the stochastic (can be both the prior or the posterior one) and recurrent states on the last dimension.
        - p: the output of the transition model, from Eq. 1 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - q: the output of the representation model, from Eq. 1 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - po: the output of the observation model, from Eq. 1 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - pr: the output of the reward model, from Eq. 1 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - pc: the output of the continue model (discout predictor), from Eq. 1 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).
        - pv: the output of the value model (critic), from Eq. 3 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

    In particular, it updates the agent as specified by Algorithm 1 in
    [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193).

    1. Dynamic Learning:
        - Encoder: encode the observations.
        - Recurrent Model: compute the recurrent state from the previous recurrent state,
            the previous stochastic state, and from the previous actions.
        - Transition Model: predict the stochastic state from the recurrent state, i.e., the deterministic state or ht.
        - Representation Model: compute the actual stochastic state from the recurrent state and
            from the embedded observations provided by the environment.
        - Observation Model: reconstructs observations from latent states.
        - Reward Model: estimate rewards from the latent states.
        - Update the models
    2. Behaviour Learning:
        - Imagine trajectories in the latent space from each latent state s_t up to the horizon H: s'_(t+1), ..., s'_(t+H).
        - Predict rewards and values in the imagined trajectories.
        - Compute lambda targets (Eq. 4 in [https://arxiv.org/abs/2010.02193](https://arxiv.org/abs/2010.02193))
        - Update the actor and the critic

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        actor (_FabricModule): the actor model wrapped with Fabric.
        critic (_FabricModule): the critic model wrapped with Fabric.
        world_optimizer (Optimizer): the world optimizer.
        actor_optimizer (Optimizer): the actor optimizer.
        critic_optimizer (Optimizer): the critic optimizer.
        data (TensorDictBase): the batch of data to use for training.
        aggregator (MetricAggregator): the aggregator to print the metrics.
        args (DreamerV3Args): the configs.
        is_continuous (bool): whether the action space is a continuous or discrete space
    """
    batch_size = args.per_rank_batch_size
    sequence_length = args.per_rank_sequence_length
    device = fabric.device
    batch_obs = {k: data[k] / 255.0 for k in cnn_keys}
    batch_obs.update({k: data[k] for k in mlp_keys})
    data["is_first"][0, :] = torch.tensor([1.0], device=fabric.device).expand_as(data["is_first"][0, :])
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    # initialize the recurrent_state that must be a tuple of tensors (one for GRU or RNN).
    # the dimension of each vector must be (1, batch_size, recurrent_state_size)
    # the recurrent state is the deterministic state (or ht)
    recurrent_state = torch.zeros(1, batch_size, args.recurrent_state_size, device=device)

    # initialize the posterior that must be of dimension (1, batch_size, stochastic_size, discrete_size), which
    # by default is set to (1, batch_size, 32, 32). The posterior state is named zt in the paper
    _, posterior = world_model.rssm._transition(recurrent_state)

    # initialize the recurrent_states, which will contain all the recurrent states
    # computed during the dynamic learning phase. Its dimension is (sequence_length, batch_size, recurrent_state_size)
    recurrent_states = torch.zeros(sequence_length, batch_size, args.recurrent_state_size, device=device)

    # initialize all the tensor to collect priors and posteriors states with their associated logits
    priors_logits = torch.empty(sequence_length, batch_size, args.stochastic_size * args.discrete_size, device=device)
    posteriors = torch.empty(sequence_length, batch_size, args.stochastic_size, args.discrete_size, device=device)
    posteriors_logits = torch.empty(
        sequence_length, batch_size, args.stochastic_size * args.discrete_size, device=device
    )

    # embedded observations from the environment
    embedded_obs = world_model.encoder(batch_obs)

    for i in range(0, sequence_length):
        # one step of dynamic learning, which take the posterior state, the recurrent state, the action
        # and the observation and compute the next recurrent, prior and posterior states
        recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
            posterior, recurrent_state, batch_actions[i : i + 1], embedded_obs[i : i + 1], data["is_first"][i : i + 1]
        )
        recurrent_states[i] = recurrent_state
        priors_logits[i] = prior_logits
        posteriors[i] = posterior
        posteriors_logits[i] = posterior_logits

    # concatenate the posteriors with the recurrent states on the last dimension
    # latent_states has dimension (sequence_length, batch_size, recurrent_state_size + stochastic_size * discrete_size)
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)

    # compute predictions for the observations
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states)

    # compute the distribution over the reconstructed observations
    po = {k: MSEDist(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:])) for k in cnn_keys}
    po.update({k: SymlogDist(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:])) for k in mlp_keys})

    # compute the distribution over the rewards
    with fabric.device:
        pr = DiscDist(world_model.reward_model(latent_states), dims=1)

    # compute the distribution over the terminal steps, if required
    pc = Independent(Bernoulli(logits=world_model.continue_model(latent_states), validate_args=False), 1)
    continue_targets = (1 - data["dones"]) * args.gamma

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], args.stochastic_size, args.discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], args.stochastic_size, args.discrete_size)

    # world model optimization step
    world_optimizer.zero_grad(set_to_none=True)
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        args.kl_dynamic,
        args.kl_representation,
        args.kl_free_nats,
        args.kl_regularizer,
        pc,
        continue_targets,
        args.continue_scale_factor,
    )
    fabric.backward(rec_loss)
    if args.world_clip_gradients is not None and args.world_clip_gradients > 0:
        world_model_grads = fabric.clip_gradients(
            module=world_model, optimizer=world_optimizer, max_norm=args.world_clip_gradients, error_if_nonfinite=False
        )
    world_optimizer.step()
    aggregator.update("Grads/world_model", world_model_grads.mean().detach())
    aggregator.update("Loss/reconstruction_loss", rec_loss.detach())
    aggregator.update("Loss/observation_loss", observation_loss.detach())
    aggregator.update("Loss/reward_loss", reward_loss.detach())
    aggregator.update("Loss/state_loss", state_loss.detach())
    aggregator.update("Loss/continue_loss", continue_loss.detach())
    aggregator.update("State/kl", kl.mean().detach())
    aggregator.update(
        "State/p_entropy",
        Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
    )
    aggregator.update(
        "State/q_entropy",
        Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
    )

    # Behaviour Learning
    # unflatten first 2 dimensions of recurrent and stochastic states in order to have all the states on the first dimension.
    # The 1 in the second dimension is needed for the recurrent model in the imagination step,
    # 1 because the agent imagines one state at a time.
    # (1, batch_size * sequence_length, stochastic_size)
    imagined_prior = posteriors.detach().reshape(1, -1, args.stochastic_size * args.discrete_size)

    # initialize the recurrent state of the recurrent model with the recurrent states computed
    # during the dynamic learning phase, its shape is (1, batch_size * sequence_length, recurrent_state_size).
    recurrent_state = recurrent_states.detach().reshape(1, -1, args.recurrent_state_size)

    # (1, batch_size * sequence_length, determinisitic_size + stochastic_size * discrete_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)

    # initialize the tensor of the imagined trajectories
    imagined_trajectories = torch.empty(
        args.horizon + 1,
        batch_size * sequence_length,
        args.stochastic_size * args.discrete_size + args.recurrent_state_size,
        device=device,
    )
    imagined_trajectories[0] = imagined_latent_state

    # initialize the tensor of the imagined actions
    imagined_actions = torch.empty(
        args.horizon + 1,
        batch_size * sequence_length,
        data["actions"].shape[-1],
        device=device,
    )
    actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    imagined_actions[0] = actions

    # imagine trajectories in the latent space
    for i in range(1, args.horizon + 1):
        # imagination step
        imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)

        # update current state
        imagined_prior = imagined_prior.view(1, -1, args.stochastic_size * args.discrete_size)
        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state

        # actions tensor has dimension (1, batch_size * sequence_length, num_actions)
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions

    # predict values and rewards
    with fabric.device:
        predicted_values = DiscDist(critic(imagined_trajectories), dims=1).mean
        predicted_rewards = DiscDist(world_model.reward_model(imagined_trajectories), dims=1).mean
    continues = Independent(Bernoulli(logits=world_model.continue_model(imagined_trajectories)), 1).mode
    true_done = (1 - data["dones"]).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_done, continues[1:]))

    # compute the lambda_values, by passing as last values the values of the last imagined state
    # the dimensions of the lambda_values tensor are
    # (horizon, batch_size * sequence_length, recurrent_state_size + stochastic_size)
    lambda_values = compute_lambda_values(
        predicted_rewards[1:],
        predicted_values[1:],
        continues[1:] * args.gamma,
        bootstrap=predicted_values[-1:],
        horizon=args.horizon,
        lmbda=args.lmbda,
    )

    # compute the discounts to multiply to the lambda values
    with torch.no_grad():
        # the losses in Eq. 5 and Eq. 6 of the paper are weighted by the cumulative product of the predicted
        # discount factors, estimated by the continue model, so terms are wighted down based on how likely
        # the imagined trajectory would have ended.
        #
        # Suppose the case in which the continue model is not used and gamma = .99
        # continues.shape = (15, 2500, 1)
        # continues = [
        #   [ [.99], ..., [.99] ], (2500 columns)
        #   ...
        # ] (15 rows)
        # torch.ones_like(continues[:1]) = [
        #   [ [1.], ..., [1.] ]
        # ] (1 row and 2500 columns), the discount of the time step 0 is 1.
        # continues[:-2] = [
        #   [ [.99], ..., [.99] ], (2500 columns)
        #   ...
        # ] (13 rows)
        # torch.cat((torch.ones_like(continues[:1]), continues[:-2]), 0) = [
        #   [ [1.], ..., [1.] ], (2500 columns)
        #   [ [.99], ..., [.99] ],
        #   ...,
        #   [ [.99], ..., [.99] ],
        # ] (14 rows), the total number of imagined steps is 15, but one is lost because of the values computation
        # torch.cumprod(torch.cat((torch.ones_like(continues[:1]), continues[:-2]), 0), 0) = [
        #   [ [1.], ..., [1.] ], (2500 columns)
        #   [ [.99], ..., [.99] ],
        #   [ [.9801], ..., [.9801] ],
        #   ...,
        #   [ [.8775], ..., [.8775] ],
        # ] (14 rows)
        discount = torch.cumprod(continues * args.gamma, dim=0) / args.gamma

    # actor optimization step. Eq. 6 from the paper
    actor_optimizer.zero_grad(set_to_none=True)
    policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]

    baseline = predicted_values[:-1]
    offset, invscale = moments(lambda_values)
    normed_lambda_values = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda_values - normed_baseline
    if is_continuous:
        objective = advantage
    else:
        objective = (
            torch.stack(
                [
                    p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                    for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                ],
                -1,
            ).sum(-1)
            * advantage.detach()
        )
    try:
        entropy = args.actor_ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
    except NotImplementedError:
        entropy = torch.zeros_like(objective)
    policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
    fabric.backward(policy_loss)
    if args.actor_clip_gradients is not None and args.actor_clip_gradients > 0:
        actor_grads = fabric.clip_gradients(
            module=actor, optimizer=actor_optimizer, max_norm=args.actor_clip_gradients, error_if_nonfinite=False
        )
    actor_optimizer.step()
    aggregator.update("Grads/actor", actor_grads.mean().detach())
    aggregator.update("Loss/policy_loss", policy_loss.detach())

    # predict the values distribution only for the first H (horizon) imagined states (to match the dimension with the lambda values),
    # it removes the last imagined state in the trajectory because it is used only for compuing correclty the lambda values
    with fabric.device:
        qv = DiscDist(critic(imagined_trajectories.detach()[:-1]), 1)
        predicted_target_values = DiscDist(target_critic(imagined_trajectories[:-1])).mean

    # critic optimization step. Eq. 5 from the paper.
    critic_optimizer.zero_grad(set_to_none=True)
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

    fabric.backward(value_loss)
    if args.critic_clip_gradients is not None and args.critic_clip_gradients > 0:
        critic_grads = fabric.clip_gradients(
            module=critic, optimizer=critic_optimizer, max_norm=args.critic_clip_gradients, error_if_nonfinite=False
        )
    critic_optimizer.step()
    aggregator.update("Grads/critic", critic_grads.mean().detach())
    aggregator.update("Loss/value_loss", value_loss.detach())

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)


@register_algorithm()
def main():
    parser = HfArgumentParser(DreamerV3Args)
    args: DreamerV3Args = parser.parse_args_into_dataclasses()[0]
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
        args = DreamerV3Args(**state["args"])
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
            else os.path.join("logs", "dreamer_v3", datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
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
            make_env(
                args.env_id,
                args.seed + rank * args.num_envs,
                rank,
                args,
                logger.log_dir if rank == 0 else None,
                "train",
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
    clip_rewards_fn = lambda r: torch.tanh(r) if args.clip_rewards else r
    cnn_keys = []
    mlp_keys = []
    if isinstance(observation_space, gym.spaces.Dict):
        cnn_keys = []
        for k, v in observation_space.spaces.items():
            if args.cnn_keys and (
                k in args.cnn_keys or (len(args.cnn_keys) == 1 and args.cnn_keys[0].lower() == "all")
            ):
                if len(v.shape) == 3:
                    cnn_keys.append(k)
                else:
                    fabric.print(
                        f"Found a CNN key which is not an image: `{k}` of shape {v.shape}. "
                        "Try to transform the observation from the environment into a 3D image"
                    )
        mlp_keys = []
        for k, v in observation_space.spaces.items():
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
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if cnn_keys == [] and mlp_keys == []:
        raise RuntimeError(f"There must be at least one valid observation.")
    fabric.print("CNN keys:", cnn_keys)
    fabric.print("MLP keys:", mlp_keys)
    obs_keys = cnn_keys + mlp_keys

    world_model, actor, critic, target_critic = build_models(
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
    player = PlayerDV3(
        world_model.encoder.module,
        world_model.rssm,
        actor.module,
        actions_dim,
        args.expl_amount,
        args.num_envs,
        args.stochastic_size,
        args.recurrent_state_size,
        fabric.device,
        discrete_size=args.discrete_size,
    )

    # Optimizers
    world_optimizer = Adam(world_model.parameters(), lr=args.world_lr, weight_decay=0.0, eps=1e-8)
    actor_optimizer = Adam(actor.parameters(), lr=args.actor_lr, weight_decay=0.0, eps=1e-5)
    critic_optimizer = Adam(critic.parameters(), lr=args.critic_lr, weight_decay=0.0, eps=1e-5)
    if args.checkpoint_path:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_optimizer, critic_optimizer
    )
    moments = Moments(fabric, args.moments_decay, args.moment_max, args.moments_perclo, args.moments_perchi)

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
                "State/kl": MeanMetric(sync_on_compute=False),
                "State/p_entropy": MeanMetric(sync_on_compute=False),
                "State/q_entropy": MeanMetric(sync_on_compute=False),
                "Params/exploration_amout": MeanMetric(sync_on_compute=False),
                "Grads/world_model": MeanMetric(sync_on_compute=False),
                "Grads/actor": MeanMetric(sync_on_compute=False),
                "Grads/critic": MeanMetric(sync_on_compute=False),
            }
        )
        aggregator.to(fabric.device)

    # Local data
    buffer_size = (
        args.buffer_size // int(args.num_envs * fabric.world_size * args.action_repeat) if not args.dry_run else 2
    )
    buffer_type = args.buffer_type.lower()
    if buffer_type == "sequential":
        rb = AsyncReplayBuffer(
            buffer_size,
            args.num_envs,
            device="cpu",
            memmap=args.memmap_buffer,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
            sequential=True,
        )
    elif buffer_type == "episode":
        rb = EpisodeBuffer(
            buffer_size,
            sequence_length=args.per_rank_sequence_length,
            device="cpu",
            memmap=args.memmap_buffer,
            memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        )
    else:
        raise ValueError(f"Unrecognized buffer type: must be one of `sequential` or `episode`, received: {buffer_type}")
    if args.checkpoint_path and args.checkpoint_buffer:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], (AsyncReplayBuffer, EpisodeBuffer)):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")
    step_data = TensorDict({}, batch_size=[args.num_envs], device="cpu")
    expl_decay_steps = state["expl_decay_steps"] if args.checkpoint_path else 0

    # Global variables
    start_time = time.perf_counter()
    start_step = state["global_step"] // fabric.world_size if args.checkpoint_path else 1
    single_global_step = int(args.num_envs * fabric.world_size)
    step_before_training = args.train_every // single_global_step
    num_updates = int(args.total_steps // single_global_step) if not args.dry_run else 1
    learning_starts = args.learning_starts // single_global_step if not args.dry_run else 0
    if args.checkpoint_path and not args.checkpoint_buffer:
        learning_starts += start_step
    max_step_expl_decay = args.max_step_expl_decay // (args.gradient_steps * fabric.world_size)
    if args.checkpoint_path:
        player.expl_amount = polynomial_decay(
            expl_decay_steps,
            initial=args.expl_amount,
            final=args.expl_min,
            max_decay_steps=max_step_expl_decay,
        )

    # Get the first environment observation and start the optimization
    episode_steps = []
    o = envs.reset(seed=args.seed)[0]
    obs = {}
    for k in obs_keys:
        torch_obs = torch.from_numpy(o[k]).view(args.num_envs, *o[k].shape[1:])
        if k in mlp_keys:
            # Images stay uint8 to save space
            torch_obs = torch_obs.float()
        step_data[k] = torch_obs
        obs[k] = torch_obs
    step_data["dones"] = torch.zeros(args.num_envs, 1).float()
    step_data["rewards"] = torch.zeros(args.num_envs, 1).float()
    step_data["is_first"] = torch.ones_like(step_data["dones"]).float()
    player.init_states()

    gradient_steps = 0
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
                        preprocessed_obs[k] = v[None, ...].to(device) / 255.0
                    else:
                        preprocessed_obs[k] = v[None, ...].to(device)
                mask = {k: v for k, v in preprocessed_obs.items() if k.startswith("mask")}
                if len(mask) == 0:
                    mask = None
                real_actions = actions = player.get_exploration_action(preprocessed_obs, is_continuous, mask)
                actions = torch.cat(actions, -1).cpu().numpy()
                if is_continuous:
                    real_actions = torch.cat(real_actions, dim=-1).cpu().numpy()
                else:
                    real_actions = np.array([real_act.cpu().argmax(dim=-1) for real_act in real_actions])

        step_data["actions"] = torch.from_numpy(actions).view(args.num_envs, -1).float()
        data_to_add = step_data[None, ...]
        if buffer_type == "sequential":
            rb.add(data_to_add)
        else:
            episode_steps.append(data_to_add)

        o, rewards, dones, truncated, infos = envs.step(real_actions.reshape(envs.action_space.shape))
        dones = np.logical_or(dones, truncated)
        if args.dry_run and buffer_type == "episode":
            dones = np.ones_like(dones)

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

        next_obs: Dict[str, Tensor] = {}
        for k in real_next_obs.keys():  # [N_envs, N_obs]
            if k in obs_keys:
                next_obs[k] = torch.from_numpy(o[k]).view(args.num_envs, *o[k].shape[1:])
                step_data[k] = next_obs[k]
                if k in mlp_keys:
                    next_obs[k] = next_obs[k].float()
                    step_data[k] = step_data[k].float()

        # next_obs becomes the new obs
        obs = next_obs

        rewards = torch.from_numpy(rewards).view(args.num_envs, -1).float()
        dones = torch.from_numpy(dones).view(args.num_envs, -1).float()
        step_data["is_first"] = torch.zeros_like(step_data["dones"])
        step_data["dones"] = dones
        step_data["rewards"] = clip_rewards_fn(rewards)

        dones_idxes = dones.nonzero(as_tuple=True)[0].tolist()
        reset_envs = len(dones_idxes)
        if reset_envs > 0:
            reset_data = TensorDict({}, batch_size=[reset_envs], device="cpu")
            for k in real_next_obs.keys():
                if k in obs_keys:
                    reset_data[k] = real_next_obs[k][dones_idxes]
                    if k in mlp_keys:
                        reset_data[k] = reset_data[k].float()
            reset_data["dones"] = torch.ones(reset_envs, 1).float()
            reset_data["actions"] = torch.zeros(reset_envs, np.sum(actions_dim)).float()
            reset_data["rewards"] = step_data["rewards"][dones_idxes].float()
            reset_data["is_first"] = torch.zeros_like(reset_data["dones"]).float()
            if buffer_type == "episode":
                for i, d in enumerate(dones_idxes):
                    if len(episode_steps[d]) >= args.per_rank_sequence_length:
                        episode_steps[d].append(reset_data[i : i + 1][None, ...])
                        rb.add(torch.cat(episode_steps[d], dim=0))
                        episode_steps[d] = []
            else:
                rb.add(reset_data[None, ...], dones_idxes)
            step_data["rewards"][dones_idxes] = torch.zeros_like(reset_data["rewards"]).float()
            step_data["dones"][dones_idxes] = torch.zeros_like(step_data["dones"][dones_idxes]).float()
            step_data["is_first"][dones_idxes] = torch.ones_like(step_data["is_first"][dones_idxes]).float()
            player.init_states(dones_idxes)

        step_before_training -= 1

        # Train the agent
        if global_step >= learning_starts and step_before_training <= 0:
            fabric.barrier()
            if buffer_type == "sequential":
                local_data = rb.sample(
                    args.per_rank_batch_size,
                    sequence_length=args.per_rank_sequence_length,
                    n_samples=args.pretrain_steps if global_step == learning_starts else args.gradient_steps,
                ).to(device)
            else:
                local_data = rb.sample(
                    args.per_rank_batch_size,
                    n_samples=args.pretrain_steps if global_step == learning_starts else args.gradient_steps,
                    prioritize_ends=args.prioritize_ends,
                ).to(device)
            distributed_sampler = BatchSampler(range(local_data.shape[0]), batch_size=1, drop_last=False)
            for i in distributed_sampler:
                if gradient_steps % args.critic_target_network_update_freq == 0:
                    tau = 1 if gradient_steps == 0 else args.critic_tau
                    for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                        tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                train(
                    fabric,
                    world_model,
                    actor,
                    critic,
                    target_critic,
                    world_optimizer,
                    actor_optimizer,
                    critic_optimizer,
                    local_data[i].view(args.per_rank_sequence_length, args.per_rank_batch_size),
                    aggregator,
                    args,
                    is_continuous,
                    cnn_keys,
                    mlp_keys,
                    actions_dim,
                    moments,
                )
                gradient_steps += 1
            step_before_training = args.train_every // single_global_step
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
                "moments": moments.state_dict(),
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
