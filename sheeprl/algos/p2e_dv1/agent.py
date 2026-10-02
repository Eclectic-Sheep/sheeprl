from typing import Any, Dict, Sequence, Tuple, Union

import gymnasium
import hydra
from lightning.fabric import Fabric
from lightning.pytorch.utilities.seed import isolate_rng
from torch import nn

from sheeprl.algos.dreamer_v1.agent import WorldModel
from sheeprl.algos.dreamer_v1.agent import build_models as dv1_build_models
from sheeprl.algos.dreamer_v2.agent import Actor as DV2Actor
from sheeprl.algos.dreamer_v2.agent import MinedojoActor as DV2MinedojoActor
from sheeprl.models.models import MLP
from sheeprl.utils.utils import init_weights

# In order to use the hydra.utils.get_class method, in this way the user can
# specify in the configs the name of the class without having to know where
# to go to retrieve the class
Actor = DV2Actor
MinedojoActor = DV2MinedojoActor


def build_models(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gymnasium.spaces.Dict,
) -> Tuple[WorldModel, nn.Module, nn.Module, nn.Module, nn.Module, nn.ModuleList]:
    """Create all the models of P2E-DV1 with their initial weights, in the order of the old training loops (which
    gives the same weights). They are not set up with Fabric.

    Returns:
        The world model, the exploration actor and critic, the task actor and critic, and the ensembles.
    """
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic

    # Sizes
    latent_state_size = world_model_cfg.stochastic_size + world_model_cfg.recurrent_model.recurrent_state_size

    # Create exploration models
    world_model, actor_exploration, critic_exploration = dv1_build_models(
        actions_dim=actions_dim, is_continuous=is_continuous, cfg=cfg, obs_space=obs_space
    )

    # Create task models
    actor_cls = hydra.utils.get_class(cfg.algo.actor.cls)
    actor_task: Union[Actor, MinedojoActor] = actor_cls(
        latent_state_size=latent_state_size,
        actions_dim=actions_dim,
        is_continuous=is_continuous,
        init_std=actor_cfg.init_std,
        min_std=actor_cfg.min_std,
        mlp_layers=actor_cfg.mlp_layers,
        dense_units=actor_cfg.dense_units,
        activation=hydra.utils.get_class(actor_cfg.dense_act),
        distribution_cfg=cfg.distribution,
        layer_norm=False,
        expl_amount=actor_cfg.expl_amount,
        expl_decay=actor_cfg.expl_decay,
        expl_min=actor_cfg.expl_min,
    )
    critic_task = MLP(
        input_dims=latent_state_size,
        output_dim=1,
        hidden_sizes=[critic_cfg.dense_units] * critic_cfg.mlp_layers,
        activation=hydra.utils.get_class(critic_cfg.dense_act),
        flatten_dim=None,
    )
    actor_task.apply(init_weights)
    critic_task.apply(init_weights)

    # Initialize the ensembles with different seeds to be sure they have different weights
    ens_list = []
    with isolate_rng():
        for i in range(cfg.algo.ensembles.n):
            fabric.seed_everything(cfg.seed + i)
            ens_list.append(
                MLP(
                    input_dims=(
                        int(sum(actions_dim))
                        + cfg.algo.world_model.recurrent_model.recurrent_state_size
                        + cfg.algo.world_model.stochastic_size
                    ),
                    output_dim=world_model.encoder.cnn_output_dim + world_model.encoder.mlp_output_dim,
                    hidden_sizes=[cfg.algo.ensembles.dense_units] * cfg.algo.ensembles.mlp_layers,
                    activation=hydra.utils.get_class(cfg.algo.ensembles.dense_act),
                ).apply(init_weights)
            )
    return world_model, actor_exploration, critic_exploration, actor_task, critic_task, nn.ModuleList(ens_list)
