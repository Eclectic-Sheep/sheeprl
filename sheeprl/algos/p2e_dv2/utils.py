from sheeprl.algos.dreamer_v2.utils import AGGREGATOR_KEYS as AGGREGATOR_KEYS_DV2

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_loss",
    "Loss/value_loss_task",
    "Loss/policy_loss_task",
    "Loss/value_loss_exploration",
    "Loss/policy_loss_exploration",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "Loss/ensemble_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Params/exploration_amount_task",
    "Params/exploration_amount_exploration",
    "Rewards/intrinsic",
    "Values_exploration/predicted_values",
    "Values_exploration/lambda_values",
    "Grads/world_model",
    "Grads/actor_task",
    "Grads/critic_task",
    "Grads/actor_exploration",
    "Grads/critic_exploration",
    "Grads/ensemble",
}.union(AGGREGATOR_KEYS_DV2)
