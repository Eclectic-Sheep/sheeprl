# Plan2Explore
## Algorithm Overview

The Plan2Explore algorithm is designed to efficiently learn and exploit the dynamics of the environment for accomplishing multiple tasks. The algorithm employs two actors: one for exploration and one for learning the task. During the exploratory phase, the exploration actor focuses on discovering new states by selecting actions that lead to unexplored regions. Simultaneously, the task actor learns from the experiences gathered by the exploration actor in a zero-shot manner. Following the exploration phase, the agent can be fine-tuned with experiences collected by the task actor in a few-shot fashion, enhancing its performance on specific tasks.

## Implementation Details

### Scripts

The algorithm implementation is organized into two scripts:

1. **Exploration Script (`p2e_dv1_exploration.py`):**
   - Used for the exploratory phase to learn the dynamics of the environment.
   - Trains the exploration actor to select actions leading to new states.

2. **Fine-tuning Script (`p2e_dv1_finetuning.py`):**
   - Utilized for fine-tuning the agent after the exploration phase.
   - Starts with a trained agent and refines its performance or learns new tasks.

Both scripts define an `Algorithm` (`P2EDV1Exploration` and `P2EDV1Finetuning`) run by the training loop shared by all the algorithms (`sheeprl.core.loop.run`), and reuse the policy, the writer and the update functions of DreamerV1 (`DreamerV1Policy`, `DreamerV1Writer`, `world_model_learning` and `behaviour_learning` in `sheeprl/algos/dreamer_v1/dreamer_v1.py`).
   
### Configuration Constraints

The fine-tuning starts from the checkpoint of the exploration (`checkpoint.exploration_ckpt_path`) and reads the configuration of the exploration from the `config.yaml` file of its log directory. To ensure the proper functioning of the algorithm, the following constraints are applied:

- **Environment Configuration:** The fine-tuning must be executed on the same environment used during exploration (the same `env.id`, otherwise an error is raised). The frame stack, screen size, action repeat, grayscale, reward clipping, frame stack dilation, maximum episode steps and reward-as-observation settings of the environment are taken from the exploration (as the Minecraft settings, for the MineRL and MineDojo environments).

- **Hyper-parameter Consistency:** The hyper-parameters of the agent are taken from the exploration: the configurations of the world model, of the actor and of the critic, `algo.gamma`, `algo.lmbda`, `algo.horizon` and the sizes and activations of the models, the action normalization (`algo.normalize_actions`), and the cnn and mlp keys.

### Experience Collection

The implementation supports flexibility in experience collection during fine-tuning:

- **Buffer Options:** Fine-tuning can start from the buffer collected during exploration or a new one (`buffer.load_from_exploration` parameter). The buffer of the exploration can be loaded only if it was saved in the checkpoint (`buffer.checkpoint=True` during the exploration); when it is loaded, the number of environments (`env.num_envs`) and of processes (`fabric.devices` and `fabric.num_nodes`) are taken from the exploration.

- **Initial Experiences:** Users can decide whether to collect the experiences until `algo.learning_starts` with the `actor_exploration` or the `actor_task`: no random actions are played. After `algo.learning_starts`, only the `actor_task` collects experiences. (`algo.policy.actor_type` parameter, can be either `exploration` (the default) or `task`).

> [!NOTE]
>
> When exploring, the `algo.policy.actor_type` parameter is always set to `exploration`.

## Usage

To use the Plan2Explore framework, follow these steps:

1. Run the exploration script to learn the dynamics of the environment, e.g. `python sheeprl.py exp=p2e_dv1_exploration env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk algo.cnn_keys.encoder=[rgb]`.
2. Execute the fine-tuning script on the same environment, starting from a checkpoint of the exploration, e.g. `python sheeprl.py exp=p2e_dv1_finetuning env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk checkpoint.exploration_ckpt_path=/path/to/exploration/checkpoint/ckpt_1000000_0.ckpt`.

> [!NOTE]
>
> Choose whether to start fine-tuning from the exploration buffer or create a new buffer, and specify the actor for initial experience collection accordingly.
