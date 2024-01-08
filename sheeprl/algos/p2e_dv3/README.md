# P2E Dv3
In this folder the [Plan2Explore algorithm](https://arxiv.org/abs/2005.05960) based on Dreamer V3 is implemented.

The Plan2Explore algorithm is designed to efficiently learn and exploit the dynamics of the environment for accomplishing multiple tasks. The algorithm employs two actors: one for exploration and one for learning the task. During the exploratory phase, the exploration actor focuses on discovering new states by selecting actions that lead to unexplored regions. Simultaneously, the task actor learns from the experiences gathered by the exploration actor in a zero-shot manner. Following the exploration phase, the agent can be fine-tuned with experiences collected by the task actor in a few-shot fashion, enhancing its performance on specific tasks.

## Implementation Details

### Scripts

The algorithm implementation is organized into two scripts:

1. **Exploration Script (`p2e_dv2_exploration.py`):**
   - Used for the exploratory phase to learn the dynamics of the environment.
   - Trains the exploration actor to select actions leading to new states.

2. **Fine-tuning Script (`p2e_dv2_finetuning.py`):**
   - Utilized for fine-tuning the agent after the exploration phase.
   - Starts with a trained agent and refines its performance or learns new tasks.
   
### Configuration Constraints

To ensure the proper functioning of the algorithm, the following constraints must be observed:

- **Environment Configuration:** The fine-tuning must be executed with the same environment configurations used during exploration.

- **Hyper-parameter Consistency:** Hyper-parameters of the agent should remain consistent between the exploration and fine-tuning phases.

### Experience Collection

The implementation supports flexibility in experience collection during fine-tuning:

- **Buffer Options:** Fine-tuning can start from the buffer collected during exploration or a new one (`buffer.load_from_exploration` parameter).

- **Initial Experiences:** If using a new buffer, users can decide whether to collect initial experiences (until `learning_start`) with the `actor_exploration` or the `actor_task`. After `learning_start`, only the `actor_task` collects experiences. (`player.actor_type` parameter, can be either `exploration` or `task`).

> [!NOTE]
>
> When exploring, the only valid choice of the `player.actor_type` parameter is `exploration`.

## Usage

To use the Plan2Explore framework, follow these steps:

1. Run the exploration script to learn the dynamics of the environment.
2. Execute the fine-tuning script with the same environment configurations and consistent hyper-parameters.

> [!NOTE]
>
> Choose whether to start fine-tuning from the exploration buffer or create a new buffer, and specify the actor for initial experience collection accordingly.

## Critics

In P2E_DV3 we added the possibility to use more critics for the exploration:
* The exploration critics are defined in the `algo.critics_exploration` config.
* It consists of a python dictionary that contains a pair key-critic_configs.
* Each critic_config has to contain: the weight to give to the advantages (if zero, then the critic is ignored), the reward to use (`intrinsic` or `task`).

> [!NOTE]
>
> There must be at least one intrinsic critic (the reward type must be `intrinsic`)

The following example shows a possible configuration for the exploration critics:
```yaml
critics_exploration:
  intr:
    weight: 0.1
    reward_type: intrinsic
  extr:
    weight: 1.0
    reward_type: task
```