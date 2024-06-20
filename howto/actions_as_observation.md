# Actions as Observations Wrapper
In this how-to, some indications are given on how to use the Actions as Observations Wrapper.

When you want to add the last `n` actions to the observations, you must specify three parameters in the [`./configs/env/default.yaml`](../sheeprl/configs/env/default.yaml) file:
- `actions_as_observation.num_stack` (integer greater than 0): The number of actions to add to the observations.
- `actions_as_observation.dilation` (integer greater than 0): The dilation (number of steps) between one action and the next one.
- `actions_as_observation.noop` (integer or float or list of integer): The noop action to use when resetting the environment, the buffer is filled with this action. Every environment has its own NOOP action, it is strongly recommended to use that action for the correct learning of the algorithm.

## NOOP Parameter
The NOOP parameter must be:
- An integer for discrete action spaces
- A float for continuous action spaces
- A list of integers for multi-discrete action spaces: the length of the list must be equal to the number of actions in the environment.

Each environment has its own NOOP action, usually it is specified in the documentation. Below we reported the list of noop actions of the environments supported in SheepRL:
- MuJoCo (both gymnasium and DMC) environments: `0.0`.
- Atari environments: `0`.
- Crafter: `0`.
- MineRL: `0`.
- MineDojo: `[0, 0, 0]`.
- Super Mario Bros: `0`.
- Diambra:
    - Discrete: `0`.
    - Multi-discrete: `[0, 0]`.
- Box2D (gymnasium):
    - Discrete: `0`.
    - Continuous: `0.0`.