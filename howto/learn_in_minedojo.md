## Install MineDojo environment
First you need to install the JDK 1.8, on Debian based systems you can run the following:

```bash
sudo apt update -y
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt update -y
sudo apt install -y openjdk-8-jdk
sudo update-alternatives --config java
```

> **Note**
>
> If you work on another OS, you can follow the instructions [here](https://docs.minedojo.org/sections/getting_started/install.html#on-macos) to install JDK 1.8.

Now, you can install the MineDojo environment:

```bash
pip install -e .[minedojo]
```

## MineDojo environments
It is possible to train your agents on all the tasks provided by MineDojo. You need to prefix `"minedojo"` to the `task_id` of the task on which you want to train your agent, and pass it to the `env_id` cli argument.
For instance, you have to set the `env_id` cli argument to `"minedojo_open-ended"` to select the MineDojo open-ended environment.

### Observation Space
We slightly modified the observation space, by reshaping it (based on the idea proposed by Hafner in [DreamerV3](https://arxiv.org/abs/2301.04104)):
* We represent the inventory with a vector with one entry for each item of the game which gives the quantity of the corresponding item in the inventory.
* A max inventory vector with one entry for each item which contains the maximum number of items obtained by the agent so far in the episode.
* A delta inventory vector with one entry for each item which contains the difference of the items in the inventory after the performed action.
* The RGB first-person camera image.
* A vector of three elements representing the life, the food and the oxygen levels of the agent.
* A one-hot vector indicating the equipped item.
* A mask for the action type indicating which actions can be executed.
* A mask for the equip/place arguments indicating which elements can be equipped or placed.
* A mask for the destroy arguments indicating which items can be destroyed.
* A mask for *craft smelt* indicating which items can be crafted.

For more information about the MineDojo observation space, check [here](https://docs.minedojo.org/sections/core_api/obs_space.html).

### Action Space
We decided to convert the 8 multi-discrete action space into a 3 multi-discrete action space:
1. The first maps all the actions (movement, craft, jump, camera, attack, ...).
2. The second one maps the argument for the *craf* action.
3. The third one maps the argument for the *equip*, *place*, and *destroy* actions. 

Moreover, we restrict the look up/down actions between `min_pitch` and `max_pitch` degrees, where `min_pitch` and `max_pitch` are two parameters that can be defined through the `mine_min_pitch` and `mine_max_pitch` cli arguments, respectively.
In addition, we added the forward action when the agent selects one of the follwing actions: `jump`, `sprint`, and `sneak`.
Finally we added sticky action for the `jump` and `attack` actions. You can set the values of the `sticky_jump` and `sticky_attack` parameters through the `mine_sticky_jump` and `mine_sticky_attack` cli arguments, respectively. The sticky actions, if set, force the agent to repeat the selected actions for a certain number of steps.

For more information about the MineDojo action space, check [here](https://docs.minedojo.org/sections/core_api/action_space.html).

> **Note**
> Since the MineDojo environments have a multi-discrete action space, the sticky actions can be easily implemented. The agent will perform the selected action and the sticky actions simultaneously.
>
> The action repeat in the Minecraft environments is set to 1, indedd, It makes no sense to force the agent to repeat an action such as crafting (it may not have enough material for the second action).

## Headless machines

If you work on a headless machine, you need to software renderer. We recommend to adopt one of the following solutions:
1. Install the `xvfb` software with the `sudo apt install xvfb` command and prefix the train command with `xvfb-run`. For instance, to train DreamerV2 on the navigate task on an headless machine, you need to run the following command: `xvfb-run lightning run model --devices=1 sheeprl.py dreamer_v2 --env_id=minedojo_open-ended --cnn_keys rgb`, or `MINEDOJO_HEADLESS=1 lightning run model --devices=1 sheeprl.py dreamer_v2 --env_id=minedojo_open-ended --cnn_keys rgb`.
2. Exploit the [PyVirtualDisplay](https://github.com/ponty/PyVirtualDisplay) package.