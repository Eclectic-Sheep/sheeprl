## Install DIAMBRA environments
1. First you need to register on the [diambra website](https://diambra.ai/register/).
2. Second, you need to install docker and check that you have permission to run it: e.g., on Linux, you can run the following command: `sudo usermod -aG docker $USER`.
3. Install DIAMBRA with the following command:
```bash
pip install -e .[diambra]
```

## Install ROMs
The ROMs of the games provided by DIAMBRA must be downloaded and placed in a folder of your choice (must be the same folder for all the ROMs).

You can get the list of the available games with the following command:
```bash
diambra arena list-roms
```

This command will return a list of games with some additional information:
* Title: the name of the environment/game
* Difficulty levels: the available difficulties for that environment.
* SHA256 sum: to check the validity of the ROM file.
* Original ROM name: the name of the file.
* Search keywords: keywords that can be used to search the ROM file on the web.
* Characters list: the list of available characters.
* Notes: additional notes to take into account (e.g., how you have to rename the ROM file).

To install an environment you have to execute the following steps:
1. Download the ROM file and put it in your ROMs folder.
2. If the environment specifications (in `diambra arena list-roms`) contain notes, then apply those notes (for instance, if the ROM file must be renamed).
3. Set the `DIAMBRAROMSPATH` variable either temporarily in your current shell/prompt session or permanently in your profile (e.g. on Linux in `~/.bashrc`): `export DIAMBRAROMSPATH=/absolute/path/to/roms/folder`

> [!NOTE]
>
> You can check the validity of the ROM file with: `diambra arena check-roms /absolute/path/to/roms/folder/romFileName.zip` 
>
> For instance, the output of the valid ROM file of *Dead or Alive* is: `Correct ROM file for Dead Or Alive ++, sha256 = d95855c7d8596a90f0b8ca15725686567d767a9a3f93a8896b489a160e705c4e`

## Observation and Action Spaces
The observation space is composed of a python dictionary containing the RGB/grayscale frame and other vectors with additional information. For more information about the observation space, check [here](https://docs.diambra.ai/envs/#observation-space).

The action space can be either *discrete* or *multi-discrete*, in both cases, you can select whether or not to enable the *attack buttons combination* which increments the number of actions the agent can execute. For more information about the action space, check [here](https://docs.diambra.ai/envs/#action-spaces).

Each environment has its own observation and action space, so it is recommended to check them [here](https://docs.diambra.ai/envs/games/).

> [!NOTE]
>
> You have to be [registered](https://diambra.ai/register/) and logged in to acces the [DIAMRA documentation](https://docs.diambra.ai/).

The observation space is slightly modified to be compatible with our algorithms, in particular, the `gymnasium.spaces.Discrete` observations are converted into `gymnasium.spaces.Box` observations with dimension `(1,)`, of type `int` and range from `0` to `n - 1`, where `n` is the number of options of the Discrete space. Finally, the  `gymnasium.spaces.MultiDiscrete` observations are converted into `gymnasium.spaces.Box` observations with dimension `(k,)` where `k` is the length of the MultiDiscrete space, of type `int` and range from `0` to `n[i] - 1` where `n[i]` is the number of options of the *i-th* element of the MultiDiscrete.

> [!NOTE]
>
> To know more about gymnasium spaces, check [here](https://gymnasium.farama.org/api/spaces/fundamental/).

## Multi-environments / Distributed training
In order to train your agent with multiple environments or to perform distributed training, you have to specify to the `diambra run` command the number of environments you want to instantiate  (through the `-s` cli argument). So, you have to multiply the number of environments per single process and the number of processes you want to launch (the number of *player* processes for decoupled algorithms). Thus, in the case of coupled algorithms (e.g., `dreamer_v2`), if you want to distribute your training among $2$ processes each one containing $4$ environments, the total number of environments will be: $2 \cdot 4 = 8$. The command will be:
```bash
diambra run -s=8 python sheeprl.py exp=dreamer_v3 env=diambra env.id=doapp env.num_envs=4 env.sync_env=True algo.cnn_keys.encoder=[frame] fabric.devices=2
```

## Args
The IDs of the DIAMBRA environments are specified [here](https://docs.diambra.ai/envs/games/). To train your agent on a DIAMBRA environment you have to select the DIAMBRA configs with the argument `env=diambra`, then set the `env.id` argument to the environment ID, e.g., to train your agent on the *Dead Or Alive ++* game, you have to set the `env.id` argument to `doapp` (i.e., `env.id=doapp`).

```bash
diambra run -s=4 python sheeprl.py exp=dreamer_v3 env=diambra env.id=doapp env.num_envs=4 algo.cnn_keys.encoder=[frame]
```

Another possibility is to create a new config file in the `sheeprl/configs/exp` folder, where you specify all the configs you want to use in your experiment. An example of a custom configuration file is available [here](../sheeprl/configs/exp/dreamer_v3_L_doapp.yaml).

DIAMBRA enables to customize the environment with several [settings](https://docs.diambra.ai/envs/#general-environment-settings) and [wrappers](https://docs.diambra.ai/wrappers/).
To modify the default settings or add other wrappers, you have to add the settings or wrappers you want in `env.wrapper.diambra_settings` or `env.wrapper.diambra_wrappers`, respectively.

For instance, in the following example, we create the `custom_exp.yaml` file in the `sheeprl/configs/exp` folder where we select the DIAMBRA environment, in addition, the player one is selected and a step ratio of $5$ is chosen. Moreover, the rewards are normalized by a factor of $0.3$.


```yaml
# @package _global_

defaults:
    - dreamer_v3
    - override /env: diambra 
    - _self_

env:
    id: doapp
    wrapper:
    diambra_settings:
        characters: Kasumi
        step_ratio: 5
        role: P1
    diambra_wrappers:
        normalize_reward: True
        normalization_factor: 0.3
```

Now, to run your experiment, you have to execute the following command:
```bash
diambra run -s=4 python sheeprl.py exp=custom_exp env.num_envs=4
```

> [!NOTE]
>
> Some settings and wrappers are included in the cli arguments when the command is launched. These settings/wrappers cannot be specified in the `diambra_settings` and `diambra_wrappers` parameters, respectively.
> The settings/wrappers you cannot specify in the `diambra_settings` and `diambra_wrappers` parameters are the following:
> * `action_space` (settings): you can set it with the `env.wrapper.action_space` argument.
> * `n_players` (settings): you cannot set it, since it is always `1`.
> * `frame_shape` (settings and wrappers): you can set it with the `env.screen_size` argument.
> * `flatten` (wrappers): you cannot set it, since it is always `True`.
> * `repeat_action` (wrappers): you can set it with the `env.action_repeat` argument.
> * `stack_frames` (wrappers): you can set it with the `env.stack_frames` argument.
> * `dilation` (wrappers): you can set it with the `env.frame_stack_dilation` argument
>
> When you set the `action_repeat` cli argument greater than one (i.e., the `repeat_action` DIAMBRA wrapper), the `step_ratio` diambra setting is automatically modified to $1$ because it is a DIAMBRA requirement.
>
> You can increase the performance of the DIAMBRA engine with the `env.wrapper.increase_performance` parameter. When set to `True` the engine is faster, but the recorded video will have the dimension specified by the `env.screen_size` parameter. 
>
> **Important**
>
> If you want to use the `AsyncVectorEnv` ([https://gymnasium.farama.org/api/vector/#async-vector-env](https://gymnasium.farama.org/api/vector/#async-vector-env)), you **must** set the **`env.wrapper.diambra_settings.splash_screen`** cli argument to **`False`**. Moreover, you must set the number of containers to `env.num_envs + 1` (i.e., you must set the `-s` cli argument as specified before).

## Headless machines

If you work on a headless machine, you need to software renderer. We recommend to adopt one of the following solutions:
1. Install the `xvfb` software with the `sudo apt install xvfb` command and prefix the training command with `xvfb-run`. For instance, to train DreamerV2 on the navigate task on a headless machine, you need to run the following command: `xvfb-run diambra run python sheeprl.py exp=dreamer_v3 env=diambra env.id=doapp env.sync_env=True env.num_envs=1 algo.cnn_keys.encoder=[frame] fabric.devices=1`
2. Exploit the [PyVirtualDisplay](https://github.com/ponty/PyVirtualDisplay) package.