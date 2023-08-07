## Install DIAMBRA environments
1. First you need to register on the [diambra website](https://diambra.ai/register/).
2. Second, you need to install docker and check you have permissions to run it: e.g., `sudo usermod -aG docker $USER` on Linux.
3. Install DIAMBRA with the following comand:
```bash
pip install -e .[diambra]
```

## Install ROMs
The ROMs of the games provided by DIAMBRA must be downloaded and placed in the same folder.

You can get the list of the available games with the following command:
```bash
diambra arena list-roms
```

To install an environment you have to execute the following steps:
1. Download the ROM file and put it in your ROMs folder.
2. If the environment specifications (in `diambra arena list-roms`) contain notes, then apply that notes (for instance, if the ROM file must be renamed).
3. Set the `DIAMBRAROMSPATH` variable either temporarily in your current shell/prompt session, or permanently in your profile (e.g. on linux in `~/.bashrc`)

> **Note**
>
> You can check the validity of the ROM file with: `diambra arena check-roms /absolute/path/to/roms/folder/romFileName.zip` 
>
> For instance, the output of the valid ROM file of *Dead or Alive* is: `Correct ROM file for Dead Or Alive ++, sha256 = d95855c7d8596a90f0b8ca15725686567d767a9a3f93a8896b489a160e705c4e`

## Observation and Action Spaces
The observations space is composed by a python dictionary containing the RGB/grayscale frame and other vectors with additional information.

The action space can be either *discrete* or *multi-discrete*, in both cases you can select whether or not to enable the *attack buttons combination* which increments the number of actions the agent can execute.

Each environment has its own observation and action space, so it is reccomended to check them [here](https://docs.diambra.ai/envs/games/).

> **Note**
>
> You have to be [registered](https://diambra.ai/register/) and logged in to acces the [DIAMRA documentation](https://docs.diambra.ai/).

## Args
The IDs of the DIAMBRA environments are specified [here](https://docs.diambra.ai/envs/games/). To train your agent on a DIAMBRA environment you have to prefix the `"diambra_"` string to the environment ID, e.g., to train your agent on the *Dead Or Alive ++* game, you have to set the `env_id` argument to `"diambra_doapp"` (i.e., `--env_id=diambra_doapp`).

Moreover, the following are the cli arguments specific for the DIAMBRA environments:
* `diambra_action_space`: the type of the action space (either `discrete` or `multi_discrete`).
* `diambra_attack_but_combination`: whether or not to use the attack button combinations.
* `diambra_noop_max`: the maximum number of noop operations after the reset.
* `diambra_actions_stack`: the number of actions stacked in the observations.

DIAMBRA enables to customize the environment with several [settings](https://docs.diambra.ai/envs/#general-environment-settings) and [wrappers](https://docs.diambra.ai/wrappers/).
To modify the default settings or add other wrappers, you have to modify the `make_dict_env` function in the `/sheeprl/utils/utils.py` file.

For insance, in the following example, the player one is selected and a step ratio of $5$ is choosen. Moreover, the rewards are normalized by a factor of $0.3$.

```diff
    env = DiambraWrapper(
        env_id=task_id,
        action_space=args.diambra_action_space,
        screen_size=args.screen_size,
        grayscale=args.grayscale_obs,
        attack_but_combination=args.diambra_attack_but_combination,
        actions_stack=args.diambra_actions_stack,
        noop_max=args.diambra_noop_max,
        sticky_actions=args.action_repeat,
        seed=args.seed,
        rank=rank,
-       diambra_settings={},
+       diambra_settings={
+           "player": "P1"
+           "step_ratio": 5,
+       },
-       diambra_wrappers={},
+       diambra_wrappers={
+           "reward_normalization": True,
+           "reward_normalization_factor": 0.3,
+       },
    )
```

> **Note**
>
> Some settings and wrappers are included in the arguments that can be specified in the command line when the command is launched. These settings/wrappers cannot be specified in the `diambra_settings` and `diambra_wrappers` arguments, respectively.
>
> **Important**
>
> You must set the `sync_env` cli argument to `True`

## Multi-environments / Distributed training
In order to train your agent with multiple environments or to perform a distributed training, you have to specify to the `diambra run` (through the `-s` cli argument) command the number of environments you want to instantiate (both for multi-environment and distributed training). So, you have to multiply the number of environments per process and the number of processes you want to launch (the number of *player* processes for decoupled algorithms). Thus, in case of coupled algorithm (e.g., `dreamer_v2`), if you want distribute your training among $2$ processes each one containing $4$ environments, the total number of environments will be: $2 \cdot 4 = 8$. The command will be:
```bash
diambra run -s=8 lightning run model --devices=2 sheeprl.py dreamer_v2 --env_di=diambra_doapp --num_envs=4 --sync_env=True
```

## Headless machines

If you work on a headless machine, you need to software renderer. We recommend to adopt one of the following solutions:
1. Install the `xvfb` software with the `sudo apt install xvfb` command and prefix the train command with `xvfb-run`. For instance, to train DreamerV2 on the navigate task on an headless machine, you need to run the following command: `xvfb-run diambra run lightning run model --devices=1 sheeprl.py dreamer_v2 --env_id=diambra_doapp --sync_env=True --num_envs=1`
2. Exploit the [PyVirtualDisplay](https://github.com/ponty/PyVirtualDisplay) package.