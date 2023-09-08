# Run Experiments

In this document we give the user some advices to execute its experiments.

> **Warning**
>
> Please read our [configs documentation](configs.md) before continuing to read this document.

Now that you are familiar with [hydra](https://hydra.cc/docs/intro/) and the organization of the configs of this repository, we can introduce few constraints to launch experiments:
1. When you launch an experiment you **must** specify the command of the agent you want to train: `python sheeprl.py <command> ...`. The list of the available commands can be retrieved with the following command: `python sheeprl.py --sheeprl_help`
2. Then you have to specify the hyper-parameters of your experiment: you can override the hyper-parameters by specifing them as cli arguments (e.g., `env=dmc env.id=walker_walk algo=dreamer_v3 env.action_repeat=2 ...`) or you can write your custom experiment file (you must put it in the `./sheeprl/configs/exp` folder) and call your script with the command `python sheeprl.py <command> exp=custom_experiment` (the last option is recommended).
> **Note**
>
> There are some available examples, just check the [exp folder](../sheeprl/configs/exp/).
3. You **cannot mix the agent command with the configs of another algorithm**, this might raise an error or create anomalous behaviors. So if you want to train the `dreamer_v3` agent, be sure that to select the correct algorithm configuration (in our case `algo=dreamer_v3`).