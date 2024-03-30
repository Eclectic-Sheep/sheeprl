# Run Experiments

In this document, we give the user some advice to execute its experiments.

> [!WARNING]
>
> Please read our [configs documentation](configs.md) before continuing to read this document.

Now that you are familiar with [hydra](https://hydra.cc/docs/intro/) and the organization of the configs of this repository, we can introduce few constraints to launch experiments:

1. When you launch an experiment you **must** specify the experiment config of the agent you want to train: `python sheeprl.py exp=...`. The list of the available experiment configs can be retrieved with the following command: `python sheeprl.py --help`
2. Then you have to specify the hyper-parameters of your experiment: you can override the hyper-parameters by specifying them as cli arguments (e.g., `exp=dreamer_v3 algo=dreamer_v3 env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk env.action_repeat=2 ...`) or you can write your custom experiment file (you must put it in the `./sheeprl/configs/exp` folder) and call your script with the command `python sheeprl.py exp=custom_experiment` (the last option is recommended). There are some available examples, just check the [exp folder](../sheeprl/configs/exp/).
3. You **cannot mix the agent command with the configs of another algorithm**, this might raise an error or create anomalous behaviors. So if you want to train the `dreamer_v3` agent, be sure to select the correct algorithm configuration (in our case `algo=dreamer_v3`)
4. To change the optimizer of an algorithm through the CLI you must do the following: suppose that you want to run an experiment with Dreamer-V3 and want to change the world model optimizer from Adam (default in the `sheeprl/configs/algo/dreamer_v3.yaml` config) with SGD, then in the CLI you must type `python sheeprl.py algo=dreamer_v3 ... optim@algo.world_model.optimizer=sgd`, where `optim@algo.world_model.optimizer=sgd` means that the `optimizer` field of the `world_model` of the `algo` config choosen (the dreamer_v3.yaml one) will be equal to the config `sgd.yaml` found under the `sheeprl/configs/optim` folder 
