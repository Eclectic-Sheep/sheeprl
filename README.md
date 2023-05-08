# ⚡ Fabric RL
An easy-to-use framework for reinforcement learning in PyTorch, accelerated with [Lightning Fabric](https://lightning.ai/docs/fabric/stable/).

## Why
We want to provide a framework for RL algorithms that is at the same time simple and scalable thanks to Lightning Fabric.

Moreover, in many RL repositories, the RL algorithm is tightly coupled with the environment, making it harder to extend them beyond the gym interface. We want to provide a framework that allows to easily decouple the RL algorithm from the environment, so that it can be used with any environment. 

## How to use
Clone the repo.

```bash
git clone <http-url>
cd fabric_rl
```

From inside the newly create folder run [Poetry](https://python-poetry.org) to create the virtual environment and install the dependencies:
```bash
poetry install
```

Now you can use one of the already available algorithms, or create your own. 

For example, to train a PPO agent on the CartPole environment, just run
```bash
python fabricrl/algos/ppo/ppo.py --env-id CartPole-v1
```
---

That's all it takes to train an agent with Fabric RL! 🎉

### :chart_with_upwards_trend: Check your results

Once you trained an agent, a new folder called `logs` will be created, containing the logs of the training. You can visualize them with [TensorBoard](https://www.tensorflow.org/tensorboard):
```bash
tensorboard --logdir logs
```

### :nerd_face: More about running an algorithm
What you run is the PPO algorithm with the default configuration. But you can also change the configuration by passing arguments to the script.

For example, in the default configuration, the number of parallel environments is 4. Let's try to change it to 8 by passing the `--num-envs` argument:
```bash
python fabricrl/algos/ppo/ppo.py --env-id CartPole-v1 --num-envs 8
```

All the available arguments, with their descriptions, are listed in the `args.py` file under the algorithm's folder.

### Running with Lightning Fabric
To run the algorithm with Lightning Fabric, you need to call Lightning with its parameters. For example, to run the PPO algorithm with 4 parallel environments on 2 nodes, you can run:
```bash
lightning run model --accelerator=cpu --strategy=ddp --devices=2 fabricrl/algos/ppo/ppo.py --env-id CartPole-v1
```

You can check the available parameters for Lightning Fabric [here](https://lightning.ai/docs/fabric/stable/api/fabric_args.html).

## :book: Repository structure
The repository is structured as follows:
```bash
fabricrl
├── algos
│   └──<algorithm>
│       ├── __init__.py
│       ├── <algorithm>.py
│       ├── <algorithm>_decoupled.py
│       ├── agent.py
│       ├── args.py
│       ├── loss.py
│       ├── ... (other files)
│       └── utils.py
├── data
│   └── buffers.py
├── envs
│   └── wrappers.py
├── models
│   └── models.py
└── utils
    ├── metric.py
    ├── model.py
    └── utils.py
```

  * `algos`: contains the implementations of the algorithms. Each algorithm is in a separate folder, and contains the following files:
    * `<algorithm>.py`: contains the implementation of the algorithm.
    * `<algorithm>_decoupled.py`: contains the implementation of the decoupled version of the algorithm.
    * `agent`: optional, contains the implementation of the agent.
    * `args.py`: contains the arguments of the algorithm, with their default values and descriptions.
    * `loss.py`: contains the implementation of the loss functions of the algorithm.
    * `utils.py`: contains utility functions for the algorithm.
  * `data`: contains the implementation of the data buffers.
  * `envs`: contains the implementation of the environment wrappers.
  * `models`: contains the implementation of the NN models (building blocks)
  * `utils`: contains utility functions for all the algorithms.

#### Coupled vs Decoupled
In the coupled version of an algorithm, the agent interacts with the environment and executes the training loop. 

<p align="center">
  <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/fabric_coupled.png">
</p>

In the decoupled version, a process is responsible only for interacting with the environment, and all the other processes are responsible for executing the training loop. The two processes communicate throgh [collectives](https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.plugins.collectives.TorchCollective.html#lightning.fabric.plugins.collectives.TorchCollective) and thanks to Fabric's flexibility we can run Player and Trainers on different devices.

<p align="center">
  <img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/examples/fabric/reinforcement-learning/ppo_fabric_decoupled.png">
</p>

#### Coupled
The algorithm is implemented in the `<algorithm>.py` file. 

There are 2 functions inside this script:
  * `main()`: initializes all the components of the algorithm, and executes the interactions with the environment. Once enough data is collected, the training loop is executed by calling the `train()` function.
  * `train()`: executes the training loop. It samples a batch of data from the buffer, compute the loss, and updates the parameters of the policy and value networks.

#### Decoupled
The decoupled version of an algorithm is implemented in the `<algorithm>_decoupled.py` file.

There are 3 functions inside this script:
  * `main()`: initializes all the components of the algorithm, and executes the interactions with the environment. Once enough data is collected, the training loop is executed by calling the `train()` function.
  * `trainer()`: executes the training loop. It samples a batch of data from the buffer, compute the loss, and updates the parameters of the policy and value networks.
  * `player()`: executes the interactions with the environment. It samples an action from the policy network, executes it in the environment, and stores the transition in the buffer.

## Algorithms implementation
You can check inside the folder of each algorithm the `readme.md` file for the details about the implementation.

All algorithms are kept as simple as possible, in a [CleanRL](https://github.com/vwxyzjn/cleanrl) fashion. But to allow for more flexibility and also more clarity, we tried to abstract away anything that is not strictly related with the training loop of the algorithm. 

For example, we decided to create a `models` folder with ready-made NN models that can be composed to create the NN model of the agent.

For each algorithm, losses are kept in a separate module, so that their implementation is clear and can be easily utilized also for the decoupled or the recurrent version of the algorithm.

## Buffer
For the buffer implementation, we choose to use a wrapper around a [TensorDict](https://pytorch.org/rl/tensordict/reference/generated/tensordict.TensorDict.html).

TensorDict comes handy since we can easily add custom fields to the buffer as if we are working with dictionaries, but we can also easily perform operations on them as if we are working with tensors.

This flexibility makes it very simple to implement, with the single class `ReplayBuffer`, all the buffers needed for on-policy and off-policy algorithms.

### :mag: Technical details
The tensor's shape in the TensorDict are `(T, B, *)`, where `T` is the number of timesteps, `B` is the number of parallel environments, and `*` is the shape of the data.

For the `ReplayBuffer` to be used as a RolloutBuffer, the proper `buffer_size` must be specified. For example, for PPO, the `buffer_size` must be `T * B`, where `T` is the number of timesteps and `B` is the number of parallel environments.

## :bow: Contributing
The best way to contribute is by opening an issue to discuss a new feature or a bug, or by opening a PR to fix a bug or to add a new feature.