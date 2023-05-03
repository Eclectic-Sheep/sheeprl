# Fabric RL
An easy-to-use framework for reinforcement learning in PyTorch, accelerated with [Lightning Fabric](https://lightning.ai/docs/fabric/stable/).

The goal of this repo is to provide a simple framework for RL algorithms that are scalable and distributed thanks to Lightning Fabric.

## How to use
Clone the repo.

From inside the newly create folder run [Poetry](https://python-poetry.org) to create the virtual environment and install the dependencies:
```bash
poetry install
```

Then you can use one of the already available algorithms, or create your own. For example, to train a PPO agent on the CartPole environment, just run
```bash
python fabricrl/algos/ppo/ppo.py --env-id CartPole-v1
```

Once you did that, a new folder called `logs` will be created, containing the logs of the training. You can visualize them with [TensorBoard](https://www.tensorflow.org/tensorboard):
```bash
tensorboard --logdir logs
```

### More about running an algorithm
What we just run is the PPO algorithm with the default configuration. You can change the configuration by passing arguments to the script. For example, in the default configuration, the number of parallel environments is 4. You can change it to 8 by passing the `--num-envs` argument:
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

## Repository structure
The repository is structured as follows:
```bash
fabricrl
├── algos
│   ├── droq
│   │   ├── args.py
│   │   ├── droq.py
│   │   ├── __init__.py
│   │   └── loss.py
│   ├── __init__.py
│   ├── ppo
│   │   ├── args.py
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   ├── ppo_decoupled.py
│   │   ├── ppo.py
│   │   └── utils.py
│   ├── ppo_recurrent
│   │   ├── agent.py
│   │   ├── __init__.py
│   │   ├── ppo_recurrent.py
│   │   └── utils.py
│   └── sac
│       ├── agent.py
│       ├── args.py
│       ├── __init__.py
│       ├── loss.py
│       ├── sac_decoupled.py
│       ├── sac.py
│       └── utils.py
├── data
│   ├── buffers.py
│   └── __init__.py
├── envs
│   ├── __init__.py
│   └── wrappers.py
├── __init__.py
├── models
│   └── models.py
└── utils
    ├── __init__.py
    ├── metric.py
    ├── model.py
    └── utils.py
```

  * `algos`: contains the implementations of the algorithms. Each algorithm is in a separate folder, and contains the following files:
    * `<algorithm>.py`: contains the implementation of the algorithm.
    * <algorithm>_decoupled.py: contains the implementation of the decoupled version of the algorithm.
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