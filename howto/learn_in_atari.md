## Install Atari environments
First, we should install the Atari environments with:

```bash
pip install .[atari]
```

The Atari environments are provided by the [Arcade Learning Environment](https://ale.farama.org/) (`ale-py`), which already ships the game ROMs and registers the environments in Gymnasium (e.g. `ALE/Pong-v5` or `PongNoFrameskip-v4`).

For more information: https://ale.farama.org/getting-started/

## Train your agent

It is important to remind that not all the algorithms can work with images, so it is necessary to check the first table in the [README](../README.md) and select a proper algorithm.
The list of selectable algorithms is given below:
* `dreamer_v1`
* `dreamer_v2`
* `dreamer_v3`
* `p2e_dv1`
* `p2e_dv2`
* `p2e_dv3`
* `ppo`
* `ppo_decoupled`
* `sac_ae`

Once you have chosen the algorithm you want to train, you can start the train, for instance, of the ppo agent by running:

```bash
python sheeprl.py exp=ppo env=atari env.id=PongNoFrameskip-v4 algo.cnn_keys.encoder=[rgb] fabric.accelerator=cpu fabric.strategy=ddp fabric.devices=2 algo.mlp_keys.encoder=[]
```