## Install Atari environments
First we should install the Atari environments with:

```bash
pip install .[atari]
```

For more information: https://gymnasium.farama.org/environments/atari/ 

## Train your agent
First you need to select which agent you want to train. The list of the trainable agent can be retrieved as follows:

```bash
Usage: sheeprl.py [OPTIONS] COMMAND [ARGS]...

  SheepRL zero-code command line utility.

Options:
  --sheeprl_help  Show this message and exit.

Commands:
  dreamer_v1
  dreamer_v2
  droq
  p2e_dv1
  p2e_dv2
  ppo
  ppo_decoupled
  ppo_recurrent
  sac
  sac_ae
  sac_decoupled
```

It is important to remind that not all the algorithms can work with images, so it is necessary to check the first table in the [README](../README.md) and select a proper algorithm.
The list of selectable algorithms is given below:
* `dreamer_v1`
* `dreamer_v2`
* `dreamer_v3`
* `p2e_dv1`
* `p2e_dv2`
* `ppo`
* `ppo_decoupled`
* `sac_ae`

Once you have chosen the algorithm you want to train, you can start the train, for instance, of the ppo agent by running:

```bash
lightning run model --accelerator=cpu --strategy=ddp --devices=2 sheeprl.py ppo exp=ppo env=atari env.env.id=PongNoFrameskip-v4 cnn_keys.encoder=[rgb]
```