## Install Atari environments
First, we should install the Super Mario Bros environment with:

```bash
pip install .[supermario]
```

For more information: https://github.com/Kautenja/gym-super-mario-bros/tree/master

## Environment Config
The default configurations of the Super Mario Bros environment are in the `./sheeprl/configs/env/super_mario_bros.yaml` file.

```yaml
defaults:
  - default
  - _self_

# Override from `default` config
id: SuperMarioBros-v0
frame_stack: 1
sync_env: False
action_repeat: 1

# Wrapper to be instantiated
wrapper:
  _target_: sheeprl.envs.super_mario_bros.SuperMarioBrosWrapper
  id: ${env.id}
  action_space: simple # or complex or right_only
  render_mode: rgb_array
```
The parameters under the `wrapper` key are explained below:
- `id`: The id of the environment, check [here](https://github.com/Kautenja/gym-super-mario-bros/tree/master) which environments can be instantiated.
- `action_space`: The actions that can be performed by the agent (always discrete actions). The possible options are: `simple`, `right_only`, or `complex`. Check [here](https://github.com/Kautenja/gym-super-mario-bros/blob/bcb8f10c3e3676118a7364a68f5c0eb287116d7a/gym_super_mario_bros/actions.py) the differences between them.
- `render_mode`: one between `rgb_array` or `human`.


## Train your agent

It is important to remember that not all the algorithms can work with images, so it is necessary to check the first table in the [README](../README.md) and select a proper algorithm.
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
python sheeprl.py exp=ppo env=super_mario_bros env.id=SuperMarioBros-v0 algo.cnn_keys.encoder=[rgb] fabric.accelerator=cpu fabric.strategy=ddp fabric.devices=2 algo.mlp_keys.encoder=[]
```