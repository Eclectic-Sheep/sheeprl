## Observations types
There are two types of algorithms in this repository:

1. the ones that can work with both image and vector observations.
2. The ones that can work with only vector observations.

In the first case, the observations are returned in the form of python dictionary, whereas in the second case, the observations are returned as 1-dimensional arrays or 3/4-dimensional arrays for grayscale/RGB or stacked images, respectively.

### Both observations
The algorithms that can work with both image and vector observations are specified in [Table 1](../README.md) in the README, and are reported here:
* A2C
* PPO
* PPO Recurrent
* SAC-AE
* Dreamer-V1
* Dreamer-V2
* Dreamer-V3
* Plan2Explore (Dreamer-V1)
* Plan2Explore (Dreamer-V2)
* Plan2Explore (Dreamer-V3)

To run one of these algorithms, it is necessary to specify which observations to use: it is possible to select all the vector observations or only some of them or none of them. Moreover, you can select all/some/none of the image observations.
You just need to pass the  `algo.mlp_keys` and  `algo.cnn_keys` of the encoder and the decoder to the script to select the vector observations and the image observations, respectively.
> [!NOTE]
>
> The  `algo.mlp_keys` and the  `algo.cnn_keys` specified for the encoder are used by default as  `algo.mlp_keys` and  `algo.cnn_keys` of the decoder, respectively.

> **Recommended**
>
> We recommend reading [this](./work_with_multi-encoder_multi-decoder.md) to know how the encoder and decoder work with more observations.

For instance, to train the ppo algorithm on the *doapp* task provided by *DIAMBRA* using image observations and only the `opp_health` and `own_health` as vector observation, you have to run the following command:
```bash
diambra run python sheeprl.py exp=ppo env=diambra env.id=doapp env.num_envs=1 algo.cnn_keys.encoder=[frame] algo.mlp_keys.encoder=[opp_health,own_health]
```

> [!NOTE]
>
> By default the  `algo.mlp_keys` and  `algo.cnn_keys` arguments are set to `[]` (empty list), so no observations are selected for the training. This will raise an exception: if fact, **every algorithm must specify at least one of them**.

It is important to know the observations the environment provides, for instance, the *DIAMBRA* environments provide both vector observations and image observations, whereas all the atari environments provide only the image observations. 
> [!NOTE]
>
> For some environments provided by Gymnasium, e.g. `LunarLander-v2` or `CartPole-v1`, only vector observations are returned, but it is possible to extract the image observation from the render. To do this, it is sufficient to specify the `rgb` key to the  `algo.cnn_keys` args:
> `python sheeprl.py exp=... algo.cnn_keys.encoder=[rgb]`

#### Frame Stack
For image observations, it is possible to stack the last $n$ observations with the argument `frame_stack`. All the observations specified in the  `algo.cnn_keys` argument are stacked.

```bash
python sheeprl.py exp=... env=dmc algo.cnn_keys.encoder=[rgb] env.frame_stack=3
```

#### How to choose the correct keys
When the environment provides both the vector and image observations, you just need to specify which observations you want to use with the  `algo.mlp_keys` and  `algo.cnn_keys`, respectively.

Instead, for those environments that natively do not support both types of observations, we provide a method to obtain the **image observations from the vector observations (NOT VICE VERSA)**. It means that if you choose an environment with only vector observations, you can get also the image observations, but if you choose an environment with only image observations, you **cannot** get the vector observations.

There can be three possible scenarios:
1. You do **not** want to **use** the **image** observations: you don't have to specify any  `algo.cnn_keys` while you have to select the  `algo.mlp_keys`:
   1. if the environment provides more than one vector observation, then you **must choose between them**;
   2. if the environment provides only one vector observation, you can choose the name of the *mlp key*.
2. You want to **use only** the **image** observation: you don't have to specify any  `algo.mlp_keys` while **you must specify the name of the *cnn key*** (if the image observation has to be created from the vector one, the `make_env` function will automatically bind the observation with the specified key, otherwise you must choose a valid one).
3. You want to **use both** the **vector** and **image** observations: you must specify the *cnn key* (as point 2). Instead, for the vector observations, you have two possibilities:
   1. if the environment provides more than one vector observation, then you **must choose between them**;
   2. if the environment provides only one vector observation, you can choose the name of the *mlp key*.

#### Different observations for the Encoder and the Decoder
You can specify different observations for the encoder and the decoder, but there are some constraints:
1. The *mlp* and *cnn* keys of the decoder must be contained in the *mlp* and *cnn* keys of the decoder respectively.
2. Both the intersections between the *mlp* and *cnn* keys of the encoder and decoder cannot be empty.

You can specify the *mlp* and *cnn* keys of the decoder as follows:
```bash
python sheeprl.py exp=dreamer_v3 env=minerl env.id=custom_navigate algo.mlp_keys.encoder=[life_stats,inventory,max_inventory] algo.mlp_keys.decoder=[life_stats,inventory]
```

### Vector observations algorithms
The algorithms that work with only vector observations are reported here:
* SAC
* Droq

For any of them you **must select** only the environments that provide vector observations. For instance, you can train the *SAC* algorithm on the `LunarLanderContinuous-v2` environment, but you cannot train it on the `CarRacing-v2` environment.

For these algorithms, you have to specify the *mlp* keys you want to encode. As usual, you have to specify them through the `mlp_keys.encoder` and `mlp_keys.decoder` arguments (in the command or the configs).

For instance, you can train a SAC agent on the `LunarLanderContinuous-v2` with the following command:
```bash
python sheeprl.py exp=sac env=gym env.id=LunarLanderContinuous-v2 algo.mlp_keys.encoder=[state]
```


### Get Observation Space
It is possible to retrieve the observation space of a specific environment to easily select the observation keys you want to use in your training.

```bash
python examples/observation_space.py env=... env.id=... agent=dreamer_v3 algo.cnn_keys.encoder=[...] algo.mlp_keys.encoder=[...]
```

or for *DIAMBRA* environments:

```bash
diambra run python examples/observation_space.py env=diambra agent=dreamer_v3 env.id=doapp
```

The env argument is the same one you use for training your agent, so it refers to the config folder `sheeprl/configs/env`, moreover you can override the environment id and modify its parameters, such as the frame stack or whether or not to use grayscale observations.
You can modify the parameters as usual by specifying them as cli arguments:

```bash
python examples/observation_space.py env=atari agent=dreamer_v3 env.id=MsPacmanNoFrameskip-v4 env.frame_stack=5 env.grayscale=True algo.cnn_keys.encoder=[frame]
```

> [!NOTE]
>
> You can try to override some *cnn* or *mlp* keys by specifying the `algo.cnn_keys.encoder` and the `algo.mlp_keys.encoder` arguments. **Not all** environments allow it.
> 
> For instance, the `python examples/observation_space.py env=gym agent=dreamer_v3 env.id=LunarLander-v2 algo.cnn_keys.encoder=[custom_cnn_key] algo.mlp_keys.encoder=[custom_mlp_key]` command will return the following observation space: 
>```
>  Observation space of `LunarLander-v2` environment for `dreamer_v3` agent:
>  Dict(
>     'custom_mlp_key': Box([-1.5, -1.5, -5., -5., -3.1415927, -5., -0., -0.], [1.5, 1.5, 5., 5., 3.1415927, 5., 1., 1.], (8,), float32), 
>     'custom_cnn_key': Box(0, 255, (3, 64, 64), uint8)
>  )
>```