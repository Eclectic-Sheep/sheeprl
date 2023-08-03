## Observations types
There are two types of algorithms in this repository: *(i)* the ones that can work with both image-like and vector-like observations. *(ii)* The ones that can work with either image-like or vector-like observations.
In the first case the observations are returned in form of python dictionary, whereas in the second case the observations are returned as 1-dimensional arrays or 3/4-dimensional arrays for grayscale/rgb or stacked images, respectively.

### Dict observations
The algorithms that can work with both image-like and vector-like observations are specified in [Table 1](../README.md) in the README, and are reported here:
* PPO
* SAC-AE
* Dreamer-V1
* Dreamer-V2
* Dreamer-V3
* Plan2Explore (Dreamer-V1)
* Plan2Explore (Dreamer-V1)

To run one of these algorithms, it is necessary to specify which observations to use: it is possible to select all the vector-like observations or only some of them or none of them. Moreover you can select all/some/none of the image-like observations.
You just need to pass the `mlp_keys` and `cnn_keys` to the script to select the vector-like observations and the image-like observations, respectively.

For instance, to train the ppo algorithm on the *walker walk* task provided by *DMC* using image-like observations and only the `orientations` and `velocity` as vector-like observation, you have to run the following command:
```bash
lightning run model sheeprl.py ppo --env_id=dmc_walker_walk --cnn_keys rgb --mlp_keys orientations velocity
```

> **Note**
>
> By default the `mlp_keys` and `cnn_keys` arguments are set to None, so no observations are selected for the training.

It is important to know the observations the environment provides, for instance, the `dmc_walker_walk` provides both vector-like observations and image-like observations, whereas all the atari environments provide only the image-like observations. 
> **Note**
>
> In the documentation of some environments provided by gymnasium, e.g. `LunarLander-v2` or `CartPole-v1`, only vector-like observations are specified, but it is possible to extract the image-like observation from the render. To do this, it is sufficient to specify the `rgb` key to the `cnn_keys` args:
> `lightning run model sheeprl.py ppo --cnn_keys rgb`

#### Frame Stack
For image-like observations it is possible to stack the last $n$ observations with the argument `frame_stack`. If you want to stack more frames, then you must specify on which image-like observations you want to apply the `FrameStack` wrapper through `frame_stack_keys` argument.

```bash
lightning run model sheeprl.py ppo --env_id=dmc_walker_walk --frame_stack=3 --frame_stack_keys rgb
```

#### How to choose the correct keys
When the environment provides both the vector-like and image-like observations, you just need to specify which observations you want to use with the `mlp_keys` and `cnn_keys`, respectively.

Instead, for those environments that natively do not support both the type of observations, we provide a method to obtain the **image-like observations from the vector-like observations (NOT VICEVERSA)**. It means that if you choose an environment with only vector-like observations, you can get also the image-like observations, but if you choose an environment with only image-like observations, you **cannot** get the vector-like observations.
In this case, there can happen 3 situations:
1. You do **not** want to **use** the **image**-like observations: you have to let empty the `cnn_keys` argument and you have to select the `mlp_keys`: *(i)* if the environment provies more than one vector-like observation, then you have to choose between them; *(ii)* if the environment provides only one vector-like observation, you can choose the name of the *mlp key* or use the default one (`state`, used when you do not specify any *mlp key*).
2. If you want to **use only** the **image**-like observation: you have to let empty the `mlp_keys` argument and **you must specify the name of the *cnn key*** (there is not a specific name, you are giving the name to the observation, indeed, the `make_dict_env` function will automatically bind the observation with the specified key).
3. You want to **use both** the **vector**-like and **image**-like observations: You must specify the *cnn key* (as point 2). Instead, for the vector-like observations, you have two possibilities: *(i)* if the environment provies more than one vector-like observation, then you **must choose between them**; *(ii)* if the environment provides only one vector-like observation, you **must specify** the default vector-like observation key, i.e., **`state`**.

### Vector/Image-like Observations
The algorithms which works either with vector-like or image-like observations are reported here:
* PPO Recurrent
* SAC
* Droq

All of them works only with vector-like observations, so you **must select** only the environments that provide vector-like observations. For instance, you can train the *PPO Recurrent* algorithm on the `LunarLander-v2` environment, but you cannot train it on the `CarRacing-v2` environment.

For these algorithms, you do not have to specify anything about the observations space, only to indicate the environment on which you want to train the agent.
