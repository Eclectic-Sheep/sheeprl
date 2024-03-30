## Install Gymnasium MuJoCo/DMC environments
First, you should install the proper environments:

- MuJoCo (Gymnasium): you need to install extra packages: use the `pip install -e .[mujoco]` command to have available all the MuJoCo environments provided by Gymnasium (https://gymnasium.farama.org/environments/mujoco/).
- DMC: you have to install extra packages with the following command: `pip install -e .[dmc]`. (https://github.com/deepmind/dm_control).

## Install OpenGL rendering backands packages

MuJoCo/DMC supports three different OpenGL rendering backends: EGL (headless), GLFW (windowed), and OSMesa (headless).
For each of them, you need to install some packages:
- GLFW: `sudo apt-get install libglfw3 libglew2.2`
- EGL: `sudo apt-get install libglew2.2`
- OSMesa: `sudo apt-get install libgl1-mesa-glx libosmesa6`
In order to use one of these rendering backends, you need to set the `MUJOCO_GL` environment variable to `"glfw"`, `"egl"`, `"osmesa"`, respectively.

> [!NOTE]
>
> The `libglew2.2` could have a different name, based on your OS (e.g., `libglew2.2` is for Ubuntu 22.04.2 LTS).
>
> It could be necessary to install also the `PyOpenGL-accelerate` package with the `pip install PyOpenGL-accelerate` command and the `mesalib` package with the `conda install conda-forge::mesalib` command.

For more information: [https://github.com/deepmind/dm_control](https://github.com/deepmind/dm_control) and [https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl).

## MuJoCo Gymnasium
In order to train your agents on the [MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/) provided by Gymnasium, it is sufficient to select the *MuJoCo* environment (`env=mujoco`) and set the `env.id` to the name of the environment you want to use. For instance, `"Walker2d-v4"` if you want to train your agent in the *walker walk* environment.

```bash
python sheeprl.py exp=dreamer_v3 env=mujoco env.id=Walker2d-v4 algo.cnn_keys.encoder=[rgb]
```

## DeepMind Control
In order to train your agents on the [DeepMind control suite](https://github.com/deepmind/dm_control/blob/main/dm_control/suite/README.md), you have to select the *DMC* environment (`env=dmc`) and set the `domain` and the `task` of the environment you want to use. A list of the available environments can be found [here](https://arxiv.org/abs/1801.00690). For instance, if you want to train your agent on the *walker walk* environment, you need to set the `env.wrapper.domain_name` to `"walker"` and the  `env.wrapper.task_name` to `"walk"`.

```bash
python sheeprl.py exp=dreamer_v3 env=dmc env.wrapper.domain_name=walker env.wrapper.task_name=walk algo.cnn_keys.encoder=[rgb]
```

> [!NOTE]
>
> By default the `env.sync_env` parameter is set to `True`. We recommend not changing this value for the MuJoCo environments to work properly.