# Evaluate your Agents

In this document, we give the user some advice to evaluate its agents. To evaluate an agent, it is simply necessary run the `./sheeprl_eval.py` script, passing the path to the checkpoint of the agent you want to evaluate.

```bash
python sheeprl_eval.py checkpoint_path=/path/to/checkpoint.ckpt
```

The agent and the configs used during the training are loaded automatically. The user can modify only few parameters for evaluation:

1. `fabric` related ones: you can use the accelerator you want for evaluating the agent, you just need to specify it in the command. For instance, `python sheeprl_eval.py checkpoint_path=/path/to/checkpoint.ckpt fabric.accelerator=gpu` for evaluating the agent on the GPU. If you want to choose the GPU, then you need to define the `CUDA_VISIBLE_DEVICES` environment variable in the `.env` file or set it before running the script. For example, you can execute the following command to evaluate your agent on the GPU with index 2: `CUDA_VISIBLE_DEVICES="2" python sheeprl_eval.py checkpoint_path=/path/to/checkpoint.ckpt fabric.accelerator=gpu`. By default, the number of devices and nodes is set to 1, while the precision and the plugins are set to the ones set in the checkpoint config.
2. `env.capture_video`: you can decide whether to capture the video of the episode during the evaluation or not. For instance, `python sheeprl_eval.py checkpoint_path=/path/to/checkpoint.ckpt env.capture_video=Ture` for capturing the video of the evaluation.
3. `seed`: the user can specify the seed used for evaluation with `python sheeprl_eval.py checkpoint_path=/path/to/checkpoint.ckpt seed=42`. By default the seed is set to `None`.

All the other parameters are loaded from the checkpoint config file used during the training. Moreover, the following parameters are automatically set during the evaluation:

* `cfg.env.num_envs`, i.e. the number of environments used during the evaluation, is set to 1
* `cfg.fabric.devices`and `cfg.fabric.num_nodes` are set to 1

> [!NOTE]
>
> You cannot modify the number of processes to spawn. The evaluation is made with only 1 process.

## Log Directory
By default, the evaluation logs are stored in the same folder of the experiment. Suppose to have trained a `PPO` agent in the `CartPole-v1` environment. The log directory is organized as follows:
```tree
logs
└── runs
    └── ppo
        └── CartPole-v1
            └── 2023-10-27_11-46-05_default_42
                ├── .hydra
                │   ├── config.yaml
                │   ├── hydra.yaml
                │   └── overrides.yaml
                ├── cli.log
                └── version_0
                    ├── checkpoint
                    │   ├── ckpt_1024_0.ckpt
                    │   ├── ckpt_1536_0.ckpt
                    │   └── ckpt_512_0.ckpt
                    ├── events.out.tfevents.1698399966.72040.0
                    ├── memmap_buffer
                    │   └── rank_0
                    │       ├── actions.memmap
                    │       ├── actions.meta.pt
                    │       ├── advantages.memmap
                    │       ├── advantages.meta.pt
                    │       ├── dones.memmap
                    │       ├── dones.meta.pt
                    │       ├── logprobs.memmap
                    │       ├── logprobs.meta.pt
                    │       ├── meta.pt
                    │       ├── returns.memmap
                    │       ├── returns.meta.pt
                    │       ├── rewards.memmap
                    │       ├── rewards.meta.pt
                    │       ├── state.memmap
                    │       ├── state.meta.pt
                    │       ├── values.memmap
                    │       └── values.meta.pt
                    └── train_videos
                        ├── rl-video-episode-0.mp4
                        ├── rl-video-episode-1.mp4
                        └── rl-video-episode-8.mp4
```

Where `./logs/runs/ppo/2023-10-27_11-46-05_default_42` contains your experiment. The evaluation script will create a subfolder, named `evaluation`, in the `./logs/runs/ppo/2023-10-27_11-46-05_default_42/version_0` folder, which will contain all the evaluations of the agents.

For example, if we run two evaluations, then the log directory of the experiment will be as follows:
```diff
logs
└── runs
    ├── .hydra
    │   ├── config.yaml
    │   ├── hydra.yaml
    │   └── overrides.yaml
    ├── cli.log
    └── ppo
        └── CartPole-v1
            └── 2023-10-27_11-46-05_default_42
                ├── .hydra
                │   ├── config.yaml
                │   ├── hydra.yaml
                │   └── overrides.yaml
                ├── cli.log
                └── version_0
                    ├── checkpoint
                    │   ├── ckpt_1024_0.ckpt
                    │   ├── ckpt_1536_0.ckpt
                    │   └── ckpt_512_0.ckpt
+                   ├── evaluation
+                   │   ├── version_0
+                   │   │   ├── events.out.tfevents.1698400212.73839.0
+                   │   │   └── test_videos
+                   │   │       └── rl-video-episode-0.mp4
+                   │   └── version_1
+                   │       ├── events.out.tfevents.1698400283.74353.0
+                   │       └── test_videos
+                   │           └── rl-video-episode-0.mp4
                    ├── events.out.tfevents.1698399966.72040.0
                    ├── memmap_buffer
                    │   └── rank_0
                    │       ├── actions.memmap
                    │       ├── actions.meta.pt
                    │       ├── advantages.memmap
                    │       ├── advantages.meta.pt
                    │       ├── dones.memmap
                    │       ├── dones.meta.pt
                    │       ├── logprobs.memmap
                    │       ├── logprobs.meta.pt
                    │       ├── meta.pt
                    │       ├── returns.memmap
                    │       ├── returns.meta.pt
                    │       ├── rewards.memmap
                    │       ├── rewards.meta.pt
                    │       ├── state.memmap
                    │       ├── state.meta.pt
                    │       ├── values.memmap
                    │       └── values.meta.pt
                    └── train_videos
                        ├── rl-video-episode-0.mp4
                        ├── rl-video-episode-1.mp4
                        └── rl-video-episode-8.mp4
```