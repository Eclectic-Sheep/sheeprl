from functools import partial

import gymnasium as gym
from lightning import Fabric

from sheeprl.envs.diambra_wrapper import DiambraWrapper
from sheeprl.envs.wrappers import RestartOnException

f = Fabric()
num_envs = 8


def make_env(i):
    def f():
        instanziate_kwargs = {}
        instanziate_kwargs["rank"] = i
        print(instanziate_kwargs)
        return DiambraWrapper(
            id="doapp",
            action_space="discrete",
            screen_size=64,
            grayscale=False,
            attack_but_combination=False,
            sticky_actions=1,
            seed=5,
            rank=i,
            diambra_settings={"player": "P1", "characters": "Kasumi"},
            diambra_wrappers={
                "actions_stack": 12,
                "noop_max": 0,
            },
        )

    return f


envs = gym.vector.SyncVectorEnv(
    [
        partial(
            RestartOnException,
            make_env(i),
        )
        for i in range(num_envs)
    ]
)

envs.reset()

for i in range(10):
    o = envs.step(envs.action_space.sample())[0]
    print(f"step: {i}")
