from gymnasium.envs.registration import register

register(
    id="TSP-v0",
    entry_point="sheeprl.envs.tsp:TSP",
)
