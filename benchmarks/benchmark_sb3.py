import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import A2C, PPO, SAC  # noqa: F401
from torchmetrics import SumMetric

from sheeprl.utils.timer import timer

# Stable Baselines3 - PPO - CartPolev1
if __name__ == "__main__":
    with timer("run_time", SumMetric, sync_on_compute=False):
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        model = PPO("MlpPolicy", env, verbose=0, device="cpu", n_steps=128)
        model.learn(total_timesteps=1024 * 64, log_interval=None)
    print(timer.compute())
    print(sb3.common.evaluation.evaluate_policy(model.policy, env))


# Stable Baselines3 - A2C - CartPolev1
# Decomment below to run A2C benchmarks

# if __name__ == "__main__":
#     with timer("run_time", SumMetric, sync_on_compute=False):
#         env = gym.make("CartPole-v1", render_mode="rgb_array")
#         model = A2C("MlpPolicy", env, verbose=0, device="cpu", vf_coef=1.0)
#         model.learn(total_timesteps=1024 * 64, log_interval=None)
#     print(timer.compute())
#     print(sb3.common.evaluation.evaluate_policy(model.policy, env))


# Stable Baselines3 SAC - LunarLanderContinuous-v2
# Decomment below to run SAC benchmarks

# if __name__ == "__main__":
#     with timer("run_time", SumMetric, sync_on_compute=False):
#         env = sb3.common.vec_env.DummyVecEnv(
#             [lambda: gym.make("LunarLanderContinuous-v2", render_mode="rgb_array") for _ in range(4)]
#         )
#         model = SAC("MlpPolicy", env, verbose=0, device="cpu")
#         model.learn(total_timesteps=1024 * 64, log_interval=None)
#     print(timer.compute())
#     print(sb3.common.evaluation.evaluate_policy(model.policy, env.envs[0]))
