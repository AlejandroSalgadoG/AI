import os
import gymnasium as gym

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="lunar_lander", name_prefix="replay")

model_name = "lunar_lander/ppo-LunarLander-v2_2"
model = PPO.load(model_name, env=env)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()

os.rename("lunar_lander/replay-episode-0.mp4", "lunar_lander/replay.mp4")