import gymnasium as gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")

model_path = load_from_hub(
	repo_id="asalgad2/rl_lunar_lander",
	filename="ppo-LunarLander-v2_2.zip",
)

model = PPO.load(model_path, env=env)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()