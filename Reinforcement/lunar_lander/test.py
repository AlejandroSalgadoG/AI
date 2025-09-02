import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")

model_name = "ppo-LunarLander-v2_2"
model = PPO.load(model_name, env=env)

observation, info = env.reset(seed=42)
for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()