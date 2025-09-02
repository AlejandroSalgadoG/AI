import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


env = gym.make("LunarLander-v3", render_mode="rgb_array")

model_name = "ppo-LunarLander-v2_2"
model = PPO.load(model_name, env=env)

eval_env = Monitor(env)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
