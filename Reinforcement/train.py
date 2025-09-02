from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env('LunarLander-v3', n_envs=16)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000, progress_bar=True)

model_name = "ppo-LunarLander-v2"
model.save(model_name)