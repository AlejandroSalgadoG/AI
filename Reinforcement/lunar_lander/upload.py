import gymnasium as gym

from huggingface_hub import login
from huggingface_sb3 import package_to_hub
from stable_baselines3 import PPO

login()

env_id = "LunarLander-v3"

env = gym.make(env_id, render_mode="rgb_array")

model_name = "ppo-LunarLander-v2_2"
model = PPO.load(model_name, env=env)

package_to_hub(
    model=model,
    model_name=model_name,
    model_architecture="PPO",
    env_id=env_id,
    eval_env=env,
    repo_id="asalgad2/rl_lunar_lander",
    commit_message="commit lunar lander rl model",
)