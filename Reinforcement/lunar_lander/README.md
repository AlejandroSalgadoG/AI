---
library_name: stable-baselines3
tags:
- LunarLander-v2
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
model-index:
- name: PPO
  results:
  - metrics:
    - type: mean_reward
      value: 262.40 +/- 24.94
      name: mean_reward
    task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: LunarLander-v2
      type: LunarLander-v2
---

# **PPO** Agent playing **LunarLander-v2**
This is a trained model of a **PPO** agent playing **LunarLander-v2** using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3).

## Usage (with Stable-baselines3)

```python
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
```
