import gymnasium as gym
from stable_baselines3 import DQN

# Create environment
env = gym.make("LunarLander-v3", render_mode="human")

# Load trained model
model = DQN.load("models/lunarlander_model")

obs, info = env.reset()

while True:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode Finished!")
        obs, info = env.reset()