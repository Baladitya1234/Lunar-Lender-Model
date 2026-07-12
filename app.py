import gymnasium as gym
from stable_baselines3 import DQN

# Create environment
env = gym.make("LunarLander-v3", render_mode="human")

# Load trained model
model = DQN.load("models/lunarlander_model.zip")

episodes = 5

for episode in range(episodes):
    obs, info = env.reset()
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode+1}: Reward = {total_reward:.2f}")
            break

env.close()