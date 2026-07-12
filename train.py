import gymnasium as gym
from stable_baselines3 import DQN
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Create environment
env = gym.make("LunarLander-v3")

# Create DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1
)

print("Training Started...")

# Train the model
model.learn(total_timesteps=200000)

# Save model
model.save("models/lunarlander_model")

print("Training Completed!")
print("Model saved successfully.")