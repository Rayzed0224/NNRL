from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from envs.portfolio_env import PortfolioEnv
import os

def train_agent(env, timesteps=100_000, model_path="models/dqn_agent.zip"):
    # Check if env is legit
    check_env(env, warn=True)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        tensorboard_log="./tensorboard/"
    )

    model.learn(total_timesteps=timesteps)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model
