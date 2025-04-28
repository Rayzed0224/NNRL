# === train_rl.py (Merged: RewardLogger + EvalCallback + Val Split) ===
import os
import time
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList, BaseCallback
from envs.portfolio_env import PortfolioEnv

# === CONFIG ===
MODEL_PATH = "models/ppo_agent.zip"
TRAIN_CSV = "data/predicted_returns_train.csv"
VAL_CSV = "data/predicted_returns_val.csv"
TIMESTEPS = 300000
EARLY_STOPPING_PATIENCE = 500
REWARD_LOG = "reports/ppo_training_live.csv"
STATUS_PATH = "reports/status.txt"

os.makedirs("reports", exist_ok=True)

# === REWARD LOGGER CALLBACK ===
class RewardLoggerWithEarlyStop(BaseCallback):
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.last_best_reward = -float("inf")
        self.counter = 0
        self.rewards = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos or "episode" not in infos[0]:
            return True

        reward = infos[0]["episode"]["r"]
        timestep = self.model.num_timesteps
        self.rewards.append((timestep, reward))

        df = pd.DataFrame(self.rewards, columns=["Timesteps", "EvalReward"])
        df["SmoothedReward"] = df["EvalReward"].rolling(window=10, min_periods=1).mean()
        df["BestReward"] = df["EvalReward"].cummax()
        df.to_csv(REWARD_LOG, index=False)

        if reward > self.last_best_reward + 1e-4:
            self.last_best_reward = reward
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"[🛑] Early stopping at timestep {timestep}, reward stagnant.")
            return False

        if timestep % 10000 == 0:
            elapsed = time.time() - self.start_time
            eta = (TIMESTEPS - timestep) * (elapsed / timestep)
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            eta_str = f"{eta_min}m {eta_sec}s"
            print(f"[⏳] ETA: {eta_str}")
            with open(STATUS_PATH, "w") as f:
                f.write(f"PPO Training: {timestep}/{TIMESTEPS} | ETA: {eta_str}")

        return True

# === ENV FACTORIES ===
def make_train_env():
    return Monitor(PortfolioEnv(csv_path=TRAIN_CSV))

def make_val_env():
    return Monitor(PortfolioEnv(csv_path=VAL_CSV))

# === INIT ENVS ===
train_env = DummyVecEnv([make_train_env])
val_env = DummyVecEnv([make_val_env])

# === MODEL ===
model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log="tensorboard/ppo_best", device="cuda")

# === CALLBACKS ===
reward_logger = RewardLoggerWithEarlyStop()
eval_callback = EvalCallback(
    val_env,
    best_model_save_path="models/best/",
    log_path="reports/eval/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    callback_after_eval=StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=EARLY_STOPPING_PATIENCE,
        verbose=1
    )
)

# === TRAIN ===
print("[⚙] Using device:", "cuda" if torch.cuda.is_available() else "cpu")
print("[⚙] Training PPO agent with train/val split and early stopping...")
with open(STATUS_PATH, "w") as f:
    f.write("PPO Training started...")

model.learn(total_timesteps=TIMESTEPS, callback=CallbackList([reward_logger, eval_callback]))

model.save(MODEL_PATH)
print("[✅] PPO Agent training complete and saved.")
with open(STATUS_PATH, "w") as f:
    f.write("System Idle")