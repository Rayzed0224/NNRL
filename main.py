# === main.py (LSTM + RL w/ auto-train if no pretrained model exists) ===

import argparse
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import subprocess

from models.lstm_model import LSTMModel
from envs.portfolio_env import PortfolioEnv
from evaluate_agent import evaluate_agent

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--timesteps", type=int, default=50000)
parser.add_argument("--input", type=str, default="data/sp500_selected.csv")
parser.add_argument("--pred_output", type=str, default="data/predicted_returns.csv")
args = parser.parse_args()

# === 0. AUTO-TRAIN LSTM IF NOT FOUND ===
if not os.path.exists("models/lstm_weights.pth"):
    print("[!] No LSTM model found — training via train_lstm.py...")
    subprocess.run(["python", "train_lstm.py"])  # auto-train LSTM

# === 1. LOAD DATA ===
df = pd.read_csv(args.input)
ticker = df.columns[0]
prices = df[ticker].values.reshape(-1, 1)

# === 2. SCALE & SEQUENCE ===
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)
SEQ_LEN = 30

def make_sequences(data, seq_len):
    return np.array([data[i:i+seq_len] for i in range(len(data) - seq_len)])

X = make_sequences(scaled_prices, SEQ_LEN)

# === 3. LOAD LSTM MODEL ===
model = LSTMModel(input_size=1, hidden_size=args.hidden, num_layers=args.layers)
model.load_state_dict(torch.load("models/lstm_weights.pth"))
model.eval()

# === 4. MAKE PREDICTIONS ===
preds = []
with torch.no_grad():
    for seq in X:
        tensor_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        preds.append(model(tensor_seq).item())

pred_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
actual_prices = prices[SEQ_LEN:].flatten()

# === 5. SAVE PREDICTIONS ===
os.makedirs("data", exist_ok=True)
pd.DataFrame({"Actual": actual_prices, "Predicted": pred_prices}).to_csv(args.pred_output, index=False)
print(f"[✔] Saved predicted returns to {args.pred_output}")

# === 6. AUTO-TRAIN RL IF NOT FOUND ===
if not os.path.exists("models/dqn_agent.zip"):
    print("[!] No RL model found — training via train_rl.py...")
    subprocess.run(["python", "train_rl.py", "--csv", args.pred_output, "--timesteps", str(args.timesteps)])

# === 7. EVALUATION ===
evaluate_agent()