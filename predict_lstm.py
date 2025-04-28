# === predict_lstm.py (Return-Based + Variability + Train/Val Split) ===

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import LSTMModel
import os
import time

# === CONFIG ===
SEQ_LEN = 60
FUTURE_STEPS = 1150
TRAIN_SPLIT_INDEX = 850
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema_20', 'sma_50', 'atr']
INPUT_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_PATH = "data/predicted_returns.csv"
TRAIN_PATH = "data/predicted_returns_train.csv"
VAL_PATH = "data/predicted_returns_val.csv"
TOP_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'XOM', 'UNH', 'JNJ', 'V', 'BRK.B']
ticker_files = [f"{ticker}.csv" for ticker in TOP_TICKERS if os.path.exists(os.path.join(INPUT_DIR, f"{ticker}.csv"))]
os.makedirs("data", exist_ok=True)

all_preds = {}

for file in ticker_files:
    ticker = file.replace(".csv", "")
    model_path = f"{MODEL_DIR}/lstm_{ticker}.pth"
    if not os.path.exists(model_path):
        print(f"[⏩] Model not found for {ticker}, skipping prediction.")
        continue

    print(f"[🔮] Predicting for {ticker}...")

    df = pd.read_csv(os.path.join(INPUT_DIR, file))
    if any(f not in df.columns for f in FEATURES):
        print(f"[❌] Missing features for {ticker}, skipping...")
        continue

    prices = df[FEATURES].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    model = LSTMModel(input_size=len(FEATURES), hidden_size=512, num_layers=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    last_seq = torch.tensor(scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
    pred_returns = []
    start_time = time.time()

    with torch.no_grad():
        for step in range(FUTURE_STEPS):
            out = model(last_seq)
            raw_return = out.item()

            noise = np.random.normal(0, 0.0025)  # ~0.25% volatility
            adjusted_return = raw_return + noise

            close_idx = FEATURES.index("close")
            prev_close = last_seq[0, -1, close_idx].item()
            next_close = prev_close * (1 + adjusted_return)

            tmp_input = scaler.inverse_transform(last_seq[0, -1, :].cpu().numpy().reshape(1, -1))
            tmp_input[0, close_idx] = next_close
            scaled_next = scaler.transform(tmp_input)[0]

            next_row = last_seq[:, -1, :].clone()
            next_row[0] = torch.tensor(scaled_next, dtype=torch.float32)

            last_seq = torch.cat([last_seq[:, 1:, :], next_row.unsqueeze(1)], dim=1)
            pred_returns.append(adjusted_return)

            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                eta = (FUTURE_STEPS - step - 1) * (elapsed / (step + 1))
                print(f"[⏳] Step {step + 1}/{FUTURE_STEPS} | ETA: {eta:.2f}s")

    all_preds[ticker] = pred_returns

pred_df = pd.DataFrame(all_preds)
pred_df.to_csv(OUTPUT_PATH, index=False)
pred_df.iloc[:TRAIN_SPLIT_INDEX].to_csv(TRAIN_PATH, index=False)
pred_df.iloc[TRAIN_SPLIT_INDEX:].to_csv(VAL_PATH, index=False)
print(f"[✅] Saved full to {OUTPUT_PATH}, train to {TRAIN_PATH}, val to {VAL_PATH}")