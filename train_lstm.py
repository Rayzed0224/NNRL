# === train_lstm.py ===
# Train LSTM model only for tickers that aren't already trained

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import LSTMModel
import os
import time

# === CONFIG ===
SEQ_LEN = 60
EPOCHS = 200
BATCH_SIZE = 32
LR = 0.0001
HIDDEN_SIZE = 512
NUM_LAYERS = 6
EARLY_STOPPING_PATIENCE = 20
INPUT_DIR = "data/processed"
MODEL_DIR = "models"
STATUS_PATH = "reports/status.txt"
FEATURES = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema_20', 'sma_50', 'atr']
TOP_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'XOM', 'UNH', 'JNJ', 'V', 'BRK.B']
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)

for ticker in TOP_TICKERS:
    model_path = f"{MODEL_DIR}/lstm_{ticker}.pth"
    if os.path.exists(model_path):
        print(f"[⏩] Model for {ticker} already exists, skipping training.")
        continue

    file_path = os.path.join(INPUT_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"[❌] Data missing for {ticker}, skipping...")
        continue

    df = pd.read_csv(file_path)
    if any(f not in df.columns for f in FEATURES):
        print(f"[❌] Missing features for {ticker}, skipping...")
        continue

    print(f"[⚙] Training LSTM for {ticker}...")
    with open(STATUS_PATH, "w") as f:
        f.write(f"LSTM Training: {ticker}")

    data = df[FEATURES].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN):
        X.append(scaled[i:i+SEQ_LEN])
        y.append(scaled[i+SEQ_LEN][FEATURES.index("close")])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    dataset = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel(input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    best_loss = float('inf')
    patience_counter = 0
    losses = []
    start_time = time.time()
    progress_path = f"reports/lstm_training_{ticker}.csv"

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for x_batch, y_batch in dataset:
            optimizer.zero_grad()
            output = model(x_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        print(f"{ticker} — Epoch {epoch} | Loss: {avg_loss:.6f}")

        # Update CSV
        pd.DataFrame(losses, columns=["Loss"]).to_csv(progress_path, index=False)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[🛑] Early stopping at epoch {epoch} for {ticker}")
                break

        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            eta = (EPOCHS - epoch) * (elapsed / epoch)
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            eta_str = f"{eta_min}m {eta_sec}s"
            print(f"[⏳] ETA: {eta_str}")
            with open(STATUS_PATH, "w") as f:
                f.write(f"LSTM Training: {ticker} | Epoch {epoch}/{EPOCHS} | ETA: {eta_str}")

print("[✅] All LSTM training complete.")
with open(STATUS_PATH, "w") as f:
    f.write("System Idle")
