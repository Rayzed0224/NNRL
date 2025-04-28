# === preprocess_data.py ===
# Extract features from all_stocks_5yr.csv and save multivariate ticker-wise CSVs

import pandas as pd
import numpy as np
import os
import ta

# === Paths ===
INPUT_PATH = "data/all_stocks_5yr.csv"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(INPUT_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['Name', 'date'])

# === Feature Engineering per Ticker ===
all_tickers = df['Name'].unique()
print(f"[📊] Processing {len(all_tickers)} tickers")

for ticker in all_tickers:
    data = df[df['Name'] == ticker].copy()
    data = data.reset_index(drop=True)

    if len(data) < 200:
        print(f"[⚠️] Skipping {ticker} — too few rows")
        continue

    # Technical Indicators
    data['return'] = data['close'].pct_change()
    data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
    data['macd'] = ta.trend.MACD(data['close']).macd_diff()
    data['ema_20'] = ta.trend.EMAIndicator(data['close'], window=20).ema_indicator()
    data['sma_50'] = ta.trend.SMAIndicator(data['close'], window=50).sma_indicator()
    data['atr'] = ta.volatility.AverageTrueRange(high=data['high'], low=data['low'], close=data['close']).average_true_range()

    # Drop NA rows from indicator calc
    data = data.dropna().reset_index(drop=True)

    # Save to individual file
    output_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
    data.to_csv(output_path, index=False)
    print(f"[✅] Saved {ticker} → {output_path}")

print("[🎉] Preprocessing complete.")