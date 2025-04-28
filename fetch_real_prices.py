#!/usr/bin/env python
# fetch_real_prices.py  ────────────────────────────────────────────
"""
Download daily *adjusted* close for a list of tickers with yfinance,
save:
    data/real_returns.csv            log‑returns
    data/real_returns_train.csv      first 80 %
    data/real_returns_val.csv        last 20 %
"""

import argparse, os, sys, time
import pandas as pd, numpy as np, yfinance as yf

# ─────────────────── CLI ───────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+",
    default="AAPL MSFT AMZN NVDA GOOGL XOM UNH JNJ V BRK-B".split())
parser.add_argument("--years", type=int, default=5,
    help="history window in years")
parser.add_argument("--out_dir", default="data")
parser.add_argument("--val_frac", type=float, default=0.2)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ─────────────────── download loop ───────────────────
rows = []
for tkr in args.tickers:
    try:
        df = yf.download(
            ticker,
            period=f"{args.years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            raise_errors=False,
        )
        close = df["Adj Close"]
        if close.dropna().empty:
            raise RuntimeError("empty data")
        rows.append(close.rename(tkr))
        print(f"✓ {tkr:<6} rows={len(close)}")
        time.sleep(0.1)             # gentle on Yahoo
    except Exception as e:
        print(f"✗ {tkr:<6} skipped – {e}")

if not rows:
    sys.exit("‼  no data downloaded – check network / ticker list")

prices = pd.concat(rows, axis=1).ffill().dropna(how="all")

# ─────── returns & splits ───────
rets = np.log(prices / prices.shift(1)).dropna()

split = int(len(rets)*(1-args.val_frac))
rets.to_csv(f"{args.out_dir}/real_returns.csv",           index=False)
rets.iloc[:split].to_csv(f"{args.out_dir}/real_returns_train.csv", index=False)
rets.iloc[split:].to_csv(f"{args.out_dir}/real_returns_val.csv",   index=False)

print(f"✔ saved → {args.out_dir}/real_returns*.csv  rows={len(rets)}")
