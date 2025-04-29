# === dashboard.py ===
# Launch with: streamlit run dashboard.py --server.fileWatcherType none

import streamlit as st
import pandas as pd, os, subprocess, sys, threading
import numpy as np
import os
import time
from PIL import Image
import subprocess
import sys
import threading
from datetime import datetime
from evaluate_agent import evaluate_agent
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="NNRL Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.title("📊 Portfolio Optimization Dashboard")

# auto‑refresh every 10 000 ms (10 s)
def job_running():
    return os.path.exists("reports/status.txt")
st_autorefresh(interval=10_000, key="dash_refresh")

# ====================== util helpers FIRST 🡇🡇🡇 ======================
@st.cache_data(show_spinner=False)
def load_csv(path):   # <<< needs to live BEFORE we call it
    return pd.read_csv(path) if os.path.exists(path) else None

def load_image(path):
    return Image.open(path) if os.path.exists(path) else None
# =====================================================================

# === Live Task Status Header ===
st.subheader("System Status")
status_file = "reports/status.txt"
if os.path.exists(status_file):
    with open(status_file, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        current_status = lines[0] if lines else "System Idle"
    st.info(f"🟡 {current_status}")
else:
    st.success("✅ System Idle")

# === Live Loss Curve (LSTM) ===
st.subheader("LSTM Training Loss")
loss_files = sorted([f for f in os.listdir("reports") if f.startswith("lstm_training_") and f.endswith(".csv")])
if loss_files:
    latest = loss_files[-1]
    loss_data = pd.read_csv(f"reports/{latest}")
    st.line_chart(loss_data["Loss"], height=300)
else:
    st.warning("No LSTM training loss files found.")

# === Live Reward Curve (PPO) ===
st.subheader("PPO Training Reward")
reward_path = "reports/ppo_training_live.csv"
if os.path.exists(reward_path):
    reward_data = pd.read_csv(reward_path)
    if not reward_data.empty:
        current_timestep = reward_data["Timesteps"].iloc[-1]
        st.progress(min(current_timestep / 300000, 1.0), text=f"Timesteps: {current_timestep}/300000")
        st.line_chart(
            reward_data.set_index("Timesteps")[["EvalReward", "SmoothedReward", "BestReward"]],
            height=300
        )
    else:
        st.info("Waiting for PPO reward data...")
else:
    st.warning("No PPO reward logs yet.")

# === Equity, Drawdown and Sharpe ===
st.subheader("Live Equity, Drawdown & Rolling Sharpe")

curve = load_csv("reports/ppo_portfolio_curve.csv")  # columns: Time, Equity
if curve is not None:
    curve['Drawdown'] = 1 - curve['Equity'] / curve['Equity'].cummax()
    curve['Ret']      = curve['Equity'].pct_change()
    curve['RollSharpe'] = (
        curve['Ret'].rolling(60).mean() /
        curve['Ret'].rolling(60).std().replace(0,np.nan)
    ) * np.sqrt(60)

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(curve.set_index("Time")["Equity"], height=250)
        st.line_chart(curve.set_index("Time")["Drawdown"], height=250)
    with col2:
        st.line_chart(curve.set_index("Time")["RollSharpe"], height=250)
else:
    st.info("Run an evaluation to create portfolio curve.")

weights = load_csv("reports/ppo_weights.csv")  # Time, Ticker1, Ticker2, ...
if weights is not None:
    st.subheader("Allocation Heat‑Map")
    st.dataframe(weights.tail(20).set_index("Time"), use_container_width=True)
    st.line_chart(weights.set_index("Time"), height=200)


# === Manual Refresh Button ===
if st.sidebar.button("🔄 Manual Refresh"):
    st.rerun()

def write_status(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("reports/status.txt", "w") as f:
        f.write(f"{text} (Started at {timestamp})")

def clear_status():
    if os.path.exists("reports/status.txt"):
        os.remove("reports/status.txt")

def trigger_training(script):
    def run_script():
        write_status(f"Running {script}")
        subprocess.Popen([sys.executable, "-u", script])
        clear_status()
    threading.Thread(target=run_script).start()
    st.success(f"🚀 {script} launched in background")

def full_pipeline():
    write_status("Running Full Auto Pipeline (LSTM → Predict → PPO → Eval)")
    subprocess.run([sys.executable, "train_lstm.py"])
    subprocess.run([sys.executable, "predict_lstm.py"])
    subprocess.run([sys.executable, "train_rl.py"])
    subprocess.run([sys.executable, "evaluate_agent.py"])
    clear_status()

# === Sidebar Options ===
st.sidebar.header("Controls")
if st.sidebar.button("Train LSTM Only"):
    trigger_training("train_lstm.py")

if st.sidebar.button("Predict LSTM Only"):
    trigger_training("predict_lstm.py")

if st.sidebar.button("Train RL Agent Only"):
    trigger_training("train_rl.py")

if st.sidebar.button("Run Evaluation"):
    with st.spinner("Evaluating PPO agent..."):
        sharpe, mdd, _ = evaluate_agent(model_type="ppo", dashboard_mode=True)
        st.session_state.eval_sharpe = sharpe
        st.session_state.eval_mdd = mdd
        st.success("Evaluation complete!")

if st.sidebar.button("🚀 Full Auto Pipeline"):
    threading.Thread(target=full_pipeline).start()
    st.success("🚀 Full Auto Pipeline launched in background")

# === Tabs ===
tabs = st.tabs(["📉 LSTM", "🤖 RL Agent", "📈 Evaluation", "📁 Reports", "🧠 LSTM Predictions"])

# === Tab: LSTM ===
with tabs[0]:
    st.subheader("LSTM Training Progress")
    loss_data = load_csv("reports/lstm_training_live.csv")
    if loss_data is not None:
        latest = loss_data.groupby("Ticker").last().reset_index()
        for _, row in latest.iterrows():
            st.write(f"{row['Ticker']} — Epoch {row['Epoch']} | Loss: {row['Loss']:.6f}")
        st.line_chart(loss_data.pivot(index="Epoch", columns="Ticker", values="Loss"))

# === Tab: RL Agent ===
with tabs[1]:
    st.subheader("PPO Training Reward")
    reward_data = load_csv("reports/ppo_training_live.csv")
    if reward_data is not None and not reward_data.empty:
        current_timestep = reward_data["Timesteps"].iloc[-1]
        st.progress(min(current_timestep / 300000, 1.0), text=f"Timesteps: {current_timestep}/300000")
        st.line_chart(reward_data.set_index("Timesteps"))
    else:
        st.warning("No PPO training data yet. Run PPO or wait for log to generate.")

# === Tab: Evaluation ===
with tabs[2]:
    st.subheader("Portfolio Growth Evaluation")
    eval_plot = load_image("reports/ppo_evaluation_chart.png")
    eval_data = load_csv("reports/ppo_portfolio_curve.csv")

    if eval_plot:
        st.image(eval_plot, use_container_width=True)
    if eval_data is not None:
        st.line_chart(eval_data)

    if 'eval_sharpe' in st.session_state and 'eval_mdd' in st.session_state:
        st.metric("Sharpe Ratio", f"{st.session_state.eval_sharpe:.3f}")
        st.metric("Max Drawdown", f"{st.session_state.eval_mdd:.2%}")

# === Tab: Reports ===
with tabs[3]:
    st.subheader("Generated Report Files")
    files = os.listdir("reports") if os.path.exists("reports") else []
    for f in files:
        st.markdown(f"- [{f}](reports/{f})")

# === Tab: Predictions ===
with tabs[4]:
    st.subheader("Latest LSTM Predicted Returns")
    pred_df = load_csv("data/predicted_returns.csv")
    if pred_df is not None:
        st.write(pred_df.head(15))
        st.line_chart(pred_df)
        st.caption("Predicted returns for each ticker over the next 30 timesteps.")

