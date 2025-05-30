# === evaluate_agent.py (Auto-detect PPO or DQN + Visuals + Report) ===

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
from scipy.optimize import minimize
import seaborn as sns
#from envs.portfolio_env import PortfolioEnv
from envs.portfolio_env_real import PortfolioEnvReal
from stable_baselines3 import PPO, DQN

print("PYTHON PATH:", sys.executable)

MODEL_PATHS = {
    "ppo": "models/ppo_agent.zip",
    "dqn": "models/dqn_agent.zip"
}

OUT_DIR = "results";   os.makedirs(OUT_DIR, exist_ok=True)

def evaluate_agent(model_type="ppo",
                   #csv_path="data/predicted_returns.csv",
                   csv_path="data/historical_returns.csv",
                   dashboard_mode=False,
                   steps=1000,
                   roll=60):
    assert model_type in MODEL_PATHS, "bad model type"
    model = (PPO if model_type=="ppo" else DQN).load(MODEL_PATHS[model_type], device='cpu')
    #env   = PortfolioEnv(csv_path=csv_path)
    env = PortfolioEnvReal(csv_path="data/historical_prices.csv")

    obs,_ = env.reset(); balances=[env.balance]
    for _ in tqdm(range(steps), desc="eval"):
        act,_ = model.predict(obs)
        act = act if isinstance(env.action_space, gym.spaces.Box) else int(np.squeeze(act))
        obs,_,done,_,_ = env.step(act)
        balances.append(env.balance)
        if done: break

    bal = pd.Series(balances, name="Equity")
    ret = bal.pct_change().dropna();  roll_sharpe = (
        ret.rolling(roll).mean() / ret.rolling(roll).std()).dropna()*np.sqrt(roll)

    dd   = 1 - bal / bal.cummax()
    stats = pd.DataFrame({
        "FinalBalance":[bal.iloc[-1]],
        "CAGR":[(bal.iloc[-1]/bal.iloc[0])**(252/len(bal))-1],
        "Sharpe":[ret.mean()/ret.std()*np.sqrt(252)],
        "MaxDD":[dd.max()]
    })
    stats.to_csv(f"{OUT_DIR}/{model_type}_stats.csv",index=False)

    # -------- plots --------
    plt.figure(figsize=(9,4)); bal.plot(); plt.title("Equity Curve"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{model_type}_equity.png"); plt.close()

    plt.figure(figsize=(9,3)); dd.plot(color="red"); plt.title("Drawdown"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{model_type}_drawdown.png"); plt.close()

    plt.figure(figsize=(9,3)); roll_sharpe.plot(color="green"); plt.title(f"Rolling{roll} Sharpe"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{model_type}_roll_sharpe.png"); plt.close()

    plt.figure(figsize=(6,4)); plt.hist(ret, bins=50); plt.title("Return Histogram"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{model_type}_ret_hist.png"); plt.close()

    plot_efficient_frontier(pred_csv=csv_path,
        weight_log="reports/ppo_weights.csv",
        out_path=f"results/{model_type}_efficient_frontier.png")

    if not dashboard_mode:
        print(stats.T)

# ---------------------------------------------------------------------
def plot_efficient_frontier(pred_csv      ="data/predicted_returns.csv",
                            weight_log    ="reports/ppo_weights.csv",
                            out_path      ="results/efficient_frontier.png",
                            n_sims        =20000,
                            risk_free     =0.0,
                            ann_factor    =252):
    """Monte‑Carlo frontier using predicted daily returns (shape T×N)"""
    import matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns
    sns.set(style="whitegrid"); plt.figure(figsize=(10,6))

    # ---- asset stats ----
    df   = pd.read_csv(pred_csv)
    mu   = df.mean()               # daily exp. return
    cov  = df.cov()                # daily covariance
    tick = df.columns
    ann_mu  = mu * 252             # annual
    ann_cov = cov * 252

    # ---- simulate portfolios ----
    rets, vols, sharpes, W = [], [], [], []
    for _ in range(n_sims):
        w = np.random.dirichlet(np.ones(len(tick)))
        r = np.dot(w, mu)
        v = np.sqrt(np.dot(w, np.dot(cov, w)))
        s = (r - risk_free) / v
        rets.append(r); vols.append(v); sharpes.append(s); W.append(w)

    rets, vols, sharpes = map(np.array, (rets, vols, sharpes))
    idx_sr  = sharpes.argmax()
    idx_var = vols.argmin()
    

    # ---- efficient‑frontier spline (quadratic fit to top 2 %) ----
    top = sharpes.argsort()[-int(0.02*n_sims):]
    z   = np.polyfit(vols[top], rets[top], 2); p = np.poly1d(z)
    xs  = np.linspace(vols.min(), vols.max(), 100)
    plt.plot(xs, p(xs), "--k", lw=2, label="Efficient Frontier")

    # ---- scatter portfolios ----
    sc = plt.scatter(vols, rets, c=sharpes, cmap="coolwarm", s=6, alpha=.6)

    # ---- highlight key pts ----
    plt.scatter(vols[idx_var], rets[idx_var], 150, marker="v", c="lime", edgecolors="k",
                label="EF min Volatility")
    plt.scatter(vols[idx_sr],  rets[idx_sr], 150, marker="^", c="red",  edgecolors="k",
                label="EF max Sharpe Ratio")

    # individual assets
    for t in tick:
        plt.scatter(cov.loc[t,t]**0.5, mu[t], 120, c="dodgerblue", edgecolors="k")
        plt.text(cov.loc[t,t]**0.5, mu[t], f" {t}", va="center", weight="bold")

    # agent’s final allocation (if weight csv exists)
    if os.path.exists(weight_log):
        w_df  = pd.read_csv(weight_log)
        w_ini = w_df.iloc[0][tick].values     # <‑‑ first allocation
        r_0   = np.dot(w_ini, mu)
        v_0   = np.sqrt(np.dot(w_ini, cov @ w_ini))
        plt.scatter(v_0, r_0, 160, marker="^", c="black",
            edgecolors="k", label="Initial Portfolio")

    cbar = plt.colorbar(sc); cbar.set_label("Sharpe Ratio (annualised)")
    plt.xlabel("Volatility (σ)"); plt.ylabel("Expected Return (μ)")
    plt.title("Portfolio Optimisation")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()
# ---------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_agent("ppo")
# =====================================================================

# === OPTIONAL: Risk Aversion Sweep Analysis ===
def objective_with_transaction_costs(w, prev_w, returns, cov, alpha):
    ret = np.dot(w, returns.mean())
    risk = np.dot(w, np.dot(cov, w))
    cost = np.sum(np.abs(w - prev_w)) * 0.001
    return -(ret - alpha * risk - cost)

def analyze_risk_aversion(returns, cov, risk_aversion_values, n_assets):
    results = []
    for alpha in risk_aversion_values:
        w = np.random.rand(n_assets)
        w = w / w.sum()
        prev_w = w.copy()

        result = minimize(
            lambda w: -objective_with_transaction_costs(w, prev_w, returns, cov, alpha),
            w,
            method='SLSQP',
            bounds=[(0.05, 1) for _ in range(n_assets)],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )

        if result.success:
            opt_w = result.x / result.x.sum()
            ann_ret = np.dot(opt_w, returns.mean()) * 252
            ann_cov = cov * 252
            risk = np.sqrt(np.dot(opt_w, np.dot(ann_cov, opt_w)))
            sharpe = (ann_ret - 0.02) / risk
            results.append({
                "Alpha": alpha,
                "Return": ann_ret,
                "Risk": risk,
                "Sharpe": sharpe,
                **{f"W_{i}": w_i for i, w_i in enumerate(opt_w)}
            })

    return pd.DataFrame(results)

# === RUN RISK ANALYSIS ===
try:
    df_returns = pd.read_csv("data/historical_returns.csv")
    cov = df_returns.cov()
    alphas = [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    df_risk = analyze_risk_aversion(df_returns, cov, alphas, df_returns.shape[1])
    df_risk.to_csv("results/risk_aversion_analysis.csv", index=False)
    print("\n--- Risk Aversion Sweep Complete ---")
    print(df_risk[["Alpha", "Return", "Risk", "Sharpe"]])
except Exception as e:
    print(f"Risk sweep failed: {e}")
