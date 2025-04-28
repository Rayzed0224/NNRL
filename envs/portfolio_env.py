import os, gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """
    PPO environment that lets the agent output a full weight vector.
    • action  : R^N, clipped to [0,1], normalised ⇒ % capital per ticker
    • obs     : rolling window of raw predicted returns (shape = window × N)
    • reward  : log‑change of equity → encourages compounding / risk‑aware
    • balance : starts at 1 000 and compounds every step
    """
    def __init__(
        self,
        csv_path       = "data/predicted_returns.csv",
        window_size    = 30,
        initial_balance= 1_000,
        weight_log     = "reports/ppo_weights.csv",
    ):
        super().__init__()

        # === data ===
        self.data   = pd.read_csv(csv_path)
        self.tickers= list(self.data.columns)
        self.n_assets = len(self.tickers)

        # === env params ===
        self.window_size     = window_size
        self.initial_balance = initial_balance
        self.weight_log      = weight_log
        self.max_steps       = len(self.data) - self.window_size - 2

        # === gym spaces ===
        self.action_space      = spaces.Box(0.0, 1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.n_assets),
            dtype=np.float32
        )

        self._reset_state(first=True)

    # -------------------------------------------------------------
    # core
    # -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def step(self, action):
        end_idx   = self.current_step + self.window_size
        terminated= end_idx >= len(self.data) - 1
        truncated = False  # no time‑limit truncation

        # -------- termination guard BEFORE touching data ----------
        if end_idx + 1 >= len(self.data):          # need t  and  t+1
            obs = self._get_observation()
            info = {"episode": {"r": self.episode_reward}}
            self.episode_reward = 0
            return obs, 0.0, True, False, info
        # -----------------------------------------------------------

        # --- weight vector ---
        w_raw     = np.clip(action, 0, 1)
        w_sum   = w_raw.sum()
        weights = w_raw / w_sum if w_sum > 1e-8 else np.full(self.n_assets, 1/self.n_assets)

        # --- asset returns between t and t+1 ---
        price_now  = self.data.iloc[end_idx     ].values   # shape (N,)
        price_next = self.data.iloc[end_idx + 1].values
        asset_ret  = (price_next - price_now) / price_now
        port_ret   = float(np.dot(weights, asset_ret))
        port_ret   = np.clip(port_ret, -0.05, 0.05)        # safety clamp

        # --- update equity & reward ---
        self.balance *= (1 + port_ret)
        reward        = np.log(self.balance / self.prev_balance)
        self.prev_balance = self.balance
        self.episode_reward += reward

        # --- logging weights ---
        self._log_weights(weights)

        # --- step ++ ---
        self.current_step += 1
        obs = self._get_observation()

        print(f"[RL] step {self.current_step}  R={reward:+.5f}  bal={self.balance:,.2f}")

        if terminated:
            info = {"episode": {"r": self.episode_reward}}
            self.episode_reward = 0
        else:
            info = {}

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------
    def _reset_state(self, first=False):
        self.current_step   = 0
        self.balance        = self.initial_balance
        self.prev_balance   = self.initial_balance
        self.episode_reward = 0

        # fresh weight log at every env reset of main training 🧽
        if first and os.path.exists(self.weight_log):
            os.remove(self.weight_log)

    def _get_observation(self):
        sl = slice(self.current_step, self.current_step + self.window_size)
        return self.data.iloc[sl].values.astype(np.float32)

    def _log_weights(self, weights):
        row = {"Time": self.current_step, **{t: w for t, w in zip(self.tickers, weights)}}
        pd.DataFrame([row]).to_csv(
            self.weight_log,
            mode="a", header=not os.path.exists(self.weight_log),
            index=False
        )

    def render(self):
        print(f"step={self.current_step}  balance={self.balance:,.2f}")
