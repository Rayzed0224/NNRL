# 🧠 Portfolio Optimization via Heuristics and Reinforcement Learning

This project compares four algorithmic strategies to optimise stock portfolio weights using historical S&P 500 stock data.

### 🧪 Compared Methods
- **Gradient Descent (GD)**
- **Genetic Algorithm (GA)**
- **Particle Swarm Optimisation (PSO)**
- **Proximal Policy Optimisation (PPO)** (pre-trained)

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/Rayzed0224/NNRL.git

### 2. Install Dependencies

```bash
pip install -r requirements.txt

### 3. Generating CSV (Skip if data/historical_returns.csv exists)

```bash
python import_data.py

### 4. Running the script

```bash
python compare.py