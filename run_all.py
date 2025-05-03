import subprocess
import sys
import os

REQUIRED_PACKAGES = [
    "pandas", "numpy", "matplotlib", "seaborn", "scipy",
    "gymnasium", "stable-baselines3", "torch", "tqdm", "yfinance"
]

def ensure_pip():
    try:
        import pip
    except ImportError:
        print("Installing pip...")
        subprocess.run([sys.executable, "-m", "ensurepip"], check=True)

def install_packages():
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.split("[")[0])
        except ImportError:
            print(f"Installing: {pkg}")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

def run_script(path):
    print(f"\n▶ Running {path}")
    subprocess.run([sys.executable, path], check=True)

if __name__ == "__main__":
    ensure_pip()
    install_packages()

    run_script("evaluate_agent.py")
    run_script("compare.py")
