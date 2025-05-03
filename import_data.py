import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'XOM', 'UNH', 'JNJ', 'V', 'BRK-B']

# download with auto_adjust=True so you get clean prices
data = yf.download(tickers, start='2013-01-01', end='2018-12-31', auto_adjust=True)

# extract price data
adj_close = data['Close']  # use this if auto_adjust=True
# or use data['Adj Close'] if auto_adjust=False

# compute returns
returns = adj_close.pct_change().dropna()

# save to CSV
returns.to_csv("data/historical_returns.csv", index=False)
print("✅ Saved: data/historical_returns.csv")


# Load historical returns
returns = pd.read_csv("data/historical_returns.csv")

# Generate synthetic price series starting from 100
prices = (1 + returns).cumprod() * 100

# Save to data folder
prices.to_csv("data/historical_prices.csv", index=False)

print("✅ Saved: data/historical_prices.csv")