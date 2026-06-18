import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load combined dataset
df = pd.read_csv("data/combined_tcs.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)

# Create visuals folder
os.makedirs("visuals", exist_ok=True)

# Moving averages
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

# 1. Close Price Trend
plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Close"], label="Close Price")
plt.title("TCS Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.savefig("visuals/close_price_trend.png")
plt.close()

# 2. Volume Trend
plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Volume"], color='orange')
plt.title("TCS Volume Traded Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.savefig("visuals/volume_trend.png")
plt.close()

# 3. Moving Average (50 & 200)
plt.figure(figsize=(12,5))
plt.plot(df["Date"], df["Close"], label="Close Price")
plt.plot(df["Date"], df["MA50"], label="MA50")
plt.plot(df["Date"], df["MA200"], label="MA200")
plt.title("TCS Moving Averages (50 & 200)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig("visuals/moving_averages.png")
plt.close()

# 4. Daily Returns
df["Daily_Return"] = df["Close"].pct_change()

plt.figure(figsize=(12,5))
sns.histplot(df["Daily_Return"].dropna(), bins=50, kde=True)
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.savefig("visuals/daily_returns_hist.png")
plt.close()

print("All visuals exported successfully to /visuals folder.")
