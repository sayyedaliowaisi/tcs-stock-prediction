import pandas as pd

# File paths
info_path = r"C:\Users\acer\Downloads\TCS_stock_info.csv"
history_path = r"C:\Users\acer\Downloads\TCS_stock_history.csv"
action_path = r"C:\Users\acer\Downloads\TCS_stock_action.csv"

print("Loading CSV files...")

df_info = pd.read_csv(info_path)
df_history = pd.read_csv(history_path)
df_action = pd.read_csv(action_path)

print("Files loaded successfully!")

# Clean history data (main OHLC stock data)
df_history['Date'] = pd.to_datetime(df_history['Date'])
df_history.sort_values('Date', inplace=True)

# Combine everything
df_combined = df_history.copy()

# Add info file (static data)
for col in df_info.columns:
    if col not in df_combined.columns:
        df_combined[col] = df_info[col].iloc[0]

# Add action data (dividends, splits)
df_action['Date'] = pd.to_datetime(df_action['Date'])
df_combined = df_combined.merge(df_action, on="Date", how="left")

# Fill missing values
df_combined.fillna(0, inplace=True)

# Save final dataset
output_path = r"data\combined_tcs.csv"
df_combined.to_csv(output_path, index=False)

print(f"Combined dataset saved as: {output_path}")
print(df_combined.head())
print(df_combined.info())
