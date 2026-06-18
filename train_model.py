import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load combined dataset
df = pd.read_csv("data/combined_tcs.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)

# Moving averages
df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

# Previous day's close
df["Prev_Close"] = df["Close"].shift(1)

# Remove rows with NaN (caused by rolling/shift)
df = df.dropna()

# Features & Target
features = ["Open", "High", "Low", "Volume", "MA50", "MA200", "Prev_Close"]
target = "Close"

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# Models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/rf_tcs_close.joblib")

# Save metrics
with open("models/metrics.txt", "w") as f:
    f.write(f"MAE: {mae}\nR2 Score: {r2}\n")

print("Model training complete!")
print("MAE:", mae)
print("R2 Score:", r2)
print("Model saved to models/rf_tcs_close.joblib")
