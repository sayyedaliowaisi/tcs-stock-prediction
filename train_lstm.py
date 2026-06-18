import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

# Load datasetrg 
df = pd.read_csv("data/combined_tcs.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)

# Use only the Close price for LSTM
close_data = df["Close"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_data)

# Sequence length
SEQ_LEN = 60  # use last 60 days to predict next

X = []
y = []

for i in range(SEQ_LEN, len(scaled_close)):
    X.append(scaled_close[i-SEQ_LEN:i, 0])
    y.append(scaled_close[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    LSTM(50),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
model.fit(X, y, epochs=8, batch_size=32)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
model.save("models/lstm_tcs_close.h5")

# Save scaler
joblib.dump(scaler, "models/lstm_scaler.save")

print("LSTM training complete!")
print("Model saved to models/lstm_tcs_close.h5")
