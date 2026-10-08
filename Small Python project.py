# LSTM Time-Series Prediction
# Predict the next value using previous time-series values

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# --------------------------------------------------
# 1. Create sample time-series data
# --------------------------------------------------

data = np.sin(np.arange(0, 100, 0.1))

data = data.reshape(-1, 1)


# --------------------------------------------------
# 2. Scale the data
# --------------------------------------------------

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(data)


# --------------------------------------------------
# 3. Create sequences
# --------------------------------------------------

sequence_length = 50

X = []
y = []

for i in range(sequence_length, len(scaled_data)):

    X.append(scaled_data[i-sequence_length:i, 0])

    y.append(scaled_data[i, 0])


X = np.array(X)
y = np.array(y)


# --------------------------------------------------
# 4. Reshape for LSTM
# --------------------------------------------------
# LSTM input shape:
# samples, timesteps, features

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)


# --------------------------------------------------
# 5. Build LSTM model
# --------------------------------------------------

model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(X.shape[1], 1)
    )
)

model.add(
    LSTM(50)
)

model.add(
    Dense(1)
)


# --------------------------------------------------
# 6. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# --------------------------------------------------
# 8. Prediction
# --------------------------------------------------

predictions = model.predict(X)


# Convert predictions back to original scale

predictions = scaler.inverse_transform(
    predictions
)


actual = scaler.inverse_transform(
    y.reshape(-1, 1)
)


# --------------------------------------------------
# 9. Plot results
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    actual,
    label="Actual"
)

plt.plot(
    predictions,
    label="Predicted"
)

plt.title(
    "LSTM Time-Series Prediction"
)

plt.xlabel("Time")

plt.ylabel("Value")

plt.legend()

plt.show()


# --------------------------------------------------
# 10. Training loss
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.show()
