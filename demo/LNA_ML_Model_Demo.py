
# LNA ML Modeling Demo - Python
# Dataset: Frequency, Material -> Gain, Noise
# Model: Random Forest Regression
# Libraries: pandas, numpy, sklearn, matplotlib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create dummy dataset
data = {
    "Frequency_GHz": [1.5, 2.4, 3.0, 4.5, 6.0, 8.0],
    "Material": ["GaAs", "GaAs", "GaN", "GaN", "GaAs", "GaN"],
    "Gain_dB": [14.8, 16.3, 18.2, 19.7, 13.5, 17.9],
    "Noise_dB": [0.8, 0.44, 1.2, 1.4, 1.0, 1.6]
}
df = pd.DataFrame(data)

# Encode categorical variable
le = LabelEncoder()
df["Material_encoded"] = le.fit_transform(df["Material"])

# Define features and targets
X = df[["Frequency_GHz", "Material_encoded"]]
y_gain = df["Gain_dB"]
y_noise = df["Noise_dB"]

# Train-test split
X_train, X_test, y_gain_train, y_gain_test = train_test_split(X, y_gain, test_size=0.3, random_state=42)
_, _, y_noise_train, y_noise_test = train_test_split(X, y_noise, test_size=0.3, random_state=42)

# Train models
gain_model = RandomForestRegressor(random_state=42)
noise_model = RandomForestRegressor(random_state=42)
gain_model.fit(X_train, y_gain_train)
noise_model.fit(X_train, y_noise_train)

# Predictions
y_gain_pred = gain_model.predict(X_test)
y_noise_pred = noise_model.predict(X_test)

# Evaluation
gain_mse = mean_squared_error(y_gain_test, y_gain_pred)
gain_r2 = r2_score(y_gain_test, y_gain_pred)
noise_mse = mean_squared_error(y_noise_test, y_noise_pred)
noise_r2 = r2_score(y_noise_test, y_noise_pred)

print(f"Gain - MSE: {gain_mse:.3f}, R2: {gain_r2:.3f}")
print(f"Noise - MSE: {noise_mse:.3f}, R2: {noise_r2:.3f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(y_gain_test.values, label="Actual Gain", marker='o')
plt.plot(y_gain_pred, label="Predicted Gain", marker='x')
plt.title("Gain Prediction")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_noise_test.values, label="Actual Noise", marker='o')
plt.plot(y_noise_pred, label="Predicted Noise", marker='x')
plt.title("Noise Prediction")
plt.legend()

plt.tight_layout()
plt.savefig("gain_noise_prediction.png")
plt.show()
