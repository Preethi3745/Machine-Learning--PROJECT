# Treasure Hunt AI using Kaggle Mine Dataset (Multiple Linear Regression)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_excel("Mine Dataset.xlsx")

print("Treasure (Mine) Dataset")
print(df.head())

# Rename columns for better project presentation
df.columns = ["VerticalSignal", "HorizontalSignal", "SensorNoise", "TreasureFound"]

print("\nRenamed Dataset")
print(df.head())

# Features (inputs) and Target (output)
X = df[["VerticalSignal", "HorizontalSignal", "SensorNoise"]]
y = df["TreasureFound"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nModel Performance")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature Impact
impact = pd.DataFrame(model.coef_, X.columns, columns=["Impact on Treasure Detection"])
print("\nFeature Impact")
print(impact)

# Visualization
plt.scatter(df["VerticalSignal"], df["TreasureFound"])
plt.xlabel("Vertical Magnetic Signal")
plt.ylabel("Treasure Found (0 or 1)")
plt.title("Magnetic Signal vs Treasure Detection")
plt.show()

# User Input Section
print("\nEnter Survey Data for Treasure Hunt")
v = float(input("Enter Vertical Magnetic Signal: "))
h = float(input("Enter Horizontal Magnetic Signal: "))
s = float(input("Enter Sensor Noise: "))

user_data = pd.DataFrame({
    "VerticalSignal": [v],
    "HorizontalSignal": [h],
    "SensorNoise": [s]
})

prediction = model.predict(user_data)

print("\nTreasure Probability Score:", prediction[0])

if prediction[0] > 0.5:
    print("💎 Treasure Likely Present in this Location!")
else:
    print("❌ Low Chance of Treasure Here.")
