import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# LOAD DATASET
df = pd.read_csv("train.csv")

# SHOW FIRST 5 ROWS
print(df.head())

# SELECT FEATURES
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# TARGET COLUMN
y = df["SalePrice"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# CREATE MODEL
model = LinearRegression()

# TRAIN MODEL
model.fit(X_train, y_train)

# MAKE PREDICTIONS
predictions = model.predict(X_test)

# RESULTS
print("\n===== MODEL RESULTS =====")

print("\nR2 Score:", r2_score(y_test, predictions))

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# SAMPLE HOUSE PREDICTION
sample_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "FullBath": [2]
})

predicted_price = model.predict(sample_house)

print("\nPredicted House Price:", predicted_price[0])

# ---------------- GRAPH 1 ----------------

plt.figure(figsize=(8,5))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()

# ---------------- GRAPH 2 ----------------

plt.figure(figsize=(8,5))

plt.scatter(df["GrLivArea"], df["SalePrice"])

plt.xlabel("Living Area")

plt.ylabel("Sale Price")

plt.title("Living Area vs Sale Price")

plt.show()
