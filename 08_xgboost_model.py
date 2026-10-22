
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

folder = Path(__file__).resolve().parent
train = pd.read_csv(folder / "train.csv")
features = pd.read_csv(folder / "features.csv")

train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

# Merge the two datasets
data = train.merge(
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

# Select Store 1, Department 1
data = data[
    (data["Store"] == 1) &
    (data["Dept"] == 1)
].copy()

data = data.sort_values("Date").reset_index(drop=True)

# Create calendar features
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Week"] = data["Date"].dt.isocalendar().week.astype(int)

# Create sales lag features
data["Lag_1"] = data["Weekly_Sales"].shift(1)
data["Lag_4"] = data["Weekly_Sales"].shift(4)
data["Lag_52"] = data["Weekly_Sales"].shift(52)

data["Rolling_Mean_4"] = data["Weekly_Sales"].shift(1).rolling(4).mean()

# Remove rows without enough historical data
data = data.dropna().copy()

feature_columns = [
    "Store", "Dept", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Year", "Month", "Week",
    "Lag_1", "Lag_4", "Lag_52", "Rolling_Mean_4"
]

X = data[feature_columns].copy()
y = data["Weekly_Sales"]

# Convert holiday flag to numeric
X["IsHoliday"] = X["IsHoliday"].astype(int)

# Chronological split: first 80% train, last 20% test
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training XGBoost model...")
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5

print("\nXGBoost Evaluation")
print("------------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Save actual and predicted test values
results = pd.DataFrame({
    "Date": data["Date"].iloc[split:].to_numpy(),
    "Actual_Sales": y_test.to_numpy(),
    "Predicted_Sales": predictions
})

results.to_csv(folder / "xgboost_test_predictions.csv", index=False)

# Plot actual versus predicted sales
plt.figure(figsize=(12, 5))
plt.plot(results["Date"], results["Actual_Sales"], label="Actual sales")
plt.plot(results["Date"], results["Predicted_Sales"], label="XGBoost predictions")
plt.title("Walmart Sales: XGBoost Test Predictions")
plt.xlabel("Date")
plt.ylabel("Weekly Sales ($)")
plt.legend()
plt.tight_layout()

graphs = folder / "graphs"
graphs.mkdir(exist_ok=True)
plt.savefig(graphs / "08_xgboost_test.png", dpi=150)
plt.show()

# Plot feature importance
importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values()

importance.plot(kind="barh", figsize=(9, 6))
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(graphs / "08_xgboost_importance.png", dpi=150)
plt.show()

print("\nSaved: xgboost_test_predictions.csv")
print("Saved graphs in the graphs folder.")
