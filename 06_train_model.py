
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project folder
folder = Path(__file__).resolve().parent

# Load prepared data
data = pd.read_csv(folder / "prepared_sales.csv")
data["Date"] = pd.to_datetime(data["Date"])

# Remove rows without previous sales history
data = data.dropna(
    subset=["Sales_Last_Week", "Sales_4_Weeks_Ago"]
).copy()

# Select input features
features = [
    "Store",
    "Dept",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Week",
    "Sales_Last_Week",
    "Sales_4_Weeks_Ago"
]

target = "Weekly_Sales"

# Convert holiday values to numbers
data["IsHoliday"] = data["IsHoliday"].astype(int)

# Use earlier dates for training and later dates for testing
split_date = data["Date"].quantile(0.8)

train_data = data[data["Date"] < split_date]
test_data = data[data["Date"] >= split_date]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Train the model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# Predict sales
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel evaluation results")
print("------------------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Save trained model
model_path = folder / "walmart_sales_model.pkl"
joblib.dump(model, model_path)

print("\nModel saved successfully:", model_path)
