
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the datasets
train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")

# 2. Merge sales and features
df = train.merge(
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

# 3. Select Store 1, Department 1
df = df[(df["Store"] == 1) & (df["Dept"] == 1)].copy()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# 4. Create date-based features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsHoliday"] = df["IsHoliday"].astype(int)

# Use only past sales to predict future sales
df["Lag_1"] = df["Weekly_Sales"].shift(1)
df["Lag_4"] = df["Weekly_Sales"].shift(4)
df["Lag_52"] = df["Weekly_Sales"].shift(52)
df["Rolling_Mean_4"] = df["Weekly_Sales"].shift(1).rolling(4).mean()

df = df.dropna().reset_index(drop=True)

features_list = [
    "Year", "Month", "Week", "IsHoliday",
    "Lag_1", "Lag_4", "Lag_52", "Rolling_Mean_4"
]

# 5. Six expanding-window validation folds,
# each testing the next four weeks
initial_train_end = len(df) - (6 * 4)

if initial_train_end < 10:
    raise ValueError(
        f"Not enough data for validation. Available rows: {len(df)}"
    )
actual_all = []
predicted_all = []
fold_results = []

for fold in range(6):
    train_end = initial_train_end + fold * 4
    test_end = train_end + 4

    train_fold = df.iloc[:train_end]
    test_fold = df.iloc[train_end:test_end]

    X_train = train_fold[features_list]
    y_train = train_fold["Weekly_Sales"]

    # Recursive forecast: future lag features use prior predictions
    history = list(train_fold["Weekly_Sales"].values)
    predictions = []

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    for _, row in test_fold.iterrows():
        recent = np.array(history)

        lag_1 = recent[-1]
        lag_4 = recent[-4]
        lag_52 = recent[-52:]
        lag_52_value = lag_52[0] if len(lag_52) == 52 else lag_1
        rolling_mean_4 = np.mean(recent[-4:])

        X_next = pd.DataFrame([{
            "Year": row["Year"],
            "Month": row["Month"],
            "Week": row["Week"],
            "IsHoliday": row["IsHoliday"],
            "Lag_1": lag_1,
            "Lag_4": lag_4,
            "Lag_52": lag_52_value,
            "Rolling_Mean_4": rolling_mean_4
        }])

        prediction = float(model.predict(X_next)[0])
        predictions.append(prediction)
        history.append(prediction)

    actual = test_fold["Weekly_Sales"].values
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    fold_results.append({
        "Fold": fold + 1,
        "Test_Start": test_fold["Date"].min().date(),
        "Test_End": test_fold["Date"].max().date(),
        "MAE": mae,
        "RMSE": rmse
    })

    actual_all.extend(actual)
    predicted_all.extend(predictions)

    print(
        f"Fold {fold + 1}: "
        f"MAE = {mae:.2f}, RMSE = {rmse:.2f}"
    )

# 6. Save fold-level results
results = pd.DataFrame(fold_results)
results.to_csv("walk_forward_results.csv", index=False)

# 7. Overall validation metrics
overall_mae = mean_absolute_error(actual_all, predicted_all)
overall_rmse = np.sqrt(
    mean_squared_error(actual_all, predicted_all)
)

print("\nOverall Walk-Forward Results")
print(f"MAE:  {overall_mae:.2f}")
print(f"RMSE: {overall_rmse:.2f}")

# 8. Save actual versus predicted results
predictions_df = pd.DataFrame({
    "Actual_Sales": actual_all,
    "Predicted_Sales": predicted_all
})
predictions_df.to_csv(
    "walk_forward_predictions.csv",
    index=False
)

# 9. Plot results
plt.figure(figsize=(12, 6))
plt.plot(actual_all, label="Actual Sales")
plt.plot(predicted_all, label="Predicted Sales")
plt.title("XGBoost: Six-Fold Walk-Forward Validation")
plt.xlabel("Test Week")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/09_walk_forward_validation.png", dpi=300)
plt.show()

print("\nSaved:")
print("- walk_forward_results.csv")
print("- walk_forward_predictions.csv")
print("- graphs/09_walk_forward_validation.png")
