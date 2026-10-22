
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

os.makedirs("graphs", exist_ok=True)

# 1. Load the Walmart sales data
train = pd.read_csv("train.csv")
train["Date"] = pd.to_datetime(train["Date"])

# 2. Select Store 1, Department 1
df = train[
    (train["Store"] == 1) &
    (train["Dept"] == 1)
].copy()

df = df.sort_values("Date")

# Combine duplicate dates, if any
series = df.groupby("Date")["Weekly_Sales"].sum().sort_index()

print("Number of weekly observations:", len(series))
print("First date:", series.index.min().date())
print("Last date:", series.index.max().date())

if len(series) < 60:
    raise ValueError(
        "Too few weekly observations for this forecast. "
        "Check train.csv and the selected Store/Department."
    )

# 3. Use the last 12 weeks for testing
test_size = 12
train_series = series.iloc[:-test_size]
test_series = series.iloc[-test_size:]

# 4. Fit SARIMA model
print("\nTraining SARIMA model...")

model = SARIMAX(
    train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fitted_model = model.fit(disp=False)

# 5. Evaluate test predictions
test_predictions = fitted_model.forecast(steps=test_size)
mae = mean_absolute_error(test_series, test_predictions)
rmse = np.sqrt(mean_squared_error(test_series, test_predictions))

print("\nSARIMA Test Results")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save test predictions
test_df = pd.DataFrame({
    "Date": test_series.index,
    "Actual_Sales": test_series.values,
    "Predicted_Sales": test_predictions.values
})
test_df.to_csv("sarima_test_predictions.csv", index=False)

# 6. Refit on all available data
print("\nFitting final model on all observations...")

final_model = SARIMAX(
    series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_fitted = final_model.fit(disp=False)

# 7. Forecast the next four weekly periods
forecast = final_fitted.get_forecast(steps=4)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

future_dates = pd.date_range(
    start=series.index.max() + pd.Timedelta(days=7),
    periods=4,
    freq="7D"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast_Sales": forecast_mean.values,
    "Lower_95": forecast_ci.iloc[:, 0].values,
    "Upper_95": forecast_ci.iloc[:, 1].values
})

forecast_df.to_csv("four_week_forecast.csv", index=False)

print("\nFour-week forecast:")
print(forecast_df.to_string(index=False))

# 8. Plot historical sales and forecast
plt.figure(figsize=(12, 6))
plt.plot(series.index, series.values, label="Historical Sales")
plt.plot(
    future_dates,
    forecast_mean.values,
    marker="o",
    label="Four-Week Forecast"
)
plt.fill_between(
    future_dates,
    forecast_ci.iloc[:, 0].values,
    forecast_ci.iloc[:, 1].values,
    alpha=0.2,
    label="95% Confidence Interval"
)
plt.title("Walmart Store 1 Department 1: Four-Week Forecast")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/10_final_forecast.png", dpi=300)
plt.show()

print("\nFiles saved:")
print("- sarima_test_predictions.csv")
print("- four_week_forecast.csv")
print("- graphs/10_final_forecast.png")
