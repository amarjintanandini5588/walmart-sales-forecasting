
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

folder = Path(__file__).resolve().parent
train = pd.read_csv(folder / "train.csv")
train["Date"] = pd.to_datetime(train["Date"])

# Select Store 1, Department 1
STORE_ID = 1
DEPT_ID = 1

sales = train[
    (train["Store"] == STORE_ID) &
    (train["Dept"] == DEPT_ID)
].copy()

sales = sales.groupby("Date")["Weekly_Sales"].sum().sort_index()

print("Selected Store:", STORE_ID)
print("Selected Department:", DEPT_ID)
print("Weekly observations:", len(sales))
print("Date range:", sales.index.min(), "to", sales.index.max())

if len(sales) < 104:
    raise ValueError("At least 104 weekly observations are recommended.")

# Chronological train/test split
split = int(len(sales) * 0.8)
train_sales = sales.iloc[:split]
test_sales = sales.iloc[split:]

print("Training observations:", len(train_sales))
print("Testing observations:", len(test_sales))

# Fit seasonal ARIMA
print("\nFitting SARIMA model...")

model = SARIMAX(
    train_sales,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False, maxiter=100)

# Predict the held-out test period
predictions = results.get_forecast(
    steps=len(test_sales)
).predicted_mean

predictions.index = test_sales.index

mae = mean_absolute_error(test_sales, predictions)
rmse = mean_squared_error(test_sales, predictions) ** 0.5

print("\nSARIMA Evaluation")
print("-----------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot actual versus predicted sales
plt.figure(figsize=(12, 5))
plt.plot(train_sales.index, train_sales, label="Training sales")
plt.plot(test_sales.index, test_sales, label="Actual test sales")
plt.plot(predictions.index, predictions, label="SARIMA predictions")
plt.title("Walmart Sales: SARIMA Model")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()

graphs = folder / "graphs"
graphs.mkdir(exist_ok=True)

plt.savefig(graphs / "07_sarima_test.png", dpi=150)
plt.show()

# Refit on all historical data for the future forecast
print("\nGenerating four-week forecast...")

final_model = SARIMAX(
    sales,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_results = final_model.fit(disp=False, maxiter=100)
forecast = final_results.get_forecast(steps=4).predicted_mean

future_dates = pd.date_range(
    start=sales.index.max() + pd.Timedelta(days=7),
    periods=4,
    freq="7D"
)

forecast_table = pd.DataFrame({
    "Date": future_dates,
    "SARIMA_Forecast": forecast.to_numpy()
})

forecast_table.to_csv(
    folder / "sarima_four_week_forecast.csv",
    index=False
)

print(forecast_table.to_string(index=False))
print("\nSaved graph: graphs/07_sarima_test.png")
print("Saved forecast: sarima_four_week_forecast.csv")
