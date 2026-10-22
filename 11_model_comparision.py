
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the actual and predicted values from the SARIMA test
df = pd.read_csv("sarima_test_predictions.csv")

actual = df["Actual_Sales"]
predicted = df["Predicted_Sales"]

# Calculate evaluation metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

# Calculate MAPE, avoiding division by zero
nonzero = actual != 0
if nonzero.any():
    mape = (
        np.abs(
            (actual[nonzero] - predicted[nonzero])
            / actual[nonzero]
        ).mean() * 100
    )
else:
    mape = np.nan

# Create a results table
results = pd.DataFrame({
    "Model": ["SARIMA"],
    "MAE": [mae],
    "RMSE": [rmse],
    "MAPE_Percent": [mape]
})

results.to_csv("model_comparison.csv", index=False)

print("\nMODEL EVALUATION RESULTS")
print(results.round(2).to_string(index=False))
print("\nSaved: model_comparison.csv")
