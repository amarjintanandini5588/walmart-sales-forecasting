
import pandas as pd
from pathlib import Path

# Project folder
folder = Path(__file__).resolve().parent

# Load datasets
train = pd.read_csv(folder / "train.csv")
features = pd.read_csv(folder / "features.csv")

# Convert dates
train["Date"] = pd.to_datetime(train["Date"])
features["Date"] = pd.to_datetime(features["Date"])

# Merge sales with additional features
data = pd.merge(
    train,
    features,
    on=["Store", "Date", "IsHoliday"],
    how="left"
)

# Sort data by store, department, and date
data = data.sort_values(
    ["Store", "Dept", "Date"]
).reset_index(drop=True)

# Create date-based features
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Week"] = data["Date"].dt.isocalendar().week.astype(int)

# Create lagged sales features
data["Sales_Last_Week"] = (
    data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
)

data["Sales_4_Weeks_Ago"] = (
    data.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)
)

# Save prepared data
output_path = folder / "prepared_sales.csv"
data.to_csv(output_path, index=False)

print("Data preparation completed successfully!")
print("Prepared dataset shape:", data.shape)
print("Saved to:", output_path)
print("\nFirst 5 rows:")
print(data.head())
