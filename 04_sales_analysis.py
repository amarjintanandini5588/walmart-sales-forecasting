
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the training dataset
folder = Path(__file__).resolve().parent
train = pd.read_csv(folder / "train.csv")

# Convert Date column to datetime
train["Date"] = pd.to_datetime(train["Date"])

# 1. Total weekly sales
weekly_sales = train.groupby("Date")["Weekly_Sales"].sum()

plt.figure(figsize=(12, 5))
weekly_sales.plot()
plt.title("Walmart Total Weekly Sales")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.tight_layout()
plt.savefig(folder / "graphs" / "04_weekly_sales.png")
plt.show()

# 2. Top 10 stores by total sales
store_sales = (
    train.groupby("Store")["Weekly_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Stores by Total Sales:")
print(store_sales.head(10))

# 3. Top 10 departments by total sales
department_sales = (
    train.groupby("Dept")["Weekly_Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Departments by Total Sales:")
print(department_sales.head(10))

print("\nSales analysis completed successfully!")
