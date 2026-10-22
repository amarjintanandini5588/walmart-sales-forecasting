import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
folder=Path(__file__).resolve().parent
train=pd.read_csv(folder/ "train.csv")
print(train.columns.tolist())
train["Date"]=pd.to_datetime(train["Date"])

data = train[
    (train["Store"] == 1) &
    (train["Dept"] == 1)
].copy()

weekly_sales = data.groupby("Date")["Weekly_Sales"].sum().sort_index()

plt.figure(figsize=(12,5))
plt.plot(weekly_sales.index,weekly_sales.values)
plt.title("Walmart Weekly Sales - Store 1, Department 1")
plt.xlabel("Date")
plt.ylabel("Weekly Sales ($)")
plt.grid(True)
plt.tight_layout()
# Save the graph
output_folder = folder / "graphs"
output_folder.mkdir(exist_ok=True)
plt.savefig(
    output_folder / "01_raw_sales.png",
    dpi=150
)
plt.show()
print("Graph saved successfully!")
print("Number of weekly records:", len(weekly_sales))
print("Average weekly sales:", round(weekly_sales.mean(), 2))
print("Minimum weekly sales:", round(weekly_sales.min(), 2))
print("Maximum weekly sales:", round(weekly_sales.max(), 2))
