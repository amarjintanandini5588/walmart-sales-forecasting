
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Locate the project folder
folder = Path(__file__).resolve().parent

# 2. Load the Walmart sales data
train = pd.read_csv(folder / "train.csv")
train["Date"] = pd.to_datetime(train["Date"])

# 3. Select Store 1, Department 1
data = train[
    (train["Store"] == 1) &
    (train["Dept"] == 1)
].copy()

# 4. Prepare weekly sales as a time series
weekly_sales = (
    data.groupby("Date")["Weekly_Sales"]
    .sum()
    .sort_index()
)

weekly_sales = weekly_sales.asfreq("7D")

# Fill any missing weekly values by time interpolation
weekly_sales = weekly_sales.interpolate(method="time")

# 5. Check if enough observations are available
if len(weekly_sales) < 104:
    raise ValueError(
        "At least 104 weekly observations are needed "
        "for yearly seasonal decomposition."
    )

# 6. Decompose sales using a 52-week seasonal period
result = seasonal_decompose(
    weekly_sales,
    model="additive",
    period=52
)

# 7. Plot the original data and decomposition components
fig = result.plot()
fig.set_size_inches(12, 9)
fig.suptitle(
    "Walmart Sales Time-Series Decomposition",
    fontsize=14,
    y=1.02
)
plt.tight_layout()

# 8. Save the graph
output_folder = folder / "graphs"
output_folder.mkdir(exist_ok=True)

plt.savefig(
    output_folder / "02_decomposition.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Decomposition completed successfully!")
print("Number of weekly observations:", len(weekly_sales))
print("Seasonal period: 52 weeks")
print("Graph saved to: graphs/02_decomposition.png")
