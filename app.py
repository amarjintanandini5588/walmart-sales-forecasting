
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.title("Walmart Sales Forecasting")
st.write("Time Series Forecasting for Retail Sales Prediction")

file_path = Path(__file__).parent / "train.csv"

if file_path.exists():
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"])

    st.subheader("Historical Weekly Sales")
    st.write(data.head())

    weekly_sales = data.groupby("Date")["Weekly_Sales"].sum().sort_index()

    fig, ax = plt.subplots()
    weekly_sales.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    ax.set_title("Walmart Sales Over Time")
    st.pyplot(fig)
else:
    st.error("Please upload train.csv to the repository.")
