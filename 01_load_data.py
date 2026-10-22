from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent

print("Project folder:", folder)
print("train.csv exists:", (folder / "train.csv").exists())
print("features.csv exists:", (folder / "features.csv").exists())

try:
    train = pd.read_csv(folder / "train.csv")
    print("Train loaded successfully:", train.shape)
except Exception as e:
    print("Error loading train.csv:", repr(e))

try:
    features = pd.read_csv(folder / "features.csv")
    print("Features loaded successfully:", features.shape)
except Exception as e:
    print("Error loading features.csv:", repr(e))

