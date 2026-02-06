import pandas as pd
import os

folder = "data"
files = [f for f in os.listdir(folder) if f.endswith(".xlsx") and not f.startswith("~$")]
if files:
    fpath = os.path.join(folder, files[0])
    print(f"Reading {fpath}...")
    df = pd.read_excel(fpath)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
else:
    print("No .xlsx files found.")
