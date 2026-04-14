import pandas as pd
import streamlit as st
import datetime
# Mock streamlit secrets for local execution if needed, or rely on .streamlit/secrets.toml
# Assuming we run this from root where .streamlit exists.

from src.core.database import load_fv_sala_nova_data

print("--- Starting Debug ---")
try:
    df = load_fv_sala_nova_data()
    print(f"Data Loaded. Type: {type(df)}")
    
    if df.empty:
        print("DataFrame is EMPTY!")
    else:
        print(f"DataFrame Shape: {df.shape}")
        print("Index Type:", df.index.dtype)
        print("Index Example:", df.index[0])
        print("Timezone Info:", df.index.tz)
        print("\nHead:\n", df.head())
        print("\nTail:\n", df.tail())
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("--- End Debug ---")
