
import pandas as pd
import os
import glob

def process_files(data_folder="data", output_file="huawei_combined_hourly.xlsx"):
    print("Looking for files...")
    # Get all .xlsx files in data folder
    files = glob.glob(os.path.join(data_folder, "huawei_hourly_*.xlsx"))
    
    if not files:
        print("No files found to process.")
        return

    print(f"Found {len(files)} files. Reading...")
    
    all_data = []
    
    for f in files:
        try:
            # Skip temp files
            if "~$" in f: continue
            
            # Read file, header is on row 1 (0-indexed)
            df = pd.read_excel(f, header=1)
            
            # Ensure columns exist
            if "Fecha" not in df.columns or "Salida de FV (kW)" not in df.columns:
                print(f"Skipping {f}: Invalid columns {df.columns.tolist()}")
                continue
                
            # Rename for clarity
            df = df.rename(columns={"Fecha": "Datetime", "Salida de FV (kW)": "Power_kW"})
            
            # Convert Datetime
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            
            # Filter valid dates if needed?
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No valid data loaded.")
        return

    # Combine all
    print("Merging data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort
    combined_df = combined_df.sort_values("Datetime")
    
    # Fill NaN with 0 (assuming NaN means no power generation)
    # Be careful not to fill missing timestamps, only NaNs in existing rows
    combined_df["Power_kW"] = combined_df["Power_kW"].fillna(0)
    
    # Set index
    combined_df = combined_df.set_index("Datetime")
    
    # Resample to Hourly
    # Logic: Sum / 12 for each hour
    print("Aggregating to hourly (Sum / 12)...")
    
    # We first group by Hour (floor)
    # We want strictly to group by '1H'. 
    # 'sum' is easier, then divide by 12.
    hourly_df = combined_df.resample("h").sum()
    
    # Apply the user's specific formula: Sum / 12
    hourly_df["Hourly_Power_kW"] = hourly_df["Power_kW"] / 12

    # Ensure continuous range (fill missing nights with 0)
    if not hourly_df.empty:
        # Full day range
        full_idx = pd.date_range(
            start=hourly_df.index.min().normalize(),
            end=hourly_df.index.max().normalize() + pd.Timedelta(hours=23),
            freq='h'
        )
        hourly_df = hourly_df.reindex(full_idx, fill_value=0)
        hourly_df.index.name = 'Datetime'
    
    # Clean up
    final_df = hourly_df[["Hourly_Power_kW"]]
    
    # Save to Excel
    print(f"Saving to {output_file}...")
    final_df.to_excel(output_file)
    
    # Save to CSV with comma decimals (User request)
    csv_output = output_file.replace(".xlsx", ".csv")
    print(f"Saving to {csv_output}...")
    final_df.to_csv(csv_output, sep=";", decimal=",", float_format="%.3f")
    
    print("Done!")
    print(final_df.head())

if __name__ == "__main__":
    process_files()
