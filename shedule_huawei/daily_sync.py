import sys
import os
import datetime
import time
import pandas as pd
from supabase import create_client

# Add parent directory to path to import huawei_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from huawei_client import HuaweiClient
except ImportError:
    print("Error: Could not import huawei_client. Make sure it exists in the parent directory.")
    sys.exit(1)

# --- Configuration ---
# Credentials from Environment Variables (GitHub Actions)
HUAWEI_USER = os.getenv("HUAWEI_USER")
HUAWEI_PASS = os.getenv("HUAWEI_PASS")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([HUAWEI_USER, HUAWEI_PASS, SUPABASE_URL, SUPABASE_KEY]):
    print("Error: Missing environment variables. Ensure HUAWEI_USER, HUAWEI_PASS, SUPABASE_URL, SUPABASE_KEY are set.")
    sys.exit(1)

STATION_NAME_FILTER = "Sala Nova" # Optional filter if user has multiple stations

def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def main():
    print(f"Starting Daily Sync for {datetime.datetime.now()}...")
    
    # 1. Calculate Yesterday
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    
    # Timestamp for Huawei API (Midnight of target day in milliseconds)
    # We want 00:00 of Yesterday
    ts = int(datetime.datetime.combine(yesterday, datetime.time(0,0)).timestamp() * 1000)
    print(f"Target Date: {yesterday} (TS: {ts})")

    # 2. Connect to Huawei
    client = HuaweiClient(HUAWEI_USER, HUAWEI_PASS)
    print("Logging in to FusionSolar...")
    if not client.login():
        print("Login Failed.")
        sys.exit(1)
    print("Login Successful.")

    # 3. Get Station
    print("Fetching Station List...")
    stations = client.get_station_list()
    if not stations:
        print("No stations found.")
        sys.exit(1)
        
    target_station = None
    if STATION_NAME_FILTER:
        for s in stations:
            if STATION_NAME_FILTER.lower() in s.get('stationName', '').lower():
                target_station = s
                break
    
    if not target_station:
        print(f"Station '{STATION_NAME_FILTER}' not found. Using first available: {stations[0].get('stationName')}")
        target_station = stations[0]
        
    station_code = target_station['stationCode']
    print(f"Selected Station: {target_station['stationName']} (Code: {station_code})")

    # 4. Fetch Hourly Data
    print(f"Fetching Hourly Data for {yesterday}...")
    data_hour = client.get_kpi_station_hour(station_code, ts)
    
    if not data_hour or not isinstance(data_hour, list):
        print("No data received or error in response.")
        # Only exit/error if strictly no list. Empty list might mean just no generation (unlikely for a whole day).
        if data_hour is None: sys.exit(1)
        
    print(f"Received {len(data_hour)} data points.")
    
    # 5. Process Data
    records = []
    for item in data_hour:
        # Extract Timestamp
        ms = item.get('collectTime', ts)
        dt_item = datetime.datetime.fromtimestamp(ms/1000)
        
        # Extract Value (Logic from app.py)
        val = 0.0
        map_data = item.get('dataItemMap', {})
        if 'inverter_power' in map_data: 
            raw_val = map_data['inverter_power']
            if raw_val is not None:
                val = float(raw_val)
            else:
                val = 0.0 # Default to 0 if explicit None
        elif 'productPower' in item and item['productPower'] is not None: 
            val = float(item['productPower'])
        elif 'productPower' in map_data and map_data['productPower'] is not None: 
            val = float(map_data['productPower'])
        
        # Prepare Record for Supabase (fv_sala_nova schema: reading_time, potencia_fv)
        records.append({
            "reading_time": dt_item.strftime("%Y-%m-%d %H:%M:%S"),
            "potencia_fv": val
        })

    if not records:
        print("No valid records extracted.")
        sys.exit(0)

    print(f"Prepared {len(records)} records for upload.")
    
    # 6. Upload to Supabase
    print("Uploading to Supabase (FV_Sala_Nova)...")
    supabase = init_supabase()
    
    try:
        # Batch upload just in case, though 24 records is small
        response = supabase.table("FV_Sala_Nova").upsert(records, on_conflict="reading_time").execute()
        print("Upload Successful!")
        # print(response)
    except Exception as e:
        print(f"Upload Error: {e}")
        sys.exit(1)

    print("Daily Sync Completed Successfully.")

if __name__ == "__main__":
    main()
