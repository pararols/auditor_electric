
import streamlit as st
import subprocess
import os
import pandas as pd
import datetime
from dotenv import load_dotenv
from supabase import create_client

# Load Env
load_dotenv()

# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATA_CSV = "huawei_combined_hourly.csv"

# --- Utils ---
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("Missing Supabase Credentials in .env")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def run_scraper(start_date):
    """Runs scraper.py as a subprocess"""
    cmd = ["python", "scraper.py", "--start_date", str(start_date), "--end_date", str(datetime.date.today())]
    
    with st.spinner(f"Running Scraper from {start_date}... Check terminal for logs."):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Scraper finished successfully!")
            st.text_area("Scraper Output", result.stdout, height=200)
        else:
            st.error("Scraper Failed")
            st.text_area("Scraper Error", result.stderr, height=200)
            
def run_processor():
    """Runs process_data.py"""
    with st.spinner("Processing Data..."):
        result = subprocess.run(["python", "process_data.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Data Processing Complete!")
            st.text_area("Processor Output", result.stdout, height=200)
        else:
            st.error("Processing Failed")
            st.text_area("Processor Error", result.stderr, height=200)

def upload_to_supabase():
    """Uploads CSV to FV_Sala_Nova"""
    if not os.path.exists(DATA_CSV):
        st.error(f"File {DATA_CSV} not found. Run Processor first.")
        return

    supabase = init_supabase()
    if not supabase: return

    # Read CSV (European format: sep=; decimal=,)
    df = pd.read_csv(DATA_CSV, sep=';', decimal=',')
    
    # Columns expected: Datetime, Hourly_Power_kW
    # Target Columns: reading_time, potencia_fv
    
    st.write("Preview of Data to Upload:")
    st.dataframe(df.head())
    
    if st.button("Confirm Upload 🚀"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare Data
        # Drop NaNs?
        df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='coerce')
        
        # Drop invalid dates (NaT)
        invalid_mask = df['Datetime'].isna()
        if invalid_mask.any():
            dropped_rows = df[invalid_mask]
            st.warning(f"⚠️ Dropping {len(dropped_rows)} rows with invalid dates:")
            st.dataframe(dropped_rows)
            
            df = df.dropna(subset=['Datetime'])

        records = []
        for _, row in df.iterrows():
            # Match format from app.py: '%Y-%m-%d %H:%M:%S'
            records.append({
                "reading_time": row['Datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                "potencia_fv": row['Hourly_Power_kW']
            })
            
        # Batch Upload
        chunk_size = 500
        total = len(records)
        
        for i in range(0, total, chunk_size):
            chunk = records[i:i+chunk_size]
            try:
                # Upsert based on reading_time (unique constraint)
                supabase.table("FV_Sala_Nova").upsert(chunk, on_conflict="reading_time").execute()
                
                prog = min((i + len(chunk)) / total, 1.0)
                progress_bar.progress(prog)
                status_text.text(f"Uploaded {i + len(chunk)}/{total}")
            except Exception as e:
                st.error(f"Error uploading chunk {i}: {e}")
                return
                
        status_text.success("Upload Complete! used table: FV_Sala_Nova")

# --- UI ---
st.title("☀️ Huawei Data Manager")

st.markdown("### 1. Scrape Data")
start_date = st.date_input("Start Date", value=datetime.date(2024, 12, 4))
if st.button("Run Scraper"):
    run_scraper(start_date)

st.markdown("### 2. Process Data")
st.caption("Combines daily files into one CSV")
if st.button("Run Processor"):
    run_processor()

st.markdown("### 3. Upload to Supabase")
st.caption(f"Uploads {DATA_CSV} to 'FV_Sala_Nova'")
upload_to_supabase()

st.divider()
st.info("Check terminal for detailed logs during scraping.")
