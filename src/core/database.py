import streamlit as st
import pandas as pd
from supabase import create_client, Client

# --- Supabase Initialization ---
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error("❌ No s'han trobat els secrets de Supabase. Configuració necessària a .streamlit/secrets.toml")
        return None

# --- Data Fetching Helpers ---

@st.cache_data(ttl=600, show_spinner=False)
def fetch_fv_data_chunked(start_date=None, end_date=None, chunk_size=1000):
    """Fetches FV_Sala_Nova data using pagination to bypass row limits."""
    client = init_supabase()
    if not client: return []
    
    all_rows = []
    offset = 0
    
    while True:
        q = client.table("FV_Sala_Nova").select("*").order("reading_time")
        if start_date:
            q = q.filter("reading_time", "gte", start_date)
        if end_date:
            q = q.filter("reading_time", "lte", end_date)
            
        res = q.range(offset, offset + chunk_size - 1).execute()
        if not res.data: break
        
        all_rows.extend(res.data)
        if len(res.data) < chunk_size: break
        offset += chunk_size
        if offset > 100000: break # Safety brake
        
    return all_rows

def load_from_supabase_db(start_date=None, end_date=None):
    """Fetch readings from 'energy_readings_wide' (JSONB optimized format)."""
    supabase = init_supabase()
    if not supabase: return None
    
    all_rows = []
    chunk_size = 1000
    offset = 0
    
    try:
        while True:
            query = supabase.table("energy_readings_wide").select("*").order("reading_time")
            if start_date:
                query = query.filter("reading_time", "gte", start_date)
            if end_date:
                query = query.filter("reading_time", "lte", end_date)
            
            res = query.range(offset, offset + chunk_size - 1).execute()
            if not res.data:
                break
            
            all_rows.extend(res.data)
            if len(res.data) < chunk_size:
                break
            offset += chunk_size
            if offset > 200000: break # Safety brake (approx 20 years of hourly data)
            
        if not all_rows: return None
        
        raw_df = pd.DataFrame(all_rows)
        
        # Expand JSONB 'data' column
        data_df = pd.json_normalize(raw_df['data'])
        # Handle DateTime and Timezones
        dt_index = pd.DatetimeIndex(raw_df['reading_time'])
        if dt_index.tz is not None:
            dt_index = dt_index.tz_localize(None)
        data_df.index = dt_index
        
        # Restore MultiIndex Columns: CUPS___Variable -> (CUPS, Variable)
        new_cols = []
        for c in data_df.columns:
            if "___" in c:
                parts = c.split("___")
                new_cols.append((parts[0], parts[1]))
            else:
                new_cols.append(("Unknown", c))
        
        data_df.columns = pd.MultiIndex.from_tuples(new_cols)
        data_df.index.name = 'Datetime'
        return data_df
        
    except Exception as e:
        st.error(f"Error carregant de Supabase: {e}")
        return None

def sync_csv_to_db(df, mode="merge"):
    """Saves DataFrame to 'energy_readings_wide' using JSONB format."""
    supabase = init_supabase()
    if not supabase: return
    
    if mode == "replace":
        st.warning("⚠️ Esborrant dades existents...")
        try:
            supabase.rpc("truncate_energy_readings", {}).execute()
        except Exception as e:
            st.error(f"Error esborrant: {e}")
            return

    # Flatten columns
    df_flat = df.copy()
    df_flat.columns = [f"{c[0]}___{c[1]}" for c in df_flat.columns]
    df_flat.index.name = 'reading_time'
    df_flat = df_flat.reset_index()
    df_flat = df_flat.drop_duplicates(subset=['reading_time'], keep='last')
    
    # Convert time to string
    df_flat['reading_time'] = df_flat['reading_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    total_rows = len(df_flat)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    chunk_size = 500
    meas_cols = [c for c in df_flat.columns if c != 'reading_time']
    data_dicts = df_flat[meas_cols].to_dict(orient='records')
    times = df_flat['reading_time'].tolist()
    
    payload = []
    for t, d in zip(times, data_dicts):
        clean_d = {k: v for k, v in d.items() if pd.notna(v)}
        payload.append({'reading_time': t, 'data': clean_d})
        
    for k in range(0, len(payload), chunk_size):
        chunk = payload[k:k+chunk_size]
        try:
            if mode == "replace":
                supabase.table("energy_readings_wide").upsert(chunk, on_conflict='reading_time').execute()
            else:
                supabase.rpc("merge_readings", {"payload": chunk}).execute()
            
            prog = min((k + len(chunk)) / total_rows, 1.0)
            progress_bar.progress(prog)
            status_text.text(f"Sincronitzant... {int(prog*100)}%")
        except Exception as e:
            st.error(f"Error en el bloc {k}: {e}")
            
    status_text.success("Dades sincronitzades correctament! 🚀")
