import pandas as pd
import streamlit as st
from ..core.config import CUPS_MAPPING

def parse_processed_csv(uploaded_file):
    """
    Parses the specific Multi-Index Header CSV format manually to avoid header length mismatches.
    Row 0: CUPS IDs (Merged cells need forward fill)
    Row 1: Variable names (AE_kWh, AE_AUTOCONS_kWh)
    """
    try:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, header=None, sep=';', dtype=object, engine='python')
        
        # 1. Extract Headers
        header_r0 = df_raw.iloc[0].copy().ffill()
        header_r1 = df_raw.iloc[1].copy()
        
        # 2. Extract Data
        df_data = df_raw.iloc[2:].copy()
        
        # 3. Construct MultiIndex Columns
        new_columns = []
        for i in range(len(header_r0)):
            h0 = str(header_r0.iloc[i]).strip() if pd.notna(header_r0.iloc[i]) else 'Metadata'
            h1 = str(header_r1.iloc[i]).strip() if pd.notna(header_r1.iloc[i]) else ""
            
            if h0 != 'Metadata':
                h0 = CUPS_MAPPING.get(h0, h0)
            new_columns.append((h0, h1))
            
        df_data.columns = pd.MultiIndex.from_tuples(new_columns)
        
        # 4. Identification of Date/Time
        # Fallback to pos 0 and 1 if standard names not found
        date_col = df_data.columns[0]
        time_col = df_data.columns[1]

        def clean_time(t):
            t = str(t).strip()
            if t.isdigit(): return f"{int(t):02d}:00"
            return t
            
        datetime_str = df_data[date_col].astype(str) + ' ' + df_data[time_col].apply(clean_time)
        df_data.index = pd.to_datetime(datetime_str, dayfirst=True, errors='coerce')
        df_data.index.name = 'Datetime'
        
        df_data = df_data[df_data.index.notna()]
        df_data = df_data.drop(columns=[date_col, time_col])
        
        # 5. Convert Numeric
        for col in df_data.columns:
            if df_data[col].dtype == object:
                val_series = df_data[col].astype(str).str.replace(',', '.', regex=False)
                df_data[col] = pd.to_numeric(val_series, errors='coerce').fillna(0)
                
        return df_data
    except Exception as e:
        st.error(f"Error parsejant CSV Processat: {e}")
        return None

def process_edistribucion_files(uploaded_files):
    """Parses and merges multiple raw Edistribucion CSVs."""
    all_records = []
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        progress_text.text(f"Processant {file.name}...")
        try:
            df = pd.read_csv(file, sep=';', encoding='latin-1', dtype=str)
            required = ['CUPS', 'Fecha', 'Hora', 'AE_kWh']
            if not all(col in df.columns for col in required):
                st.warning(f"Saltant {file.name}: Falten columnes requerides.")
                continue
                
            # Normalize Hours
            hours = pd.to_numeric(df['Hora'], errors='coerce')
            min_h, max_h = hours.min(), hours.max()
            has_zero, has_24 = (hours == 0).any(), (hours == 24).any()
            
            if not has_zero and (has_24 or min_h == 1):
                df['Hora'] = hours - 1
            else:
                df['Hora'] = hours
                
            # Timestamps
            df['datetime_str'] = df['Fecha'] + ' ' + df['Hora'].astype(str).str.pad(2, fillchar='0') + ':00'
            df['reading_time'] = pd.to_datetime(df['datetime_str'], format='%d/%m/%Y %H:%M', errors='coerce')
            
            # Numeric conversion
            df['AE_kWh'] = pd.to_numeric(df['AE_kWh'].str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce').fillna(0)
            if 'AE_AUTOCONS_kWh' in df.columns:
                 df['AE_AUTOCONS_kWh'] = pd.to_numeric(df['AE_AUTOCONS_kWh'].str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce').fillna(0)
            else:
                 df['AE_AUTOCONS_kWh'] = 0
            
            df = df.dropna(subset=['reading_time'])
            cups_raw = df['CUPS'].iloc[0] if not df.empty else "UNKNOWN"
            cups_file = CUPS_MAPPING.get(cups_raw, cups_raw)
            
            subset = df[['reading_time', 'AE_kWh', 'AE_AUTOCONS_kWh']].copy()
            subset['CUPS'] = cups_file
            all_records.append(subset)
            
        except Exception as e:
            st.error(f"Error fitxer {file.name}: {e}")
            
        progress_bar.progress((i + 1) / len(uploaded_files))
            
    progress_text.empty()
    progress_bar.empty()
    
    if not all_records: return None
        
    big_df = pd.concat(all_records, ignore_index=True)
    pivot = big_df.pivot_table(index='reading_time', columns='CUPS', values=['AE_kWh', 'AE_AUTOCONS_kWh'], aggfunc='last')
    pivot.columns = pivot.columns.swaplevel(0, 1)
    pivot.columns.names = [None, None]
    pivot.sort_index(axis=1, inplace=True)
    pivot.index.name = 'Datetime'
    return pivot.fillna(0)
