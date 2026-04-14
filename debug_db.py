import os
import pandas as pd
from src.core.database import init_supabase, load_from_supabase_db
from src.core.config import COMMUNITY_PARTICIPANTS
import warnings
warnings.filterwarnings('ignore')

supabase = init_supabase()
if supabase:
    print("Connected to Supabase")
    try:
        res = supabase.rpc("get_distinct_years").execute()
        print("Anys disponibles a la DB:", [r['year'] for r in res.data])
    except Exception as e:
        print("Error getting distinct years:", e)
    
    # Check 2025 data specifically
    df = load_from_supabase_db(start_date="2025-01-01", end_date="2025-12-31")
    if df is not None:
        print("Anys trobats en el rang 2025:", df.index.year.unique().tolist())
        print("CUPS trobats a la DB:", df.columns.get_level_values(0).unique().tolist())
        
        present = [c for c in COMMUNITY_PARTICIPANTS if c in df.columns.get_level_values(0)]
        print("Participants presents a la DB per 2025:", present)
    else:
        print("No s'han pogut carregar dades de la DB per al 2025.")
else:
    print("Error connectant a Supabase.")
