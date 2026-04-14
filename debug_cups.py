import json
import pandas as pd
from src.core.database import init_supabase, load_from_supabase_db
from src.core.config import COMMUNITY_PARTICIPANTS
import warnings
warnings.filterwarnings('ignore')

supabase = init_supabase()
if supabase:
    df = load_from_supabase_db(start_date="2025-01-01", end_date="2025-12-31")
    if df is not None:
        db_cups = df.columns.get_level_values(0).unique().tolist()
        with open('cups_db.json', 'w') as f:
            json.dump({'db_cups': db_cups, 'config_cups': COMMUNITY_PARTICIPANTS}, f, indent=4)
        print("Saved to JSON")
    else:
        print("No df returned for 2025.")
