import os
import sys
from supabase import create_client
from dotenv import load_dotenv

# Load env from 'scrapper huawei/.env'
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scrapper huawei', '.env')
load_dotenv(env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Supabase credentials not found in env.")
    sys.exit(1)

def main():
    print(f"Connecting to {SUPABASE_URL}...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("Fetching latest 5 records from FV_Sala_Nova...")
    try:
        response = supabase.table("FV_Sala_Nova").select("*").order("reading_time", desc=True).limit(5).execute()
        data = response.data
        
        if not data:
            print("No data found in table.")
        else:
            print(f"Found {len(data)} records:")
            for row in data:
                print(f"  {row['reading_time']} | {row.get('potencia_fv')} kW")
                
    except Exception as e:
        print(f"Error checking Supabase: {e}")

if __name__ == "__main__":
    main()
