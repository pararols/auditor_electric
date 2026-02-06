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
    target_date = "2026-02-05"
    print(f"Connecting to {SUPABASE_URL}...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print(f"Deleting records for {target_date} from FV_Sala_Nova...")
    
    # Supabase/PostgREST doesn't support "like" easily in delete without filters, 
    # but reading_time is effectively a string ISO format or timestamp.
    # We can use gte (>=) and lt (<) next day.
    
    start = f"{target_date}T00:00:00"
    end = f"{target_date}T23:59:59"
    
    try:
        # Check count first
        response = supabase.table("FV_Sala_Nova").select("*", count="exact").gte("reading_time", start).lte("reading_time", end).execute()
        count = len(response.data)
        print(f"Found {count} records to delete.")
        
        if count > 0:
            del_response = supabase.table("FV_Sala_Nova").delete().gte("reading_time", start).lte("reading_time", end).execute()
            print(f"Deleted records. Check: {len(del_response.data)} removed.")
        else:
            print("Nothing to delete.")
                
    except Exception as e:
        print(f"Error deleting from Supabase: {e}")

if __name__ == "__main__":
    main()
