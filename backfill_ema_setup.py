
import data_service as ds
from data_service import get_db, calculate_ema_setup_counts
import time

def run_backfill():
    print("Starting EMA Trend Setup Backfill...")
    db_session = next(get_db())
    
    try:
        # Get all sectors
        sectors = ds.get_sector_tickers(weight_type='cap')
        total_sectors = len(sectors)
        
        for idx, (s_name, s_ticker) in enumerate(sectors.items()):
            print(f"Processing {idx+1}/{total_sectors}: {s_name} ({s_ticker})")
            calculate_ema_setup_counts(s_ticker, db_session)
            
    except Exception as e:
        print(f"Error during backfill: {e}")
    finally:
        db_session.close()
        print("Backfill Complete.")

if __name__ == "__main__":
    run_backfill()
