
from data_service import calculate_sector_up_metric, get_sector_tickers
from database import get_db
from models import Sector

def backfill_up_metric():
    print("Starting backfill for 'Stocks > 25%' metric...")
    db = next(get_db())
    
    # Get all sectors
    sectors = db.query(Sector).filter(Sector.type == 'cap').all()
    
    total = len(sectors)
    for i, sector in enumerate(sectors):
        print(f"[{i+1}/{total}] Processing {sector.name}...")
        try:
             # Calculate for full history (no start_date = full history)
            calculate_sector_up_metric(sector.id, db, start_date=None, lookback_window=84, threshold=0.25)
        except Exception as e:
            print(f"Error calculating for {sector.name}: {e}")
            
    db.close()
    print("Backfill complete.")

if __name__ == "__main__":
    backfill_up_metric()
