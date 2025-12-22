
from data_service import calculate_active_count
from database import get_db
from models import Sector

def backfill_active_count():
    print("Starting backfill for 'active_count' metric...")
    db = next(get_db())
    
    # Get all sectors
    sectors = db.query(Sector).filter(Sector.type == 'cap').all()
    
    total = len(sectors)
    for i, sector in enumerate(sectors):
        print(f"[{i+1}/{total}] Processing {sector.name}...")
        try:
             # Calculate for full history
            calculate_active_count(sector.id, db, start_date=None)
        except Exception as e:
            print(f"Error calculating for {sector.name}: {e}")
            
    db.close()
    print("Backfill complete.")

if __name__ == "__main__":
    backfill_active_count()
