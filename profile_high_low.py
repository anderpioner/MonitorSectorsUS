import time
from database import get_db
from models import Sector
from data_service import calculate_sector_high_low
import pandas as pd

def profile_high_low():
    db = next(get_db())
    try:
        # Pick a larger sector like Technology or Financials
        sector = db.query(Sector).filter(Sector.name == 'Technology').first()
        if not sector:
            sector = db.query(Sector).first()
            
        print(f"Profiling High/Low for sector: {sector.name}")
        
        start_time = time.time()
        # Test with a 7-day window to simulate Quick Update
        test_start_date = pd.to_datetime('today') - pd.Timedelta(days=7)
        calculate_sector_high_low(sector.id, db, start_date=test_start_date)
        end_time = time.time()
        
        print(f"Total time for {sector.name}: {end_time - start_time:.2f} seconds")
        
    finally:
        db.close()

if __name__ == "__main__":
    profile_high_low()
