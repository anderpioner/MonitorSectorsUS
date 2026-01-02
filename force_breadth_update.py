from data_service import get_db, Sector, calculate_sector_breadth, calculate_sector_up_metrics, calculate_active_count, calculate_sector_high_low
import pandas as pd

def force_breadth_update():
    db = next(get_db())
    try:
        sectors = db.query(Sector).filter(Sector.type == 'cap').all()
        print(f"Forcing breadth update for {len(sectors)} sectors...")
        
        for s in sectors:
            print(f"Processing {s.name}...")
            # Calculate Breadth
            calculate_sector_breadth(s.id, db)
            
            # Calculate Up Metrics
            calculate_sector_up_metrics(s.id, db, lookback_window=84, thresholds=[0.25, 0.50, 1.00])
            
            # Calculate Active Count
            calculate_active_count(s.id, db)
            
            # Calculate High/Low 
            calculate_sector_high_low(s.id, db)
            
            print(f"  > Done.")
            
    finally:
        db.close()

if __name__ == "__main__":
    force_breadth_update()
