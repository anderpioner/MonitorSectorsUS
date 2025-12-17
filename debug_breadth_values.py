from database import get_db
from models import BreadthMetric, Sector
import pandas as pd
from sqlalchemy import func

def check_breadth_values():
    db = next(get_db())
    try:
        # Check specific sector "Health Care" or "Industrials"
        # The user mentioned Industrial, Materials, Technology, Real Estate, Utilities
        target_names = ['Industrials', 'Materials', 'Technology', 'Real Estate', 'Utilities']
        
        print(f"{'Sector':<20} | {'Metric':<15} | {'Date':<12} | {'Value'}")
        print("-" * 60)
        
        for name in target_names:
            sector = db.query(Sector).filter(Sector.name == name).first()
            if not sector:
                print(f"{name} not found")
                continue
                
            # Get latest 5 metrics
            metrics = db.query(BreadthMetric).filter(BreadthMetric.sector_id == sector.id)\
                        .order_by(BreadthMetric.date.desc()).limit(10).all()
            
            if not metrics:
                print(f"{name:<20} | NO DATA")
            else:
                for m in metrics:
                    print(f"{name:<20} | {m.metric:<15} | {m.date} | {m.value}")
    finally:
        db.close()

if __name__ == "__main__":
    check_breadth_values()
