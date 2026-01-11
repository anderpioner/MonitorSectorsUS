
import data_service as ds
from models import BreadthMetric, Sector
from database import get_db
import pandas as pd

def check_data():
    db = next(get_db())
    sector_name = 'Technology Sector'
    
    print(f"Checking data for {sector_name} since 2026-01-01...")
    
    sector = db.query(Sector).filter_by(name=sector_name).first()
    if not sector:
        print("Sector not found")
        return

    # Check max date generally
    max_date = db.query(BreadthMetric.date).order_by(BreadthMetric.date.desc()).first()
    print(f"Max date in BreadthMetric: {max_date[0] if max_date else 'None'}")

    metrics = ['ema_trend_setup', 'active_count']
    
    for m in metrics:
        results = db.query(BreadthMetric.date, BreadthMetric.value)\
            .filter(BreadthMetric.sector_id == sector.id)\
            .filter(BreadthMetric.metric == m)\
            .filter(BreadthMetric.date >= '2026-01-01')\
            .order_by(BreadthMetric.date)\
            .all()
            
        print(f"\nMetric: {m}")
        for r in results:
            print(f"  {r[0]}: {r[1]}")

    db.close()

if __name__ == "__main__":
    check_data()
