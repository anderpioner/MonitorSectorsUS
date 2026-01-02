from data_service import get_db, Sector, ConstituentPrice, BreadthMetric
from sqlalchemy import func
import pandas as pd

def check_dates():
    db = next(get_db())
    try:
        sectors = db.query(Sector).filter(Sector.type == 'cap').all()
        print(f"{'Sector':<25} | {'Latest Price':<12} | {'Latest Breadth':<12} | {'Diff'}")
        print("-" * 65)
        
        for s in sectors:
            # Latest Price Date
            max_price_date = db.query(func.max(ConstituentPrice.date))\
                .join(ConstituentPrice.constituent)\
                .filter(ConstituentPrice.constituent.has(sector_id=s.id))\
                .scalar()
                
            # Latest Breadth Metric Date
            max_breadth_date = db.query(func.max(BreadthMetric.date))\
                .filter(BreadthMetric.sector_id == s.id)\
                .filter(BreadthMetric.metric == 'pct_above_ma50')\
                .scalar()
                
            diff = "OK"
            if max_price_date and max_breadth_date:
                if max_price_date > max_breadth_date:
                    diff = "MISSING METRICS"
            elif max_price_date and not max_breadth_date:
                 diff = "NO METRICS"
            
            p_str = str(max_price_date) if max_price_date else "None"
            b_str = str(max_breadth_date) if max_breadth_date else "None"
            
            print(f"{s.name:<25} | {p_str:<12} | {b_str:<12} | {diff}")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_dates()
