from database import get_db
from models import Sector, BreadthMetric, ConstituentPrice, Constituent
from sqlalchemy import func
import pandas as pd

def diagnose_dates():
    db = next(get_db())
    try:
        sectors = db.query(Sector).filter(Sector.type == 'cap').all()
        print(f"{'Sector':<25} | {'Latest Price Date':<12} | {'Latest Breadth Date'}")
        print("-" * 60)
        
        for sector in sectors:
            # Check latest constituent price date
            last_price_date = db.query(func.max(ConstituentPrice.date))\
                .join(Constituent)\
                .filter(Constituent.sector_id == sector.id)\
                .scalar()
                
            # Check latest breadth metric date
            last_breadth_date = db.query(func.max(BreadthMetric.date))\
                .filter(BreadthMetric.sector_id == sector.id)\
                .scalar()
                
            print(f"{sector.name:<25} | {str(last_price_date):<12} | {str(last_breadth_date)}")
            
    finally:
        db.close()

if __name__ == "__main__":
    diagnose_dates()
