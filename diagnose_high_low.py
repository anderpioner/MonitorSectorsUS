from database import get_db
from database import get_db
from models import BreadthMetric, Sector
from sqlalchemy import func

def check_high_low_dates():
    db = next(get_db())
    try:
        sectors = db.query(Sector).filter(Sector.type == 'cap').all()
        print(f"{'Sector':<25} | {'Latest High/Low Date'}")
        print("-" * 50)
        
        for sector in sectors:
            last_date = db.query(func.max(BreadthMetric.date))\
                .filter(BreadthMetric.sector_id == sector.id)\
                .filter(BreadthMetric.metric == 'new_highs_252')\
                .scalar()
            
            print(f"{sector.name:<25} | {last_date}")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_high_low_dates()
