from database import get_db
from models import Sector, Constituent, PriceData, ConstituentPrice
from sqlalchemy import func

def check_counts():
    db = next(get_db())
    try:
        print("--- Database Counts ---")
        
        # 1. Total Sectors
        sector_count = db.query(func.count(Sector.id)).scalar()
        print(f"Total Sectors: {sector_count}")
        
        # 2. Total Constituents
        constituent_count = db.query(func.count(Constituent.id)).scalar()
        print(f"Total Constituents (Tickers): {constituent_count}")
        
        # 3. Breakdown by Sector
        print("\n--- Tickers per Sector ---")
        results = db.query(Sector.name, func.count(Constituent.id))\
                    .join(Constituent, Sector.id == Constituent.sector_id)\
                    .group_by(Sector.name)\
                    .order_by(func.count(Constituent.id).desc())\
                    .all()
                    
        for name, count in results:
            print(f"{name}: {count}")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_counts()
