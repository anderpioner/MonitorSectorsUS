from database import get_db
from models import Sector, Constituent, ConstituentPrice
from sqlalchemy import func
import pandas as pd

def verify_coverage():
    db = next(get_db())
    try:
        # 1. Get Latest Date
        latest_date = db.query(func.max(ConstituentPrice.date)).scalar()
        print(f"Latest Data Date in DB: {latest_date}")
        
        if not latest_date:
            print("No data in ConstituentPrice!")
            return

        # 2. Get Total Constituents per Sector
        print("\n--- Coverage Report (Latest Date) ---")
        
        # Query: Sector Name, Total Tickers, Tickers with Data on Latest Date
        # We can do this with two queries and merge, or one complex one.
        # Two queries is clearer.
        
        # A. Total Counts
        total_counts = db.query(Sector.name, func.count(Constituent.id))\
            .join(Constituent)\
            .filter(Sector.type == 'cap')\
            .group_by(Sector.name)\
            .all()
        total_map = {r[0]: r[1] for r in total_counts}
        
        # B. Analyzed Counts (on latest_date)
        analyzed_counts = db.query(Sector.name, func.count(Constituent.id))\
            .join(Constituent)\
            .join(ConstituentPrice)\
            .filter(Sector.type == 'cap')\
            .filter(ConstituentPrice.date == latest_date)\
            .group_by(Sector.name)\
            .all()
        analyzed_map = {r[0]: r[1] for r in analyzed_counts}
        
        # Print Comparison
        print(f"{'Sector':<30} | {'Total':<8} | {'Analyzed':<8} | {'Missing':<8} | {'Coverage':<8}")
        print("-" * 75)
        
        for sector, total in total_map.items():
            analyzed = analyzed_map.get(sector, 0)
            missing = total - analyzed
            coverage = (analyzed / total * 100) if total > 0 else 0
            
            print(f"{sector:<30} | {total:<8} | {analyzed:<8} | {missing:<8} | {coverage:.1f}%")
            
    finally:
        db.close()

if __name__ == "__main__":
    verify_coverage()
