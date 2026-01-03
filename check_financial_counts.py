from database import get_db
from models import Sector, Constituent
from sqlalchemy import func

def check_financial_breakdown():
    db = next(get_db())
    try:
        print("--- Financial Services Sector Breakdown ---")
        
        # Get Sector ID
        sector = db.query(Sector).filter(Sector.name == "Financial Services Sector").first()
        if not sector:
            print("Sector not found!")
            return

        print(f"Sector ID: {sector.id}")
        
        # Count per Industry
        results = db.query(Constituent.industry, func.count(Constituent.id))\
                    .filter(Constituent.sector_id == sector.id)\
                    .group_by(Constituent.industry)\
                    .order_by(func.count(Constituent.id).desc())\
                    .all()
                    
        total = 0
        for industry, count in results:
            print(f"{industry}: {count}")
            total += count
            
        print(f"\nTotal in DB: {total}")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_financial_breakdown()
