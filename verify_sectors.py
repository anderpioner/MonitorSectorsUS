from database import get_db, init_db
from models import Sector, Constituent
from sqlalchemy import text, inspect

def verify_changes():
    db = next(get_db())
    try:
        # 1. Verify Sector Names
        print("Verifying Sector Name Changes...")
        sectors = db.query(Sector).all()
        sector_names = sorted([s.name for s in sectors])
        
        expected_names = sorted([
            "Basic Materials Sector",
            "Communication Services Sector",
            "Consumer Cyclical Sector",
            "Consumer Defensive Sector",
            "Energy Sector",
            "Financial Services Sector",
            "Healthcare Sector",
            "Industrials Sector",
            "Real Estate Sector",
            "Technology Sector",
            "Utilities Sector"
        ])
        
        # We need to account that there might be duplicate entries if something went wrong, 
        # or multiple types (cap, equal). Let's check unique names.
        unique_db_names = sorted(list(set(sector_names)))
        
        missing = [n for n in expected_names if n not in unique_db_names]
        if not missing:
            print("  [OK] All expected sector names are present.")
        else:
            print(f"  [FAIL] Missing sectors: {missing}")
            print(f"  Found: {unique_db_names}")

        # 2. Verify Industry Column
        print("\nVerifying 'industry' column...")
        inspector = inspect(db.bind)
        columns = [c['name'] for c in inspector.get_columns('constituents')]
        
        if 'industry' in columns:
            print("  [OK] Column 'industry' found in 'constituents' table.")
        else:
            print("  [FAIL] Column 'industry' NOT found in 'constituents' table.")
            
    finally:
        db.close()

if __name__ == "__main__":
    verify_changes()
