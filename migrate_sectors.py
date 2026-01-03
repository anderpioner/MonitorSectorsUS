from database import get_db, init_db
from models import Sector
from sqlalchemy import text

# New Sector Mapping (Old Name -> New Name)
SECTOR_RENAME_MAP = {
    "Materials": "Basic Materials Sector",
    "Communication Services": "Communication Services Sector",
    "Consumer Discretionary": "Consumer Cyclical Sector",
    "Consumer Staples": "Consumer Defensive Sector",
    "Energy": "Energy Sector",
    "Financials": "Financial Services Sector",
    "Health Care": "Healthcare Sector",
    "Industrials": "Industrials Sector",
    "Real Estate": "Real Estate Sector",
    "Technology": "Technology Sector",
    "Utilities": "Utilities Sector"
}

def migrate_db():
    print("Starting DB Migration...")
    db = next(get_db())
    
    try:
        # 1. Rename Sectors
        print("Renaming Sectors...")
        for old_name, new_name in SECTOR_RENAME_MAP.items():
            sector = db.query(Sector).filter(Sector.name == old_name).first()
            if sector:
                print(f"  Renaming '{old_name}' -> '{new_name}'")
                sector.name = new_name
            else:
                # Check if already renamed
                new_sector = db.query(Sector).filter(Sector.name == new_name).first()
                if new_sector:
                    print(f"  Sector '{new_name}' already exists.")
                else:
                    print(f"  Warning: Sector '{old_name}' not found.")
        
        db.commit()
        
        # 2. Add Industry Column
        print("Adding 'industry' column to constituents table...")
        try:
            db.execute(text("ALTER TABLE constituents ADD COLUMN industry VARCHAR"))
            print("  Column 'industry' added successfully.")
        except Exception as e:
            if "duplicate column name" in str(e).lower():
                print("  Column 'industry' already exists.")
            else:
                print(f"  Error adding column: {e}")
                
        db.commit()
        print("Migration Complete!")
        
    except Exception as e:
        db.rollback()
        print(f"Migration Failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    migrate_db()
