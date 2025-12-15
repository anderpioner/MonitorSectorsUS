from database import get_db
from models import PriceData, Sector
from sqlalchemy import func

db = next(get_db())

try:
    print("--- Sectors ---")
    sectors = db.query(Sector).all()
    for s in sectors:
        print(f"ID: {s.id}, Name: {s.name}, Ticker: {s.ticker}, Type: {s.type}")

    print("\n--- PriceData Stats ---")
    count = db.query(PriceData).count()
    print(f"Total Rows: {count}")

    if count > 0:
        min_date = db.query(func.min(PriceData.date)).scalar()
        max_date = db.query(func.max(PriceData.date)).scalar()
        print(f"Min Date: {min_date}")
        print(f"Max Date: {max_date}")
        
    print("\n--- PriceData per Sector (Last 5) ---")
    # Check counts per sector
    stats = db.query(PriceData.sector_id, func.count(PriceData.id)).group_by(PriceData.sector_id).all()
    for sid, cnt in stats:
        print(f"Sector ID {sid}: {cnt} rows")

finally:
    db.close()
