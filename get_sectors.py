
from database import get_db
from models import Sector

def list_sectors():
    db = next(get_db())
    try:
        sectors = db.query(Sector).all()
        print(f"{'ID':<5} | {'Name':<30} | {'Type':<10} | {'Ticker':<10}")
        print("-" * 60)
        for s in sectors:
            print(f"{s.id:<5} | {s.name:<30} | {s.type:<10} | {s.ticker:<10}")
    finally:
        db.close()

if __name__ == "__main__":
    list_sectors()
