
from database import get_db
from models import BreadthMetric, Sector

def check_active_count():
    db = next(get_db())
    try:
        count = db.query(BreadthMetric).filter(BreadthMetric.metric == 'active_count').count()
        print(f"Total 'active_count' records found: {count}")
        
        if count > 0:
            sample = db.query(BreadthMetric).filter(BreadthMetric.metric == 'active_count').first()
            print(f"Sample: {sample.date} - Sector {sample.sector_id} - Value: {sample.value}")
    finally:
        db.close()

if __name__ == "__main__":
    check_active_count()
