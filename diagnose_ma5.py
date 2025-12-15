from database import get_db
from models import ConstituentPrice, Constituent, Sector
from sqlalchemy import func, case

def check_ma5_status():
    db = next(get_db())
    try:
        results = db.query(
            Sector.name, 
            func.count(ConstituentPrice.id).label('total_rows'),
            func.sum(case((ConstituentPrice.ma5.is_(None), 1), else_=0)).label('null_ma5_count')
        ).select_from(Sector).join(Constituent).join(ConstituentPrice).group_by(Sector.name).all()
        
        print(f"{'Sector':<25} | {'Total Rows':<10} | {'Null MA5':<10} | {'% Missing':<10}")
        print("-" * 65)
        
        missing_sectors = []
        for r in results:
            sector_name = r[0]
            total = r[1]
            null_cnt = r[2] or 0
            pct = (null_cnt / total * 100) if total > 0 else 0
            
            print(f"{sector_name:<25} | {total:<10} | {null_cnt:<10} | {pct:.1f}%")
            
            if pct > 0:
                missing_sectors.append(sector_name)
                
        return missing_sectors
        
    finally:
        db.close()

if __name__ == "__main__":
    missing = check_ma5_status()
    if missing:
        print("\nSectors needing repair:", missing)
    else:
        print("\nAll sectors have MA5 data.")
