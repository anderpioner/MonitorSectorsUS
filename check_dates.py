from database import get_db
from models import PriceData, ConstituentPrice
from sqlalchemy import func

db = next(get_db())
try:
    max_price = db.query(func.max(PriceData.date)).scalar()
    max_const = db.query(func.max(ConstituentPrice.date)).scalar()
    
    print(f"Max Date in PriceData (ETFs): {max_price}")
    print(f"Max Date in ConstituentPrice (Stocks): {max_const}")
    
    if str(max_price) != str(max_const):
        print("WARNING: Dates are out of sync!")
finally:
    db.close()
