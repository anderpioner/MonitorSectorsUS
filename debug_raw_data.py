
import data_service as ds
from models import ConstituentPrice, Constituent, Sector
from database import get_db
import pandas as pd

def check_raw_prices():
    db = next(get_db())
    sector_name = 'Technology Sector'
    target_date = '2026-01-05'
    
    print(f"Checking ConstituentPrice for {sector_name} on {target_date}...")
    
    # Get a few tickers from this sector
    tickers = ds.get_sector_constituents('XLK')[:5] # XLK is Tech
    print(f"Sample tickers: {tickers}")
    
    query = db.query(Constituent.ticker, ConstituentPrice.date, ConstituentPrice.close, ConstituentPrice.ema8)\
        .join(Constituent)\
        .filter(Constituent.ticker.in_(tickers))\
        .filter(ConstituentPrice.date == target_date)
        
    results = query.all()
    
    print(f"Found {len(results)} records for these 5 tickers on {target_date}.")
    for r in results:
        print(f"  {r[0]} | {r[1]} | Close: {r[2]} | EMA8: {r[3]}")
        
    db.close()

if __name__ == "__main__":
    check_raw_prices()
