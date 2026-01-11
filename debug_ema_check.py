
from data_service import get_db, ConstituentPrice, Constituent
import pandas as pd

def check_data(ticker='AAPL'):
    db = next(get_db())
    try:
        print(f"Checking data for {ticker}...")
        
        # Join ConstituentPrice with Constituent to filter by ticker
        query = db.query(ConstituentPrice.date, ConstituentPrice.close, ConstituentPrice.ema8, ConstituentPrice.ema20, ConstituentPrice.ema50)\
            .join(Constituent)\
            .filter(Constituent.ticker == ticker)\
            .order_by(ConstituentPrice.date.desc())\
            .limit(10)
        
        df = pd.read_sql(query.statement, db.bind)
        if df.empty:
            print("No data found.")
        else:
            print(df)
            print("Null counts:")
            print(df.isnull().sum())
            
    finally:
        db.close()

if __name__ == "__main__":
    check_data()
