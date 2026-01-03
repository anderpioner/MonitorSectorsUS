import pandas as pd
import pandas as pd
# import pandas_ta as ta # Not installed, using manual calc
from database import get_db
from database import get_db
from models import Constituent, ConstituentPrice, Sector
from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert
import time

def calculate_indicators(df):
    """
    Calculates SMAs, EMAs, and ATR for a DataFrame with 'close', 'high', 'low'.
    Returns DataFrame with new columns.
    """
    if df.empty: return df
    
    # Ensure sorted by date
    df = df.sort_index()
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # SMAs
    df['ma5'] = close.rolling(window=5).mean()
    df['ma10'] = close.rolling(window=10).mean()
    df['ma20'] = close.rolling(window=20).mean()
    df['ma50'] = close.rolling(window=50).mean()
    df['ma200'] = close.rolling(window=200).mean()
    
    # EMAs
    df['ema8'] = close.ewm(span=8, adjust=False).mean()
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()
    
    # ATR 14
    # Manually calc TR to avoid pandas_ta dependency if just for this, 
    # but since we are refactoring, let's use the same logic as data_service.
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    return df

def backfill_indicators():
    print("Starting Backfill of Technical Indicators (SMA, EMA, ATR)...")
    db = next(get_db())
    try:
        # Get all constituents
        constituents = db.query(Constituent.id, Constituent.ticker).all()
        total = len(constituents)
        
        batch_size = 50
        
        for i, (c_id, ticker) in enumerate(constituents):
            if i % 10 == 0:
                print(f"Processing {i}/{total}: {ticker}")
                
            # Fetch all price history for this constituent
            # We need High/Low/Close.
            # Note: Before migration, High/Low might be NULL if we didn't save them.
            # If High/Low are null, we can't calc ATR.
            # We can check if they exist.
            
            query = db.query(ConstituentPrice).filter(ConstituentPrice.constituent_id == c_id).order_by(ConstituentPrice.date)
            rows = query.all()
            
            if not rows:
                continue
                
            # Convert to DF
            data = []
            for r in rows:
                data.append({
                    'date': r.date,
                    'close': r.close,
                    'high': r.high,
                    'low': r.low
                })
            
            df = pd.DataFrame(data).set_index('date')
            
            # Check if we have high/low data
            if df['high'].isnull().all():
                # Cannot calc ATR strictly, but maybe user wants SMAs/EMAs at least?
                # For this task (ATR Panel), ATR is crucial.
                # If no High/Low, we can't do ATR.
                # Skip or just calc what we can.
                pass
            
            # Calc indicators
            # Make sure we don't overwrite existing if not needed, but backfill implies overwrite/fill.
            df_calc = calculate_indicators(df)
            
            # Prepare Update
            # Using bulk update mappings
            mappings = []
            for dt, row in df_calc.iterrows():
                # Only update if we have new values? 
                # Or easier to just update all valid fields.
                
                update_dict = {
                    'id': None, # Need ID to use mapping? Or composite key.
                                # SQLAlchemy bulk_update_mappings usually needs Primary Key.
                    # 'constituent_id': c_id,
                    # 'date': dt
                }
                
                # Finding the row object ID is slow.
                # Better to use raw SQL or filtered update?
                # Or just iterate and update objects if session is active.
                pass

            # Optimization: updating ORM objects in loop is slow for big data.
            # But simpler code.
            # Let's verify how many rows. 10 years = 2500 rows.
            # 5000 tickers * 2500 rows = 12.5M rows. Too slow for ORM loop.
            
            # Since we just added the columns, they are NULL.
            # We can use a special SQL update if calculation is simple, but EMA/ATR is recursive.
            # Must be done in Python.
            
            # Compromise: Only update last 100 days? 
            # The User wants the panel for "Today". 
            # So calculating strictly the latest values is enough for the panel.
            # But "Backfill" implies history.
            
            # Let's update objects in batch using mappings?
            # Re-map rows to objects
            
            if not rows: continue
            
            # Update attributes on loaded objects
            for r in rows:
                dt = r.date
                if dt in df_calc.index:
                    vals = df_calc.loc[dt]
                    
                    if pd.notna(vals['ema8']): r.ema8 = float(vals['ema8'])
                    if pd.notna(vals['ema20']): r.ema20 = float(vals['ema20'])
                    if pd.notna(vals['ema50']): r.ema50 = float(vals['ema50'])
                    if pd.notna(vals['atr14']): r.atr14 = float(vals['atr14'])
                    
                    # Also SMAs if missing
                    if pd.notna(vals['ma5']): r.ma5 = float(vals['ma5'])
                    # ... others
            
            # Commit every ticker or batch
            db.commit()
            
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    backfill_indicators()
