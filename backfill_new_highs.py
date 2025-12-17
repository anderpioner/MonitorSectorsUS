import pandas as pd
from sqlalchemy import func
from database import get_db
from models import Sector, Constituent, ConstituentPrice, BreadthMetric
from datetime import date
import time

def backfill_highs_lows():
    print("Starting New Highs/Lows Backfill (252 days)...")
    db = next(get_db())
    
    try:
        # 1. Get all sectors
        sectors = db.query(Sector).all()
        
        for sector in sectors:
            print(f"\nProcessing Sector: {sector.name}...")
            
            # 2. Get constituents for this sector
            constituents = db.query(Constituent).filter(Constituent.sector_id == sector.id).all()
            if not constituents:
                print("  No constituents found.")
                continue
                
            c_ids = [c.id for c in constituents]
            
            # 3. Fetch all price data for these constituents
            # Optimization: Fetch simpler columns
            # We need to process per ticker to separate rolling windows.
            # Loading one by one might be slow, but safe.
            # Let's try loading all prices for the sector into a DF.
            
            print(f"  Fetching prices for {len(c_ids)} constituents...")
            prices_q = db.query(ConstituentPrice.constituent_id, ConstituentPrice.date, ConstituentPrice.close)\
                .filter(ConstituentPrice.constituent_id.in_(c_ids))\
                .order_by(ConstituentPrice.date)
            
            df = pd.read_sql(prices_q.statement, db.bind)
            
            if df.empty:
                print("  No price data.")
                continue
            
            df['date'] = pd.to_datetime(df['date'])
            
            # 4. Calculate New Highs / Lows per Constituent
            # Group by constituent
            high_counts = {} # date -> count
            low_counts = {}  # date -> count
            
            # Initialize with all unique dates in this sector to handle sparse data
            all_dates = df['date'].unique()
            for d in all_dates:
                high_counts[d] = 0
                low_counts[d] = 0
                
            print("  Calculating Rolling 252d Highs/Lows...")
            
            # Processing by group
            grouped = df.groupby('constituent_id')
            
            for cid, group in grouped:
                group = group.sort_values('date')
                # Rolling Max/Min (252 days)
                # Note: 'min_periods=252' means we need 252 days of history to have a valid high/low?
                # Usually yes.
                
                # Check if new high: Close >= Rolling Max(252)
                # Important: Rolling includes current row by default in pandas. 
                # If Close is the highest in the window, it's a new high.
                
                # Performance optimization:
                # We can compute rolling max/min on the whole column
                rolling_max = group['close'].rolling(window=252, min_periods=252).max()
                rolling_min = group['close'].rolling(window=252, min_periods=252).min()
                
                # Identify Highs: Close equals rolling_max
                # (using a small epsilon for float comparison logic, though usually exact match works for daily data)
                is_high = (group['close'] >= rolling_max - 1e-9)
                is_low = (group['close'] <= rolling_min + 1e-9)
                
                # Add to aggregates
                # Filter true only
                high_dates = group.loc[is_high, 'date']
                low_dates = group.loc[is_low, 'date']
                
                for d in high_dates:
                    high_counts[d] += 1
                for d in low_dates:
                    low_counts[d] += 1
            
            # 5. Store in Database (Bulk Upsert/Insert)
            # Create BreadthMetric objects
            print("  Preparing database records...")
            
            new_records = []
            
            # Convert counts to list of dicts
            # Filter out dates where count is 0? 
            # Ideally store 0 so valid dates are known, but to save space maybe only if > 0?
            # Actually, for charts continuous lines are better. 
            # But we can assume 0 if missing in the query service.
            # Let's store > 0 to be efficient.
            
            # Actually, delete existing first to avoid duplication or merge complexity?
            # Creating a delete query for this metric/sector is safer.
            db.query(BreadthMetric).filter(
                BreadthMetric.sector_id == sector.id,
                BreadthMetric.metric.in_(['new_highs_252', 'new_lows_252'])
            ).delete(synchronize_session=False)
            
            # Create objects
            # Highs
            for d, count in high_counts.items():
                if count > 0:
                    new_records.append(BreadthMetric(
                        sector_id=sector.id,
                        date=d.date(),
                        metric='new_highs_252',
                        value=float(count)
                    ))
            
            # Lows
            for d, count in low_counts.items():
                if count > 0:
                    new_records.append(BreadthMetric(
                        sector_id=sector.id,
                        date=d.date(),
                        metric='new_lows_252',
                        value=float(count)
                    ))
                    
            print(f"  Inserting {len(new_records)} records...")
            if new_records:
                db.bulk_save_objects(new_records)
                db.commit()
                
    except Exception as e:
        print(f"Error during backfill: {e}")
        db.rollback()
    finally:
        db.close()
        print("Done.")

if __name__ == "__main__":
    backfill_highs_lows()
