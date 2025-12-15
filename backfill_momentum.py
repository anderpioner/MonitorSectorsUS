import pandas as pd
import numpy as np
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
from database import DATABASE_URL
from models import PriceData, Sector

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def calculate_momentum_vectorized(df):
    """
    Calculates momentum score for a DataFrame of prices (indexed by date).
    Returns a Series of scores aligned with the index.
    """
    # Lagged Prices
    p = df['close']
    
    # We need lags of 1, 5, 10, 20, 40 days
    # Since these are trading days, we can just use shift on the sorted series
    p_1 = p.shift(1)
    p_5 = p.shift(5)
    p_10 = p.shift(10)
    p_20 = p.shift(20)
    p_40 = p.shift(40)
    
    # Returns
    r_5_1 = (p_1 / p_5) - 1
    r_10_5 = (p_5 / p_10) - 1
    r_20_10 = (p_10 / p_20) - 1
    r_40_20 = (p_20 / p_40) - 1
    
    # Score
    score = (0.3 * r_5_1) + (0.3 * r_10_5) + (0.2 * r_20_10) + (0.2 * r_40_20)
    
    return score

def run_backfill():
    print("Starting Momentum Backfill...")
    sectors = session.query(Sector).all()
    
    total_updated = 0
    
    for sector in sectors:
        print(f"Processing {sector.ticker}...")
        
        # Load data
        query = session.query(PriceData).filter(PriceData.sector_id == sector.id).order_by(PriceData.date)
        df = pd.read_sql(query.statement, session.bind)
        
        if df.empty:
            continue
            
        # Calculate Scores
        df['momentum_score'] = calculate_momentum_vectorized(df)
        
        # Update DB
        # We only update rows where score is not NaN
        updates = []
        for _, row in df.dropna(subset=['momentum_score']).iterrows():
            updates.append({
                'id': row['id'],
                'momentum_score': row['momentum_score']
            })
            
        if updates:
            # Bulk update
            session.bulk_update_mappings(PriceData, updates)
            session.commit()
            total_updated += len(updates)
            print(f"  Updated {len(updates)} rows.")
            
    print(f"Backfill Complete. Total rows updated: {total_updated}")

if __name__ == "__main__":
    run_backfill()
