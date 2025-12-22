import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from database import get_db, init_db
from models import Sector, PriceData, Constituent, BreadthMetric, ConstituentPrice
from sqlalchemy.dialects.sqlite import insert
from datetime import date, datetime, timedelta
import time

# List of US Sector ETFs (SPDR and Invesco Equal Weight)
# Format: { 'Sector Name': {'cap': 'TickerCW', 'equal': 'TickerEW'} }
SECTORS_CONFIG = {
    'Communication Services': {'cap': 'XLC', 'equal': 'RSPC'},
    'Consumer Discretionary': {'cap': 'XLY', 'equal': 'RSPD'},
    'Consumer Staples': {'cap': 'XLP', 'equal': 'RSPS'},
    'Energy': {'cap': 'XLE', 'equal': 'RSPG'},
    'Financials': {'cap': 'XLF', 'equal': 'RSPF'},
    'Health Care': {'cap': 'XLV', 'equal': 'RSPH'},
    'Industrials': {'cap': 'XLI', 'equal': 'RSPN'},
    'Materials': {'cap': 'XLB', 'equal': 'RSPM'},
    'Real Estate': {'cap': 'XLRE', 'equal': 'RSPR'},
    'Technology': {'cap': 'XLK', 'equal': 'RSPT'},
    'Utilities': {'cap': 'XLU', 'equal': 'RSPU'}
}

def initialize_sectors_in_db():
    """Ensures all sectors exist in the database."""
    init_db()
    db = next(get_db())
    try:
        current_sectors = db.query(Sector).all()
        existing_tickers = {s.ticker for s in current_sectors}
        
        for name, types in SECTORS_CONFIG.items():
            # Cap Weighted
            ticker_cap = types['cap']
            if ticker_cap not in existing_tickers:
                db.add(Sector(name=name, ticker=ticker_cap, type='cap'))
                
            # Equal Weighted
            ticker_eq = types['equal']
            if ticker_eq not in existing_tickers:
                db.add(Sector(name=name, ticker=ticker_eq, type='equal'))
                
        db.commit()
    finally:
        db.close()

def update_sector_data(period="10y"):
    """
    Fetches latest data from yfinance and updates the database.
    """
    initialize_sectors_in_db()
    db = next(get_db())
    
    # Collect all tickers
    tickers = []
    for types in SECTORS_CONFIG.values():
        tickers.append(types['cap'])
        tickers.append(types['equal'])
        
    print(f"Fetching data for: {tickers}")
    data = yf.download(tickers, period=period, auto_adjust=True)['Close']
    
    if isinstance(data, pd.Series):
        data = data.to_frame()

    try:
        # Resolve Sector IDs and Objects
        sector_map = {s.ticker: s for s in db.query(Sector).all()}
        
        for ticker in data.columns:
            if ticker not in sector_map:
                continue
                
            sector_obj = sector_map[ticker]
            series = data[ticker].dropna()
            
            # Keep track if we added/updated anything for this ticker
            updated_ticker = False
            
            for dt, price in series.items():
                if pd.isna(price):
                    continue
                    
                # Check if exists (Naive approach for now, optimize with bulk upsert later if needed)
                date_val = dt.date()
                existing = db.query(PriceData).filter_by(sector_id=sector_obj.id, date=date_val).first()
                
                if not existing:
                    new_price = PriceData(sector_id=sector_obj.id, date=date_val, close=float(price))
                    db.add(new_price)
                    updated_ticker = True
                else:
                    # Update close if changed (e.g. adjustments)
                    if abs(existing.close - float(price)) > 0.001:
                        existing.close = float(price)
                        updated_ticker = True
            
            db.commit() # Commit prices first
            
            if updated_ticker:
                 # Recalculate momentum for this sector (to ensure latest score is saved)
                # We need to load recent history to calc momentum
                # Load last 60 days
                history_query = db.query(PriceData).filter(PriceData.sector_id == sector_obj.id).order_by(PriceData.date.desc()).limit(60)
                history_rows = history_query.all()
                # Reverse to chrono order
                history_rows = history_rows[::-1]
                
                if len(history_rows) > 50:
                     # Convert to list of dicts for DataFrame
                    data_hist = [{'id': r.id, 'close': r.close} for r in history_rows]
                    df_calc = pd.DataFrame(data_hist)
                    
                    # Calculate simple manual momentum for the LATEST row only to update it
                    # Logic same as vectorized but single point
                    p = df_calc['close']
                    try:
                        p_1 = p.iloc[-2]
                        p_5 = p.iloc[-6]
                        p_10 = p.iloc[-11]
                        p_20 = p.iloc[-21]
                        p_40 = p.iloc[-41]
                        
                        r_5_1 = (p_1 / p_5) - 1
                        r_10_5 = (p_5 / p_10) - 1
                        r_20_10 = (p_10 / p_20) - 1
                        r_40_20 = (p_20 / p_40) - 1
                        
                        score = (0.25 * r_5_1) + (0.25 * r_10_5) + (0.25 * r_20_10) + (0.25 * r_40_20)
                        
                        # Update latest row in DB
                        # Need to re-fetch or use existing object if session is consistent
                        latest_obj = history_rows[-1]
                        latest_obj.momentum_score = score
                        db.commit()
                    except IndexError:
                        pass
                        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_momentum_history(sector_ticker, period_days=252):
    """
    Fetches historical momentum scores for a given sector ticker.
    Returns DataFrame with Index=Date, Col=Score
    """
    db = next(get_db())
    try:
        sector = db.query(Sector).filter(Sector.ticker == sector_ticker).first()
        if not sector:
            return pd.DataFrame()
            
        # Calc start date
        end_date = db.query(func.max(PriceData.date)).scalar()
        if not end_date: return pd.DataFrame()
        start_date = end_date - timedelta(days=period_days)
        
        query = db.query(PriceData.date, PriceData.momentum_score).filter(
            PriceData.sector_id == sector.id,
            PriceData.date >= start_date,
            PriceData.momentum_score.isnot(None)
        ).order_by(PriceData.date)
        
        df = pd.read_sql(query.statement, db.bind)
        if not df.empty:
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            df.columns = ['Score'] # Rename momentum_score to Score
            df['Score'] = df['Score'] * 100 # Scale to percentage
        return df
    finally:
        db.close()

def get_sector_tickers(weight_type='cap'):
    """
    Returns a dictionary of sector names and their tickers based on weight type.
    weight_type: 'cap' or 'equal'
    """
    return {name: config[weight_type] for name, config in SECTORS_CONFIG.items()}

def get_sector_data_from_db(period_days=365, weight_type='cap'):
    """
    Reads data from the database filtering by sector type.
    """
    db = next(get_db())
    try:
        # Calculate start date
        start_date = date.today() - pd.Timedelta(days=period_days)
        
        results = db.query(PriceData.date, PriceData.close, Sector.ticker)\
                    .join(Sector)\
                    .filter(PriceData.date >= start_date)\
                    .filter(Sector.type == weight_type)\
                    .all()
        
        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['Date', 'Close', 'Ticker'])
        
        # Pivot: Index=Date, Columns=Ticker, Values=Close
        df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
        df_pivot.index = pd.to_datetime(df_pivot.index)
        
        return df_pivot.sort_index()
    finally:
        db.close()

def get_latest_performance(data):
    """
    Calculates percentage change based on the fetched data.
    """
    if data.empty:
        return pd.Series()
    
    return (data.iloc[-1] / data.iloc[0]) - 1

def get_sector_performance_matrix(weight_type='cap', periods=[5, 10, 20, 40, 252]):
    """
    Calculates returns for multiple periods for all sectors of a given type.
    """
    # Fetch enough data for the max period
    max_period = max(periods)
    # Add buffer for non-trading days (roughly 1.5x)
    buffer_days = int(max_period * 1.5) + 10 
    
    df = get_sector_data_from_db(period_days=buffer_days, weight_type=weight_type)
    
    if df.empty:
        return pd.DataFrame()
        
    # Ensure we have enough data (at least max_period rows)
    if len(df) < max_period:
        # In a real app we might wanr, but here just calculate what we can or return partial
        pass
        
    # Calculate returns
    results = {}
    current_price = df.iloc[-1]
    
    for p in periods:
        if len(df) > p:
            past_price = df.iloc[-(p+1)] # Price p days ago
            # Simple return: (Current - Past) / Past
            ret = (current_price / past_price) - 1
            results[f"{p}d"] = ret
        else:
            results[f"{p}d"] = None # Not enough data
            
    # Combine into DataFrame
    perf_df = pd.DataFrame(results)
    
    # Transpose so Index = Ticker
    # Currently index of ret is Ticker. So DataFrame(results) has Tickers as Index.
    
    # Combine into DataFrame
    perf_df = pd.DataFrame(results)
    
    # Multiply returns by 100
    perf_df = perf_df * 100
    
    # Add Last Price and Date
    perf_df['Last Price'] = current_price
    perf_df['Date'] = df.index[-1].strftime('%Y-%m-%d')
    
    return perf_df

def get_momentum_ranking(weight_type='cap'):
    """
    Calculates Momentum Ranking based on weighted returns.
    Formula:
    30% * Return (5d to 1d)
    30% * Return (10d to 5d)
    20% * Return (20d to 10d)
    20% * Return (40d to 20d)
    """
    # Need at least 40 days of history + buffer + lookback for past scores (50 days)
    # Total needed: 50 (max offset) + 40 (calc window) + buffer
    buffer_days = 200
    
    df = get_sector_data_from_db(period_days=buffer_days, weight_type=weight_type)
    
    if df.empty or len(df) < 100: # Need enough data
        return pd.DataFrame()
        
    def calculate_score_at_index(df_slice, idx_loc):
        """
        Calculates score for a given integer location index.
        idx_loc: The integer location in df_slice (e.g. -1 for latest, -6 for 5 days ago)
        """
        try:
            # We need p_1 (T-1), p_5 (T-5), etc. relative to the 'as of' date
            # If idx_loc is -1 (today), then p_1 is -2.
            # If idx_loc is -6 (5 days ago), then p_1 is -7.
            
            # Base index
            base = idx_loc
            
            p_1 = df_slice.iloc[base - 1]
            p_5 = df_slice.iloc[base - 5]
            p_10 = df_slice.iloc[base - 10]
            p_20 = df_slice.iloc[base - 20]
            p_40 = df_slice.iloc[base - 40]
            
            # Calculate Return Intervals
            r_5_1 = (p_1 / p_5) - 1
            r_10_5 = (p_5 / p_10) - 1
            r_20_10 = (p_10 / p_20) - 1
            r_40_20 = (p_20 / p_40) - 1
            
            # Score
            score = (0.25 * r_5_1) + (0.25 * r_10_5) + (0.25 * r_20_10) + (0.25 * r_40_20)
            return score, r_5_1, r_10_5, r_20_10, r_40_20
        except IndexError:
            return None, None, None, None, None

    # Calculate Current Score
    results = {}
    
    for ticker in df.columns:
        series = df[ticker].dropna()
        if len(series) < 50: continue
        
        # Current (-1)
        score, r5, r10, r20, r40 = calculate_score_at_index(series, -1)
        
        if score is None: continue

        # History
        score_5d, _, _, _, _ = calculate_score_at_index(series, -6)   # 5 days ago (1 + 5)
        score_20d, _, _, _, _ = calculate_score_at_index(series, -21) # 20 days ago (1 + 20)
        score_50d, _, _, _, _ = calculate_score_at_index(series, -51) # 50 days ago (1 + 50)
        
        # Calculate Change Ratio (Score / Score -5d)
        score_chg_5d = None
        if score_5d and score_5d != 0:
            score_chg_5d = score / score_5d
        
        results[ticker] = {
            'Score': score,
            'Score Chg (5d)': score_chg_5d,
            'Score -5d': score_5d,
            'Score -20d': score_20d,
            'Score -50d': score_50d,
            'R(5-1)': r5,
            'R(10-5)': r10,
            'R(20-10)': r20,
            'R(40-20)': r40,
            'Last Price': series.iloc[-1],
            'Date': series.index[-1].strftime('%Y-%m-%d')
        }
    
    rank_df = pd.DataFrame.from_dict(results, orient='index')
    
    if rank_df.empty:
        return pd.DataFrame()

    # Scale scores and returns to percentage
    # Note: Last Price is absolute, Date is string, Score Chg is a ratio (no scaling needed usually, or maybe scale?)
    # User asked for "Score today / Score 5 days ago". If Score=2.0 and Score-5d=1.0, Ratio=2.0.
    # Scores are scaled by 100 below.
    # If I calculate ratio using raw scores (0.02 / 0.01 = 2.0), it's the same as scaled (2.0 / 1.0 = 2.0).
    # So I don't need to scale the Ratio itself.
    
    scale_cols = ['Score', 'Score -5d', 'Score -20d', 'Score -50d', 'R(5-1)', 'R(10-5)', 'R(20-10)', 'R(40-20)']
    
    # Check if cols exist (some might be None if history missing for specific ticker)
    # Fill N/As in scores? Or leave as None? Leave as None/NaN
    rank_df[scale_cols] = rank_df[scale_cols] * 100
    
    return rank_df.sort_values(by='Score', ascending=False)

def import_constituents(sector_name, tickers):
    """
    Imports a list of tickers for a specific sector.
    """
    initialize_sectors_in_db()
    db = next(get_db())
    try:
        # Find sector
        sector = db.query(Sector).filter(Sector.name == sector_name, Sector.type == 'cap').first()
        if not sector:
            print(f"Sector {sector_name} not found.")
            return
        
        # Clear existing? Or append? Assuming replace for now to keep it clean if user re-uploads
        # Actually given the user might paste chunks, maybe upsert is better.
        # Let's do upsert (ignore if exists)
        
        current_tickers = {c.ticker for c in sector.constituents}
        count = 0
        for ticker in tickers:
            ticker = ticker.strip().upper()
            if ticker and ticker not in current_tickers:
                db.add(Constituent(sector_id=sector.id, ticker=ticker))
                current_tickers.add(ticker)
                count += 1
                
        db.commit()
        print(f"Imported {count} new tickers for {sector_name}.")
    except Exception as e:
        db.rollback()
        print(f"Error importing constituents: {e}")
    finally:
        db.close()

def get_latest_data_date():
    """Returns the most recent date found in ConstituentPrice table."""
    db = next(get_db())
    try:
        last_date = db.query(func.max(ConstituentPrice.date)).scalar()
        return last_date
    finally:
        db.close()

def update_constituents_data(sector_name=None, start_date=None, progress_callback=None):
    """
    Fetches data for constituents, calculates MAs/Flags, and stores in ConstituentPrice.
    
    Args:
        sector_name: Optional, filter by sector.
        start_date: Optional, updates only from this date forward (for gap fill).
        progress_callback: Optional, function(status_string, progress_float) to upadte UI.
    """
    db = next(get_db())
    try:
        query = db.query(Sector).filter(Sector.type == 'cap')
        if sector_name:
            query = query.filter(Sector.name == sector_name)
            
        sectors = query.all()
        total_sectors = len(sectors)
        
        for idx, sector in enumerate(sectors):
            try:
                constituents = {c.ticker: c.id for c in sector.constituents}
                if not constituents:
                    print(f"No constituents for {sector.name}, skipping.")
                    continue
                    
                msg = f"Updating {sector.name} ({len(constituents)} tickers)..."
                if start_date:
                    msg += f" [From {start_date}]"
                print(msg)
                
                # Update Progress
                if progress_callback:
                    progress_callback(msg, idx / total_sectors)
                
                ticker_list = list(constituents.keys())
                batch_size = 50
                
                for i in range(0, len(ticker_list), batch_size):
                    batch = ticker_list[i:i+batch_size]
                    print(f"  > Processing batch {i//batch_size + 1} ({len(batch)} tickers)...")
                    try:
                        # Determine start date for yfinance
                        # If start_date is provided (gap fill), use it.
                        # Otherwise default to "10y" for full history or substantial logic.
                        if start_date:
                            # yf.download(start=...) expects string or datetime
                            # We add a buffer? No, if we want specific range. 
                            # But MAs need previous data! 
                            # CRITICAL: To calculate MA200 for today, we need 200 days prior.
                            # Ideally, we should fetch (start_date - 300 days) to calculate MAs correctly for the gap.
                            
                            fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=365) # Safe buffer
                            data = yf.download(batch, start=fetch_start, auto_adjust=True, progress=False)['Close']
                        else:
                            data = yf.download(batch, period="10y", auto_adjust=True, progress=False)['Close']

                        if data.empty:
                            print(f"    ! Batch returned empty data.")
                            continue
                        
                        print(f"    - Downloaded {len(data)} rows of data.")
                        if isinstance(data, pd.Series):
                            data = data.to_frame()
                            
                        # Process each ticker
                        for ticker in data.columns:
                            if ticker not in constituents: continue
                            
                            series = data[ticker].dropna()
                            if series.empty: continue
                            
                            # Calculate MAs
                            ma5 = series.rolling(window=5).mean()
                            ma10 = series.rolling(window=10).mean()
                            ma20 = series.rolling(window=20).mean()
                            ma50 = series.rolling(window=50).mean()
                            ma200 = series.rolling(window=200).mean()
                            
                            # Prepare data for insertion
                            c_id = constituents[ticker]
                            
                            processed_df = pd.DataFrame({
                                'close': series,
                                'ma5': ma5,
                                'ma10': ma10,
                                'ma20': ma20,
                                'ma50': ma50,
                                'ma200': ma200
                            })
                            
                            # If gap filling, we only want to store the NEW dates, 
                            # BUT we needed the history to calculate the MAs.
                            if start_date:
                                processed_df = processed_df[processed_df.index.date > pd.to_datetime(start_date).date()]
                            else:
                                # If full update, store last 10 years (approx 2520 trading days)
                                processed_df = processed_df.tail(2520)
                            
                            if processed_df.empty:
                                continue

                            records_to_upsert = []
                            for dt, row in processed_df.iterrows():
                                date_val = dt.date()
                                
                                def get_flag(close, ma):
                                    return 1 if pd.notna(ma) and close > ma else (0 if pd.notna(ma) else None)
                                    
                                records_to_upsert.append({
                                    'constituent_id': c_id,
                                    'date': date_val,
                                    'close': float(row['close']),
                                    'ma5': float(row['ma5']) if pd.notna(row['ma5']) else None,
                                    'ma10': float(row['ma10']) if pd.notna(row['ma10']) else None,
                                    'ma20': float(row['ma20']) if pd.notna(row['ma20']) else None,
                                    'ma50': float(row['ma50']) if pd.notna(row['ma50']) else None,
                                    'ma200': float(row['ma200']) if pd.notna(row['ma200']) else None,
                                    'above_ma5': get_flag(row['close'], row['ma5']),
                                    'above_ma10': get_flag(row['close'], row['ma10']),
                                    'above_ma20': get_flag(row['close'], row['ma20']),
                                    'above_ma50': get_flag(row['close'], row['ma50']),
                                    'above_ma200': get_flag(row['close'], row['ma200'])
                                })

                            if records_to_upsert:
                                # Bulk Upsert using SQLite-specific "INSERT OR REPLACE" logic via SQLAlchemy insert
                                stmt = insert(ConstituentPrice).values(records_to_upsert)
                                # Map conflict to update
                                stmt = stmt.on_conflict_do_update(
                                    index_elements=['constituent_id', 'date'],
                                    set_={
                                        'close': stmt.excluded.close,
                                        'ma5': stmt.excluded.ma5,
                                        'ma10': stmt.excluded.ma10,
                                        'ma20': stmt.excluded.ma20,
                                        'ma50': stmt.excluded.ma50,
                                        'ma200': stmt.excluded.ma200,
                                        'above_ma5': stmt.excluded.above_ma5,
                                        'above_ma10': stmt.excluded.above_ma10,
                                        'above_ma20': stmt.excluded.above_ma20,
                                        'above_ma50': stmt.excluded.above_ma50,
                                        'above_ma200': stmt.excluded.above_ma200,
                                    }
                                )
                                db.execute(stmt)
                                print(f"      + Upserted {len(records_to_upsert)} price records for ticker mapping.")
                                    
                    except Exception as e:
                        print(f"Error processing batch {i} for {sector.name}: {e}")
                        
                db.commit()
                print(f"Constituents updated for {sector.name}.")
                
                # After updating constituents, update Breadth Metrics
                calculate_sector_breadth(sector.id, db)

                # Update New Highs/Lows (integrated)
                # Pass start_date to optimize: only write what we need
                calculate_sector_high_low(sector.id, db, start_date=start_date)

                # Update Progress
                if progress_callback:
                    progress_callback(msg, (idx + 1) / total_sectors)
                
                print(f"  Finished {sector.name}.")
                
            except Exception as e:
                print(f"CRITICAL ERROR updating sector {sector.name}: {e}")
                db.rollback()
                continue
            
        # Finish progress
        if progress_callback:
            progress_callback("Update Complete!", 1.0)
            
    finally:
        db.close()

from sqlalchemy import func

def calculate_sector_breadth(sector_id, db_session):
    """
    Aggregates ConstituentPrice flags to create BreadthMetrics.
    """
    # We want % Above MA per day
    # SQLAlchemy aggregation
    print(f"Calculating Breadth Metrics for SectorID {sector_id}...")
    
    subq = db_session.query(
        ConstituentPrice.date,
        func.count(ConstituentPrice.id).label('total_count'),
        func.sum(ConstituentPrice.above_ma5).label('sum_5'),
        func.sum(ConstituentPrice.above_ma10).label('sum_10'),
        func.sum(ConstituentPrice.above_ma20).label('sum_20'),
        func.sum(ConstituentPrice.above_ma50).label('sum_50'),
        func.sum(ConstituentPrice.above_ma200).label('sum_200')
    ).join(Constituent).filter(Constituent.sector_id == sector_id).group_by(ConstituentPrice.date).all()
    
    metrics_to_upsert = []
    for row in subq:
        date_val = row.date
        total = row.total_count
        
        if total == 0: continue
        
        # Add metrics to list
        def add_metric(metric_name, count_val):
            val = (count_val or 0) / total * 100
            metrics_to_upsert.append({
                'sector_id': sector_id,
                'date': date_val,
                'metric': metric_name,
                'value': float(val)
            })

        add_metric('pct_above_ma5', row.sum_5)
        add_metric('pct_above_ma10', row.sum_10)
        add_metric('pct_above_ma20', row.sum_20)
        add_metric('pct_above_ma50', row.sum_50)
        add_metric('pct_above_ma200', row.sum_200)

    if metrics_to_upsert:
        stmt = insert(BreadthMetric).values(metrics_to_upsert)
        stmt = stmt.on_conflict_do_update(
            index_elements=['sector_id', 'date', 'metric'],
            set_={'value': stmt.excluded.value}
        )
        db_session.execute(stmt)
        
    db_session.commit()
    print(f"Breadth aggregation complete ({len(metrics_to_upsert)} metrics calculated).")

def calculate_sector_high_low(sector_id, db_session, start_date=None):
    """
    Calculates rolling 252-day New Highs / New Lows for a sector.
    Integrated from backfill_new_highs.py.
    Optimization: If start_date is provided, only deletes/re-inserts metrics from that date.
    """
    label_msg = f" (from {start_date})" if start_date else ""
    print(f"  [High/Low] Fetching constituents for SectorID {sector_id}{label_msg}...")
    
    # Get constituents
    constituents = db_session.query(Constituent).filter(Constituent.sector_id == sector_id).all()
    if not constituents:
        print("  [High/Low] No constituents found.")
        return

    c_ids = [c.id for c in constituents]
    print(f"  [High/Low] Fetching prices for {len(c_ids)} tickers...")
    
    # Fetch only necessary prices
    # To calculate rolling 252 for dates >= start_date, we need at least 252 trading days 
    # of history BEFORE that date. ~400 calendar days is usually safe.
    
    if start_date:
        if isinstance(start_date, str):
            base_date = pd.to_datetime(start_date).date()
        elif isinstance(start_date, (pd.Timestamp, datetime)):
            base_date = start_date.date()
        else:
            base_date = start_date # assume it's already a date
        
        # We need data starting ~400 days before the base_date to get a valid 252-day window
        fetch_start = base_date - timedelta(days=400)
    else:
        # If no start_date, assume we want at least a year of history + window
        fetch_start = date.today() - timedelta(days=500)
    
    prices_q = db_session.query(ConstituentPrice.constituent_id, ConstituentPrice.date, ConstituentPrice.close)\
        .filter(ConstituentPrice.constituent_id.in_(c_ids))\
        .filter(ConstituentPrice.date >= fetch_start)\
        .order_by(ConstituentPrice.date)
    
    df = pd.read_sql(prices_q.statement, db_session.bind)
    
    if df.empty:
        return
        
    print(f"  [High/Low] Data fetched ({len(df)} rows). Calculating rolling windows...")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by constituent
    high_counts = {} 
    low_counts = {}
    
    # We want to update metrics for RECENT dates, essentially the ones we just added?
    # Or just recalculate strictly the latest date usually?
    # If we run this daily, we mostly care about the new days.
    # But let's calculate for all dates present in the DF to be consistent with the window.
    
    # Initialize for dates in DF
    unique_dates = df['date'].unique()
    for d in unique_dates:
        high_counts[d] = 0
        low_counts[d] = 0
        
    grouped = df.groupby('constituent_id')
    
    print(f"  [High/Low] Processing {len(grouped)} constituent groups...")
    for cid, group in grouped:
        group = group.sort_values('date')
        
        # Performance optimization:
        # Rolling Max/Min (252 days)
        rolling_max = group['close'].rolling(window=252, min_periods=252).max()
        rolling_min = group['close'].rolling(window=252, min_periods=252).min()
        
        # Identify Highs/Lows
        is_high = (group['close'] >= rolling_max - 1e-9)
        is_low = (group['close'] <= rolling_min + 1e-9)
        
        high_dates = group.loc[is_high, 'date']
        low_dates = group.loc[is_low, 'date']
        
        for d in high_dates:
            high_counts[d] += 1
        for d in low_dates:
            low_counts[d] += 1
            
    # Upsert into BreadthMetric
    # We only need to write metrics for dates that changed or are new.
    # For simplicity, we can upsert the calculated values.
    
    # Get distinct dates we have counts for
    sorted_dates = sorted(high_counts.keys())
    
    # Delete/Insert is safer for the window we processed.
    
    # NEW OPTIMIZATION: Only touch records >= target_date
    if start_date:
        if isinstance(start_date, str):
            target_date = pd.to_datetime(start_date).date()
        elif isinstance(start_date, (pd.Timestamp, datetime)):
            target_date = start_date.date()
        else:
            target_date = start_date
    else:
        target_date = min(unique_dates).date()
    
    # Delete existing High/Low metrics for the target window to allow overwriting
    db_session.query(BreadthMetric).filter(
        BreadthMetric.sector_id == sector_id,
        BreadthMetric.metric.in_(['new_highs_252', 'new_lows_252']),
        BreadthMetric.date >= target_date
    ).delete(synchronize_session=False)
    
    new_records = []
    
    for d in sorted_dates:
        d_date = d.date()
        
        # Only add records that are within our target update window
        if d_date < target_date:
            continue
            
        h_val = high_counts[d]
        l_val = low_counts[d]
        
        if h_val > 0:
            new_records.append(BreadthMetric(sector_id=sector_id, date=d_date, metric='new_highs_252', value=float(h_val)))
        
        if l_val > 0:
            new_records.append(BreadthMetric(sector_id=sector_id, date=d_date, metric='new_lows_252', value=float(l_val)))
            
    if new_records:
        db_session.bulk_save_objects(new_records)
        db_session.commit()
    print(f"High/Low Logic Complete. {len(new_records)} records updated.")


def get_etf_price_history(sector_name, days=1825, weight_type='equal'):
    """
    Fetches the historical price data for the ETF of the sector.
    weight_type: 'equal' or 'cap'
    Returns a DataFrame with 'Date' and 'Close'.
    """
    if sector_name not in SECTORS_CONFIG:
        return pd.DataFrame()
        
    ticker = SECTORS_CONFIG[sector_name][weight_type]
    
    db = next(get_db())
    try:
        start_date = date.today() - pd.Timedelta(days=days)
        sector = db.query(Sector).filter_by(ticker=ticker).first()
        
        if not sector:
            # Maybe it wasn't added if type='cap' (initially we might only have added cap/equal based on logic)
            # But initialize_sectors_in_db adds both. 
            # If failing, return empty.
            return pd.DataFrame()
            
        results = db.query(PriceData.date, PriceData.close)\
            .filter(PriceData.sector_id == sector.id)\
            .filter(PriceData.date >= start_date)\
            .order_by(PriceData.date).all()
            
        df = pd.DataFrame(results, columns=['Date', 'Close'])
        if not df.empty:
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
        return df
    finally:
        db.close()

def get_breadth_data(sector_name, metric='pct_above_ma50', days=1825):
    """
    Retrieves breadth data for plotting.
    """
    db = next(get_db())
    try:
        start_date = date.today() - pd.Timedelta(days=days)
        
        results = db.query(BreadthMetric.date, BreadthMetric.value)\
                    .join(Sector)\
                    .filter(Sector.name == sector_name)\
                    .filter(BreadthMetric.metric == metric)\
                    .filter(BreadthMetric.date >= start_date)\
                    .order_by(BreadthMetric.date)\
                    .all()
                    
        df = pd.DataFrame(results, columns=['Date', 'Value'])
        if not df.empty:
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
        return df
    finally:
        db.close()

def get_price_history(ticker, period_days=252):
    """
    Fetches price history for a ticker.
    Returns Series with Index=Date, Values=Close
    """
    db = next(get_db())
    try:
        # Find sector by ticker
        sector = db.query(Sector).filter(Sector.ticker == ticker).first()
        if not sector: return pd.Series()
        
        end_date = db.query(func.max(PriceData.date)).scalar()
        if not end_date: return pd.Series()
        start_date = end_date - timedelta(days=period_days)
        
        query = db.query(PriceData.date, PriceData.close).filter(
            PriceData.sector_id == sector.id,
            PriceData.date >= start_date
        ).order_by(PriceData.date)
        
        df = pd.read_sql(query.statement, db.bind)
        if not df.empty:
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            return df['close']
        return pd.Series()
    finally:
        db.close()

def get_all_sector_options():
    """
    Returns a list of dictionaries with valid sector options for UI.
    [{'name': 'Energy (XLE)', 'ticker': 'XLE', 'type': 'cap', 'sector': 'Energy'}, ...]
    """
    options = []
    # SECTORS_CONFIG is global in this file
    for name, config in SECTORS_CONFIG.items():
        options.append({
            'name': f"{name} ({config['cap']})", 
            'ticker': config['cap'],
            'type': 'cap',
            'sector': name
        })
        options.append({
            'name': f"{name} ({config['equal']})", 
            'ticker': config['equal'],
            'type': 'equal',
            'sector': name
        })
    return options

def get_dashboard_data(weight_type='cap'):
    """
    Returns a consolidated DataFrame for the dashboard.
    Columns: Sector, Score, Score -5d, -20d, -50d, % > MA5, 10, 20, 50, 200
    """
    # 1. Get Momentum Data (already has Score history)
    df_mom = get_momentum_ranking(weight_type=weight_type)
    if df_mom.empty:
        return pd.DataFrame()
        
    # df_mom has Index=Ticker, Coins: Sector, Score, Score -ND...
    
    # 2. Get Breadth Data
    # We need latest breadth for each sector in the list relative to weight_type
    # Helper to get sector name from ticker (Index of df_mom)
    
    db = next(get_db())
    try:
        breadth_results = []
        
        # Latest date for breadth to ensure freshness?
        # We'll just take the latest available for each sector/metric logic
        # OR query for the max date globally first.
        max_date = db.query(func.max(BreadthMetric.date)).scalar()
        
        if not max_date:
            return df_mom # Return just momentum if no breadth
            
        # Optimization: Fetch all metrics for max_date
        # Use Sector.name for mapping as it is consistent across weight types (Cap/Equal)
        records = db.query(BreadthMetric.value, BreadthMetric.metric, Sector.name)\
            .join(Sector)\
            .filter(BreadthMetric.date == max_date)\
            .all()
            
        # Process into dict: {SectorName: {metric: value}}
        breadth_map = {}
        for val, metric, s_name in records:
            if s_name not in breadth_map: breadth_map[s_name] = {}
            breadth_map[s_name][metric] = val
        
        # --- NEW: Get Breadth History (Last 5 days for MA20) ---
        # Get the date 5 periods ago
        # We need distinct dates available in BreadthMetric
        dates_subq = db.query(BreadthMetric.date)\
            .distinct()\
            .order_by(BreadthMetric.date.desc())\
            .limit(6)\
            .all() # Returns list of tuples [(date,), (date,)...]
        
        breadth_map_5d = {}
        if len(dates_subq) >= 6:
            date_5d = dates_subq[-1][0] # T-5 date
            
            # Fetch MA20 values for that date
            records_5d = db.query(BreadthMetric.value, Sector.name)\
                .join(Sector)\
                .filter(BreadthMetric.date == date_5d)\
                .filter(BreadthMetric.metric == 'pct_above_ma20')\
                .all()
            
            for val, s_name in records_5d:
                breadth_map_5d[s_name] = val
                
        # Merge into df_mom
        # df_mom has 'Sector' column with the name (e.g. "Energy")
        # WAIT: get_momentum_ranking returns DF with Index=Ticker. It does NOT have 'Sector' col.
        # We need to add it.
        sector_tickers = get_sector_tickers(weight_type=weight_type)
        # map: Name -> Ticker. Reverse it: Ticker -> Name
        ticker_to_name = {v: k for k, v in sector_tickers.items()}
        df_mom['Sector'] = df_mom.index.map(ticker_to_name)
        
        # Add columns to df_mom
        metrics = ['pct_above_ma5', 'pct_above_ma10', 'pct_above_ma20', 'pct_above_ma50', 'pct_above_ma200']
        
        for m in metrics:
            # Map using the 'Sector' column in df_mom
            df_mom[m] = df_mom['Sector'].map(lambda s: breadth_map.get(s, {}).get(m, None))
            
        # Add 5d MA20
        df_mom['pct_above_ma20_5d'] = df_mom['Sector'].map(lambda s: breadth_map_5d.get(s, None))
            
        return df_mom
    finally:
        db.close()

def get_sector_high_low_data(sector_name, days=252):
    """
    Returns DataFrame with New Highs and New Lows for a specific sector.
    Columns: Date, New Highs, New Lows, Net
    """
    db = next(get_db())
    try:
        # Calculate start date
        start_date = date.today() - pd.Timedelta(days=days)
        
        # Query metrics for the sector
        # We need sector_id from name
        # Fix: Filter by type='cap' to avoid MultipleResultsFound if both Cap and Equal exist with same name
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        
        if not sector_id:
            return pd.DataFrame()
            
        results = db.query(BreadthMetric.date, BreadthMetric.metric, BreadthMetric.value)\
            .filter(BreadthMetric.sector_id == sector_id)\
            .filter(BreadthMetric.date >= start_date)\
            .filter(BreadthMetric.metric.in_(['new_highs_252', 'new_lows_252']))\
            .all()
            
        if not results:
            return pd.DataFrame()
            
        # Convert to DF
        df = pd.DataFrame(results, columns=['Date', 'Metric', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pivot
        df_pivot = df.pivot(index='Date', columns='Metric', values='Value').fillna(0)
        
        # Ensure cols exist
        if 'new_highs_252' not in df_pivot.columns: df_pivot['new_highs_252'] = 0
        if 'new_lows_252' not in df_pivot.columns: df_pivot['new_lows_252'] = 0
        
        # Calculate Net
        df_pivot['Net'] = df_pivot['new_highs_252'] - df_pivot['new_lows_252']
        
        return df_pivot.sort_index()
        
    finally:
        db.close()

def get_sector_constituent_count(sector_name):
    """Returns the number of constituents in the sector."""
    db = next(get_db())
    try:
        # We need sector_id from name (cap)
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        if not sector_id:
            return 0
        return db.query(Constituent).filter(Constituent.sector_id == sector_id).count()
    finally:
        db.close()

def get_breadth_history(sector_name, metric, days=30):
    """
    Returns DataFrame with Date and Value for a specific breadth metric.
    """
    db = next(get_db())
    try:
        start_date = date.today() - pd.Timedelta(days=days)
        
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        if not sector_id:
            return pd.DataFrame()
            
        results = db.query(BreadthMetric.date, BreadthMetric.value)\
            .filter(BreadthMetric.sector_id == sector_id)\
            .filter(BreadthMetric.metric == metric)\
            .filter(BreadthMetric.date >= start_date)\
            .order_by(BreadthMetric.date)\
            .all()
            
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results, columns=['Date', 'Value'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    finally:
        db.close()

def get_sector_constituents(sector_name):
    """
    Returns a list of ticker symbols for the constituents of a sector.
    """
    db = next(get_db())
    try:
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        if not sector_id:
            return []
            
        tickers = db.query(Constituent.ticker)\
            .filter(Constituent.sector_id == sector_id)\
            .order_by(Constituent.ticker)\
            .all()
            
        return [t[0] for t in tickers]
    finally:
        db.close()



def get_active_constituent_history(sector_name,  days=3650):
    """
    Returns the count of constituents with data for each day.
    """
    db = next(get_db())
    try:
        start_date = date.today() - timedelta(days=days)
        
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        if not sector_id:
            return pd.DataFrame()

        # Count constituents with price data per day
        # JOIN ConstituentPrice -> Constituent -> Sector check is implicit if we filter by constituent.sector_id? 
        # Actually ConstituentPrice has constituent_id. Constituent has sector_id.
        
        results = db.query(ConstituentPrice.date, func.count(ConstituentPrice.constituent_id))\
            .join(Constituent, ConstituentPrice.constituent_id == Constituent.id)\
            .filter(Constituent.sector_id == sector_id)\
            .filter(ConstituentPrice.date >= start_date)\
            .group_by(ConstituentPrice.date)\
            .order_by(ConstituentPrice.date)\
            .all()
            
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results, columns=['Date', 'Count'])
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date')
        
    finally:
        db.close()


def get_stocks_up_history(sector_name, lookback_window=84, threshold=0.25, days_history=365):
    """
    Calculates the number and percentage of stocks in a sector that are up >= threshold over the lookback_window.
    Returns a DataFrame: Index=Date, Columns=['Count', 'Total', 'Percent'].
    """
    db = next(get_db())
    try:
        # 1. Get Sector ID
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        if not sector_id:
            return pd.DataFrame()

        # 2. Get Constituents
        constituents = db.query(Constituent.id, Constituent.ticker).filter(Constituent.sector_id == sector_id).all()
        if not constituents:
            return pd.DataFrame()
            
        c_ids = [c.id for c in constituents]
        c_map = {c.id: c.ticker for c in constituents}
        
        # 3. Fetch Prices
        # We need data for days_history + lookback_window
        start_date = date.today() - timedelta(days=days_history + lookback_window + 20) # Buffer
        
        results = db.query(ConstituentPrice.date, ConstituentPrice.constituent_id, ConstituentPrice.close)\
            .filter(ConstituentPrice.constituent_id.in_(c_ids))\
            .filter(ConstituentPrice.date >= start_date)\
            .all()
            
        if not results:
            return pd.DataFrame()
            
        # 4. Load into DataFrame
        df = pd.DataFrame(results, columns=['Date', 'ConstituentID', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Map IDs to Tickers (optional, but good for debugging)
        df['Ticker'] = df['ConstituentID'].map(c_map)
        
        # 5. Pivot
        df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
        
        # 6. Calculate Pct Change
        # pct_change(periods=N) calculates (Price_t / Price_{t-N}) - 1
        df_pct = df_pivot.pct_change(periods=lookback_window)
        
        # 7. Count Matches and Valid Data Points
        # Sum boolean (True=1) row-wise
        daily_counts = (df_pct >= threshold).sum(axis=1)
        
        # Count non-NaN values row-wise to get the dynamic denominator
        valid_counts = df_pct.notna().sum(axis=1)
        
        # 8. Trim to requested history
        result_df = pd.DataFrame({
            'Count': daily_counts,
            'Total': valid_counts
        })
        
        # Calculate Percent (Handle division by zero)
        result_df['Percent'] = 0.0
        mask = result_df['Total'] > 0
        result_df.loc[mask, 'Percent'] = (result_df.loc[mask, 'Count'] / result_df.loc[mask, 'Total']) * 100
        
        # Filter for last days_history
        disp_start = pd.Timestamp(date.today() - timedelta(days=days_history))
        result_df = result_df[result_df.index >= disp_start]
        
        return result_df
        
    finally:
        db.close()

