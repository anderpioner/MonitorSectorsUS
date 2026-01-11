import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from database import get_db, init_db
from models import Sector, PriceData, Constituent, BreadthMetric, ConstituentPrice
from sqlalchemy.dialects.sqlite import insert
from datetime import date, datetime, timedelta
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Version: 1.0.1
VERSION = "1.0.1"

# List of US Sector ETFs (SPDR and Invesco Equal Weight)
# Format: { 'Sector Name': {'cap': 'TickerCW', 'equal': 'TickerEW'} }
SECTORS_CONFIG = {
    'Communication Services Sector': {'cap': 'XLC', 'equal': 'RSPC'},
    'Consumer Cyclical Sector': {'cap': 'XLY', 'equal': 'RSPD'},
    'Consumer Defensive Sector': {'cap': 'XLP', 'equal': 'RSPS'},
    'Energy Sector': {'cap': 'XLE', 'equal': 'RSPG'},
    'Financial Services Sector': {'cap': 'XLF', 'equal': 'RSPF'},
    'Healthcare Sector': {'cap': 'XLV', 'equal': 'RSPH'},
    'Industrials Sector': {'cap': 'XLI', 'equal': 'RSPN'},
    'Basic Materials Sector': {'cap': 'XLB', 'equal': 'RSPM'},
    'Real Estate Sector': {'cap': 'XLRE', 'equal': 'RSPR'},
    'Technology Sector': {'cap': 'XLK', 'equal': 'RSPT'},
    'Utilities Sector': {'cap': 'XLU', 'equal': 'RSPU'}
}

SECTOR_ABBR = {
    'Communication Services Sector': 'Comm',
    'Consumer Cyclical Sector': 'Cyc',
    'Consumer Defensive Sector': 'Def',
    'Energy Sector': 'Energy',
    'Financial Services Sector': 'Fin',
    'Healthcare Sector': 'Health',
    'Industrials Sector': 'Ind',
    'Basic Materials Sector': 'Mat',
    'Real Estate Sector': 'R.Estate',
    'Technology Sector': 'Tech',
    'Utilities Sector': 'Util'
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
    raw_data = yf.download(tickers, period=period, auto_adjust=True)
    
    if raw_data.empty:
        print("No data downloaded.")
        return

    try:
        # Resolve Sector IDs and Objects
        sector_map = {s.ticker: s for s in db.query(Sector).all()}
        
        for ticker in tickers:
            if ticker not in sector_map:
                continue
                
            sector_obj = sector_map[ticker]
            
            # Extract data for specific ticker
            if len(tickers) > 1:
                if ticker not in raw_data['Close'].columns: continue
                series = pd.DataFrame({
                    'open': raw_data['Open'][ticker],
                    'high': raw_data['High'][ticker],
                    'low': raw_data['Low'][ticker],
                    'close': raw_data['Close'][ticker]
                }).dropna()
            else:
                series = raw_data[['Open', 'High', 'Low', 'Close']].dropna()
                series.columns = ['open', 'high', 'low', 'close']
            
            # Keep track if we added/updated anything for this ticker
            updated_ticker = False
            
            for dt, row in series.iterrows():
                # Check if exists
                date_val = dt.date()
                existing = db.query(PriceData).filter_by(sector_id=sector_obj.id, date=date_val).first()
                
                if not existing:
                    new_price = PriceData(
                        sector_id=sector_obj.id, 
                        date=date_val, 
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close'])
                    )
                    db.add(new_price)
                    updated_ticker = True
                else:
                    # Update if changed
                    changes = False
                    if existing.open is None or abs(existing.open - float(row['open'])) > 0.001:
                        existing.open = float(row['open'])
                        changes = True
                    if existing.high is None or abs(existing.high - float(row['high'])) > 0.001:
                        existing.high = float(row['high'])
                        changes = True
                    if existing.low is None or abs(existing.low - float(row['low'])) > 0.001:
                        existing.low = float(row['low'])
                        changes = True
                    if abs(existing.close - float(row['close'])) > 0.001:
                        existing.close = float(row['close'])
                        changes = True
                    
                    if changes:
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
                        p_0 = p.iloc[-1]
                        p_5 = p.iloc[-6]
                        p_10 = p.iloc[-11]
                        p_20 = p.iloc[-21]
                        p_40 = p.iloc[-41]
                        
                        r_5_0 = (p_0 / p_5) - 1
                        r_10_5 = (p_5 / p_10) - 1
                        r_20_10 = (p_10 / p_20) - 1
                        r_40_20 = (p_20 / p_40) - 1
                        
                        score = (0.25 * r_5_0) + (0.25 * r_10_5) + (0.25 * r_20_10) + (0.25 * r_40_20)
                        
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
    25% * Return (5d to 0d)
    25% * Return (10d to 5d)
    25% * Return (20d to 10d)
    25% * Return (40d to 20d)
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
            
            p_0 = df_slice.iloc[base]
            p_5 = df_slice.iloc[base - 5]
            p_10 = df_slice.iloc[base - 10]
            p_20 = df_slice.iloc[base - 20]
            p_40 = df_slice.iloc[base - 40]
            
            # Calculate Return Intervals
            r_5_0 = (p_0 / p_5) - 1
            r_10_5 = (p_5 / p_10) - 1
            r_20_10 = (p_10 / p_20) - 1
            r_40_20 = (p_20 / p_40) - 1
            
            # Score
            score = (0.25 * r_5_0) + (0.25 * r_10_5) + (0.25 * r_20_10) + (0.25 * r_40_20)
            return score, r_5_0, r_10_5, r_20_10, r_40_20
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
            'R(5-0)': r5,
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
    
    scale_cols = ['Score', 'Score -5d', 'Score -20d', 'Score -50d', 'R(5-0)', 'R(10-5)', 'R(20-10)', 'R(40-20)']
    
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

def fetch_batch_history(db, map_id_to_ticker, cutoff_date, days_needed=365):
    """
    Fetches historical data for a batch of tickers from DB to support indicator calculation.
    Returns dict: {ticker: pd.DataFrame(index=date, columns=[close, high, low])}
    """
    # Calculate fetch start for DB
    # We need ~252 trading days. 365 calendar days + buffer is safe.
    db_start = pd.to_datetime(cutoff_date) - pd.Timedelta(days=days_needed + 50) 
    
    ids = list(map_id_to_ticker.keys())
    
    # We need to explicitly select columns to avoid huge objects if extended
    query = db.query(
        ConstituentPrice.constituent_id, 
        ConstituentPrice.date,
        ConstituentPrice.close, 
        ConstituentPrice.open,
        ConstituentPrice.high, 
        ConstituentPrice.low
    ).filter(
        ConstituentPrice.constituent_id.in_(ids),
        ConstituentPrice.date >= db_start.date(),
        ConstituentPrice.date <= pd.to_datetime(cutoff_date).date() # Include overlap date if any to stitch
    )
    
    rows = query_rows = query.all()
    if not query_rows:
        return {}
        
    # Process into dict of DFs
    from collections import defaultdict
    data_by_id = defaultdict(list)
    for r in query_rows:
        # Check for None values in High/Low and fallback to Close if needed
        h = r.high if r.high is not None else r.close
        l = r.low if r.low is not None else r.close
        
        data_by_id[r.constituent_id].append({
            'date': pd.to_datetime(r.date),
            'open': float(r.open if r.open is not None else r.close),
            'close': float(r.close),
            'high': float(h),
            'low': float(l)
        })
        
    result = {}
    for cid, records in data_by_id.items():
        if not records: continue
        ticker = map_id_to_ticker[cid]
        df = pd.DataFrame(records)
        if not df.empty:
            df.set_index('date', inplace=True)
            result[ticker] = df.sort_index()
        
    return result

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
            # Robust check for list/iterable
            if not isinstance(sector_name, str) and hasattr(sector_name, '__iter__'):
                print(f"Filtering by list of sectors: {sector_name}")
                query = query.filter(Sector.name.in_(sector_name))
            else:
                print(f"Filtering by single sector: {sector_name}")
                query = query.filter(Sector.name == sector_name)
            
        sectors = query.all()
        total_sectors = len(sectors)
        
        # Tracking variables for reporting
        all_sector_names = [s.name for s in sectors]
        updated_sectors = []
        
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
                batch_size = 20 # Reduced from 50 to 20 to avoid timeouts/freezes
                
                for i in range(0, len(ticker_list), batch_size):
                    batch = ticker_list[i:i+batch_size]
                    print(f"  > Processing batch {i//batch_size + 1} ({len(batch)} tickers)...")
                    
                    # Add delay to avoid rate limiting
                    time.sleep(1) 
                    
                    try:
                        # Determine start date for yfinance
                        print(f"    - Starting download for {batch[0]}...{batch[-1]}")
                        
                        batch_history = {}
                        try:
                            if start_date:
                                # New Optimization: Fetch history from DB
                                fetch_start = pd.to_datetime(start_date)
                                batch_mapping = {constituents[t]: t for t in batch}
                                batch_history = fetch_batch_history(db, batch_mapping, fetch_start)
                                
                                # Download from start_date
                                raw_data = yf.download(batch, start=fetch_start, auto_adjust=True, progress=False, timeout=30)
                            else:
                                raw_data = yf.download(batch, period="10y", auto_adjust=True, progress=False, timeout=30)
                        except Exception as e_down:
                             print(f"    ! Download failed for batch: {e_down}")
                             continue
                             
                        print("    - Download finished.")

                        if raw_data.empty:
                            print(f"    ! Batch returned empty data.")
                            continue
                        
                        # Normalize Data Access
                        # Loop through the batch tickers and extract data for each
                        for ticker in batch:
                            if ticker not in constituents: continue
                            
                            # Extract data for specific ticker
                            try:
                                if len(batch) > 1:
                                    # MultiIndex: (Price, Ticker)
                                    # Need to handle cases where some tickers fail
                                    if ticker not in raw_data['Close'].columns:
                                        continue
                                    t_open = raw_data['Open'][ticker]
                                    t_close = raw_data['Close'][ticker]
                                    t_high = raw_data['High'][ticker]
                                    t_low = raw_data['Low'][ticker]
                                else:
                                    # Single Index: (Price)
                                    t_open = raw_data['Open']
                                    t_close = raw_data['Close']
                                    t_high = raw_data['High']
                                    t_low = raw_data['Low']
                            except KeyError:
                                continue

                            # Combine into DataFrame and Drop NaNs
                            df_t = pd.DataFrame({
                                'open': t_open,
                                'close': t_close, 
                                'high': t_high, 
                                'low': t_low
                            }).dropna()
                            
                            if df_t.empty: continue
                            
                            # Stitch History if needed
                            if start_date and ticker in batch_history:
                                hist_df = batch_history[ticker]
                                combined_df = pd.concat([hist_df, df_t])
                                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                                combined_df.sort_index(inplace=True)
                                calc_df = combined_df
                            else:
                                calc_df = df_t
                            
                            opens = calc_df['open']
                            series = calc_df['close']
                            highs = calc_df['high']
                            lows = calc_df['low']

                            # --- Calculate Indicators ---
                            
                            # SMAs
                            ma5 = series.rolling(window=5).mean()
                            ma10 = series.rolling(window=10).mean()
                            ma20 = series.rolling(window=20).mean()
                            ma50 = series.rolling(window=50).mean()
                            ma200 = series.rolling(window=200).mean()
                            
                            # EMAs
                            ema8 = series.ewm(span=8, adjust=False).mean()
                            ema20 = series.ewm(span=20, adjust=False).mean()
                            ema50 = series.ewm(span=50, adjust=False).mean()
                            
                            # ATR (14)
                            # TR = Max(H-L, Abs(H-Cp), Abs(L-Cp))
                            prev_close = series.shift(1)
                            tr1 = highs - lows
                            tr2 = (highs - prev_close).abs()
                            tr3 = (lows - prev_close).abs()
                            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                            # Wilder's Smoothing: alpha=1/14
                            atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
                            
                            # Prepare data for insertion
                            c_id = constituents[ticker]
                            
                            processed_df = pd.DataFrame({
                                'open': opens,
                                'close': series,
                                'high': highs,
                                'low': lows,
                                'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
                                'ema8': ema8, 'ema20': ema20, 'ema50': ema50,
                                'atr14': atr14
                            })
                            
                            # Filter by date if needed
                            if start_date:
                                processed_df = processed_df[processed_df.index.date > pd.to_datetime(start_date).date()]
                            else:
                                processed_df = processed_df.tail(2520)
                            
                            if processed_df.empty:
                                continue

                            records_to_upsert = []
                            for dt, row in processed_df.iterrows():
                                date_val = dt.date()
                                
                                def get_flag(close, ma):
                                    return 1 if pd.notna(ma) and close > ma else (0 if pd.notna(ma) else None)
                                
                                # Helper to clean NaN
                                def val(v): return float(v) if pd.notna(v) else None
                                
                                records_to_upsert.append({
                                    'constituent_id': c_id,
                                    'date': date_val,
                                    'open': val(row['open']),
                                    'close': float(row['close']),
                                    'high': val(row['high']),
                                    'low': val(row['low']),
                                    'ma5': val(row['ma5']),
                                    'ma10': val(row['ma10']),
                                    'ma20': val(row['ma20']),
                                    'ma50': val(row['ma50']),
                                    'ma200': val(row['ma200']),
                                    'ema8': val(row['ema8']),
                                    'ema20': val(row['ema20']),
                                    'ema50': val(row['ema50']),
                                    'atr14': val(row['atr14']),
                                    'above_ma5': get_flag(row['close'], row['ma5']),
                                    'above_ma10': get_flag(row['close'], row['ma10']),
                                    'above_ma20': get_flag(row['close'], row['ma20']),
                                    'above_ma50': get_flag(row['close'], row['ma50']),
                                    'above_ma200': get_flag(row['close'], row['ma200'])
                                })

                            if records_to_upsert:
                                upsert_batch_size = 500
                                total_recs = len(records_to_upsert)
                                
                                for k in range(0, total_recs, upsert_batch_size):
                                    batch_recs = records_to_upsert[k : k + upsert_batch_size]
                                    stmt = insert(ConstituentPrice).values(batch_recs)
                                    stmt = stmt.on_conflict_do_update(
                                        index_elements=['constituent_id', 'date'],
                                        set_={
                                            'open': stmt.excluded.open,
                                            'close': stmt.excluded.close,
                                            'high': stmt.excluded.high,
                                            'low': stmt.excluded.low,
                                            'ma5': stmt.excluded.ma5,
                                            'ma10': stmt.excluded.ma10,
                                            'ma20': stmt.excluded.ma20,
                                            'ma50': stmt.excluded.ma50,
                                            'ma200': stmt.excluded.ma200,
                                            'ema8': stmt.excluded.ema8,
                                            'ema20': stmt.excluded.ema20,
                                            'ema50': stmt.excluded.ema50,
                                            'atr14': stmt.excluded.atr14,
                                            'above_ma5': stmt.excluded.above_ma5,
                                            'above_ma10': stmt.excluded.above_ma10,
                                            'above_ma20': stmt.excluded.above_ma20,
                                            'above_ma50': stmt.excluded.above_ma50,
                                            'above_ma200': stmt.excluded.above_ma200,
                                        }
                                    )
                                    db.execute(stmt)
                                
                                min_date = min(r['date'] for r in records_to_upsert)
                                max_date = max(r['date'] for r in records_to_upsert)
                                print(f"      + Upserted {len(records_to_upsert)} records. Range: {min_date} to {max_date}. (Batched)")
                                    
                    except Exception as e:
                        print(f"Error processing batch {i} for {sector.name}: {e}")
                        
                db.commit()
                print(f"Constituents updated for {sector.name}.")
                
                # After updating constituents, update Breadth Metrics
                calculate_sector_breadth(sector.id, db)

                # Update New Highs/Lows (integrated)
                # Pass start_date to optimize: only write what we need
                calculate_sector_high_low(sector.id, db, start_date=start_date)

                # Update Stocks > 25%, 50%, 100% Metrics
                calculate_sector_up_metrics(sector.id, db, start_date=start_date, lookback_window=84, thresholds=[0.25, 0.50, 1.00])

                # Update Active Constituent Count
                calculate_active_count(sector.id, db, start_date=start_date)

                # Update EMA Trend Setup Metric (New)
                calculate_ema_setup_counts(sector.ticker, db, start_date=start_date)

                # Update Progress
                if progress_callback:
                    progress_callback(msg, (idx + 1) / total_sectors)
                
                print(f"  Finished {sector.name}.")

                # --- Status Report per User Request ---
                updated_sectors.append(sector.name)
                pending_sectors = [s for s in all_sector_names if s not in updated_sectors]
                
                print(f"  [Status Report]")
                print(f"  > Updated ({len(updated_sectors)}): {', '.join(updated_sectors)}")
                if pending_sectors:
                    print(f"  > Pending ({len(pending_sectors)}): {', '.join(pending_sectors)}")
                else:
                    print(f"  > All sectors updated.")
                print("-" * 50)
                # --------------------------------------
                
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
        batch_size = 500
        for i in range(0, len(metrics_to_upsert), batch_size):
            batch = metrics_to_upsert[i:i+batch_size]
            stmt = insert(BreadthMetric).values(batch)
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
    Retrieves from BreadthMetric for speed.
    """
    db = next(get_db())
    try:
        start_date = date.today() - timedelta(days=days)
        
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        if not sector_id:
            return pd.DataFrame()

        results = db.query(BreadthMetric.date, BreadthMetric.value)\
            .filter(BreadthMetric.sector_id == sector_id)\
            .filter(BreadthMetric.metric == 'active_count')\
            .filter(BreadthMetric.date >= start_date)\
            .order_by(BreadthMetric.date)\
            .all()
            
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results, columns=['Date', 'Count'])
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date')
        
    finally:
        db.close()


def get_stocks_up_history(sector_name, metric_name='pct_up_25_84d', days_history=365):
    """
    Retrieves the pre-calculated percentage of stocks exceeding a threshold.
    Metric names: 'pct_up_25_84d', 'pct_up_50_84d', 'pct_up_100_84d'.
    Returns a DataFrame: Index=Date, Columns=['Percent'].
    """
    db = next(get_db())
    try:
        # 1. Get Sector ID
        sector_id = db.query(Sector.id).filter(Sector.name == sector_name, Sector.type == 'cap').scalar()
        if not sector_id:
            return pd.DataFrame()

        # 2. Query BreadthMetric
        start_date = date.today() - timedelta(days=days_history + 20)
        
        results = db.query(BreadthMetric.date, BreadthMetric.value)\
            .filter(BreadthMetric.sector_id == sector_id)\
            .filter(BreadthMetric.metric == metric_name)\
            .filter(BreadthMetric.date >= start_date)\
            .order_by(BreadthMetric.date)\
            .all()
            
        if not results:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(results, columns=['Date', 'Percent'])
        result_df['Date'] = pd.to_datetime(result_df['Date'])
        result_df.set_index('Date', inplace=True)
        
        # Filter for last days_history
        disp_start = pd.Timestamp(date.today() - timedelta(days=days_history))
        result_df = result_df[result_df.index >= disp_start]
        
        return result_df
        
    finally:
        db.close()





def calculate_sector_up_metrics(sector_id, db_session, start_date=None, lookback_window=84, thresholds=[0.25, 0.50, 1.00]):
    """
    Calculates the 'Stocks > X% in 84d' metrics and stores them in BreadthMetric table.
    Metric Names: 'pct_up_25_84d', 'pct_up_50_84d', 'pct_up_100_84d'
    """
    try:
        # Get Constituents
        constituents = db_session.query(Constituent.id).filter(Constituent.sector_id == sector_id).all()
        if not constituents:
            return
            
        c_ids = [c.id for c in constituents]
        
        # Determine fetch start date
        if start_date:
            fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_window + 20)
        else:
            fetch_start = date.today() - pd.Timedelta(days=3650 + lookback_window + 20)
            
        # Fetch Prices
        prices_q = db_session.query(ConstituentPrice.date, ConstituentPrice.constituent_id, ConstituentPrice.close)\
            .filter(ConstituentPrice.constituent_id.in_(c_ids))\
            .filter(ConstituentPrice.date >= fetch_start)\
            .order_by(ConstituentPrice.date)
            
        df = pd.read_sql(prices_q.statement, db_session.bind)
        
        if df.empty:
            return
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot
        df_pivot = df.pivot(index='date', columns='constituent_id', values='close')
        
        # Calculate Pct Change
        df_pct = df_pivot.pct_change(periods=lookback_window)
        
        valid_counts = df_pct.notna().sum(axis=1) # Common denominator
        
        # Filter dates if start_date provided
        if start_date:
             process_mask = df_pct.index >= pd.to_datetime(start_date)
             valid_counts = valid_counts[process_mask]
             df_process_pct = df_pct[process_mask]
        else:
             df_process_pct = df_pct

        metrics_to_upsert = []
        
        # Iterate over thresholds
        for threshold in thresholds:
            t_label = int(threshold * 100)
            metric_name = f'pct_up_{t_label}_84d'
            
            daily_counts = (df_process_pct >= threshold).sum(axis=1)
            
            for dt, count in daily_counts.items():
                total = valid_counts.loc[dt]
                if total > 0:
                    val = (count / total) * 100
                    metrics_to_upsert.append({
                        'sector_id': sector_id,
                        'date': dt.date(),
                        'metric': metric_name,
                        'value': float(val)
                    })

        # Bulk Upsert
        # Bulk Upsert
        if metrics_to_upsert:
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            
            batch_size = 500
            for i in range(0, len(metrics_to_upsert), batch_size):
                batch = metrics_to_upsert[i:i+batch_size]

                stmt = insert(BreadthMetric).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['sector_id', 'date', 'metric'],
                    set_={'value': stmt.excluded.value}
                )
                db_session.execute(stmt)
            db_session.commit()
            
        print(f"  > Metrics updated for thresholds {thresholds}. Total records: {len(metrics_to_upsert)}.")
        
    except Exception as e:
        print(f"Error calculating up metrics: {e}")
        db_session.rollback()


def calculate_active_count(sector_id, db_session, start_date=None):
    """
    Calculates the number of active constituents (with price data) per day and stores it.
    Metric Name: 'active_count'
    """
    try:
        # Determine fetch start date
        if start_date:
            fetch_start = pd.to_datetime(start_date)
        else:
            fetch_start = date.today() - pd.Timedelta(days=3650 + 20)
            
        # Helper to get constituent IDs (optional filter if needed, but we join anyway)
        # Actually simplest query is grouping by date on ConstituentPrice joined with Sector
        
        # We need to filter by sector.
        # Query: Count(distinct constituent_id) group by date where sector_id = X
        
        results = db_session.query(ConstituentPrice.date, func.count(ConstituentPrice.constituent_id))\
            .join(Constituent, ConstituentPrice.constituent_id == Constituent.id)\
            .filter(Constituent.sector_id == sector_id)\
            .filter(ConstituentPrice.date >= fetch_start)\
            .group_by(ConstituentPrice.date)\
            .all()
            
        metrics_to_upsert = []
        for row in results:
            dt = row[0] # date object
            count = row[1]
            
            metrics_to_upsert.append({
                'sector_id': sector_id,
                'date': dt,
                'metric': 'active_count',
                'value': float(count)
            })
            
        # Bulk Upsert
        if metrics_to_upsert:
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            
            batch_size = 500
            for i in range(0, len(metrics_to_upsert), batch_size):
                batch = metrics_to_upsert[i:i+batch_size]
                
                stmt = insert(BreadthMetric).values(batch)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['sector_id', 'date', 'metric'],
                    set_={'value': stmt.excluded.value}
                )
                db_session.execute(stmt)
            db_session.commit()
            
        print(f"  > Calculate 'active_count': Updated {len(metrics_to_upsert)} records.")
        
    except Exception as e:
        print(f"Error calculating active count: {e}")
        db_session.rollback()


def get_sectors_for_tickers(tickers: list) -> dict:
    """
    Identifies the sector for a list of tickers.
    Returns: dict {ticker: sector_name}
    """
    db = next(get_db())
    try:
        # Clean inputs
        clean_tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        if not clean_tickers:
            return {}
            
        results = db.query(Constituent.ticker, Sector.name)\
            .join(Sector)\
            .filter(Constituent.ticker.in_(clean_tickers))\
            .all()
            
        return {r[0]: r[1] for r in results}
    finally:
        db.close()

def get_sector_counts() -> dict:
    """
    Returns the total number of constituents per sector in the database.
    Returns: dict {sector_name: count}
    """
    db = next(get_db())
    try:
        results = db.query(Sector.name, func.count(Constituent.id))\
            .join(Constituent)\
            .filter(Sector.type == 'cap')\
            .group_by(Sector.name)\
            .all()
            
        return {r[0]: r[1] for r in results}
    finally:
        db.close()

def get_sector_constituents(sector_ticker):
    """
    Returns a list of tickers belonging to the sector identified by sector_ticker.
    """
    db = next(get_db())
    try:
        results = db.query(Constituent.ticker)\
            .join(Sector)\
            .filter(Sector.ticker == sector_ticker)\
            .all()
        return [r[0] for r in results]
    finally:
        db.close()

def calculate_ema_setup_counts(sector_ticker, db_session, start_date=None):
    """
    Calculates the number of stocks in the sector that meet the EMA Trend Setup criteria:
    EMA8 > EMA20 AND EMA20 > EMA50 AND Close > EMA20
    """
    # Get Sector ID
    sector = db_session.query(Sector).filter_by(ticker=sector_ticker).first()
    if not sector:
        print(f"Sector {sector_ticker} not found")
        return
    sector_id = sector.id

    # Get all constituents for the sector
    constituents = get_sector_constituents(sector_ticker)
    if not constituents:
        return
        
    start_time = time.time()
    
    # Query Data - Explicitly join with Constituent to filter by Sector if needed, 
    # but get_sector_constituents already gives us the list
    query = db_session.query(Constituent.ticker, ConstituentPrice.date, ConstituentPrice.close, ConstituentPrice.ema8, ConstituentPrice.ema20, ConstituentPrice.ema50)\
        .join(Constituent)\
        .filter(Constituent.ticker.in_(constituents))
    
    if start_date:
        query = query.filter(ConstituentPrice.date >= start_date)
        
    df = pd.read_sql(query.statement, db_session.bind)
    
    if df.empty:
        return

    # Filter Criteria
    # EMA8 > EMA20 AND EMA20 > EMA50 AND Close > EMA20
    condition = (df['ema8'] > df['ema20']) & (df['ema20'] > df['ema50']) & (df['close'] > df['ema20'])
    
    df_filtered = df[condition]
    
    # Count per date
    daily_counts = df_filtered.groupby('date')['ticker'].count().reset_index()
    daily_counts.columns = ['date', 'value']
    
    # Insert/Update BreadthMetric
    metric_key = 'ema_trend_setup'
    
    # Bulk optimization or iterate? Iterate is safer for upsert logic unless we have bulk_upsert
    for _, row in daily_counts.iterrows():
        val = int(row['value'])
        
        # Use sector_id instead of sector_ticker
        existing = db_session.query(BreadthMetric).filter_by(
            sector_id=sector_id,
            date=row['date'],
            metric=metric_key
        ).first()
        
        if existing:
            existing.value = val
        else:
            new_metric = BreadthMetric(
                sector_id=sector_id,
                date=row['date'],
                metric=metric_key,
                value=val
            )
            db_session.add(new_metric)
            
    db_session.commit()
    print(f"[{sector_ticker}] Updated {metric_key} for {len(daily_counts)} days. Time: {time.time() - start_time:.2f}s")


def get_sector_fear_greed(sector_name, days=365):
    """
    Calculates a composite Fear & Greed Index (0-100) for the sector.
    Components:
    1. Breadth Strength (60%): Average of % > MA20, % > MA50, % > MA200
    2. Momentum (20%): RSI(14) of Sector ETF (Normalized 0-100)
    3. Trend Quality (20%): EMA Setup % (Normalized relative to its own recent history? No, use absolute 0-100)
    
    Actually, let's keep it simple and robust:
    - Pct > MA20
    - Pct > MA50
    - Pct > MA200
    - RSI(14) ETF
    
    Returns: Series with Index=Date, Values=Index(0-100)
    """
    # 1. Fetch Breadth Data
    df_ma20 = get_breadth_data(sector_name, metric='pct_above_ma20', days=days)
    df_ma50 = get_breadth_data(sector_name, metric='pct_above_ma50', days=days)
    df_ma200 = get_breadth_data(sector_name, metric='pct_above_ma200', days=days)
    
    # 2. Fetch ETF Price for Momentum (RSI)
    df_etf = get_etf_price_history(sector_name, days=days + 30, weight_type='cap') # Extra buffer for RSI
    
    if df_ma50 is None or df_ma50.empty or df_etf.empty:
        return pd.Series()
        
    # Align Data
    # Rename cols
    df_all = df_ma50.rename(columns={'Value': 'ma50'})
    if df_ma20 is not None and not df_ma20.empty:
        df_all = df_all.join(df_ma20.rename(columns={'Value': 'ma20'}), how='inner')
    else:
        df_all['ma20'] = df_all['ma50'] # Fallback
        
    if df_ma200 is not None and not df_ma200.empty:
        df_all = df_all.join(df_ma200.rename(columns={'Value': 'ma200'}), how='inner')
    else:
        df_all['ma200'] = df_all['ma50'] # Fallback
        
    df_all = df_all.join(df_etf.rename(columns={'Close': 'price'}), how='inner')
    
    # Calculate RSI(14)
    delta = df_all['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_all['rsi'] = 100 - (100 / (1 + rs))
    
    # Fill NaN RSI (first 14 days)
    df_all['rsi'] = df_all['rsi'].fillna(50)
    
    # Composite Score Calculation
    # Weights: Breadth 60% (20% each), RSI 40%? Or Equal?
    # Let's try equal weights for robustness -> 25% each
    df_all['fng'] = (
        df_all['ma20'] * 0.25 + 
        df_all['ma50'] * 0.25 + 
        df_all['ma200'] * 0.25 + 
        df_all['rsi'] * 0.25
    )
    
    return df_all['fng']


def get_sector_fear_greed(sector_name, days=365):
    """
    Calculates a composite Fear & Greed Index (0-100) for the sector.
    Components:
    1. Breadth Strength (60%): Average of % > MA20, % > MA50, % > MA200
    2. Momentum (20%): RSI(14) of Sector ETF (Normalized 0-100)
    3. Trend Quality (20%): EMA Setup % (Normalized relative to its own recent history? No, use absolute 0-100)
    
    Actually, let's keep it simple and robust:
    - Pct > MA20
    - Pct > MA50
    - Pct > MA200
    - RSI(14) ETF
    
    Returns: Series with Index=Date, Values=Index(0-100)
    """
    # 1. Fetch Breadth Data
    df_ma20 = get_breadth_data(sector_name, metric='pct_above_ma20', days=days)
    df_ma50 = get_breadth_data(sector_name, metric='pct_above_ma50', days=days)
    df_ma200 = get_breadth_data(sector_name, metric='pct_above_ma200', days=days)
    
    # 2. Fetch ETF Price for Momentum (RSI)
    df_etf = get_etf_price_history(sector_name, days=days + 30, weight_type='cap') # Extra buffer for RSI
    
    if df_ma50 is None or df_ma50.empty or df_etf.empty:
        return pd.Series()
        
    # Align Data
    # Rename cols
    df_all = df_ma50.rename(columns={'Value': 'ma50'})
    if df_ma20 is not None and not df_ma20.empty:
        df_all = df_all.join(df_ma20.rename(columns={'Value': 'ma20'}), how='inner')
    else:
        df_all['ma20'] = df_all['ma50'] # Fallback
        
    if df_ma200 is not None and not df_ma200.empty:
        df_all = df_all.join(df_ma200.rename(columns={'Value': 'ma200'}), how='inner')
    else:
        df_all['ma200'] = df_all['ma50'] # Fallback
        
    df_all = df_all.join(df_etf.rename(columns={'Close': 'price'}), how='inner')
    
    # Calculate RSI(14)
    delta = df_all['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_all['rsi'] = 100 - (100 / (1 + rs))
    
    # Fill NaN RSI (first 14 days)
    df_all['rsi'] = df_all['rsi'].fillna(50)
    
    # Composite Score Calculation
    # Weights: Breadth 60% (20% each), RSI 40%? Or Equal?
    # Let's try equal weights for robustness -> 25% each
    df_all['fng'] = (
        df_all['ma20'] * 0.25 + 
        df_all['ma50'] * 0.25 + 
        df_all['ma200'] * 0.25 + 
        df_all['rsi'] * 0.25
    )
    
    return df_all['fng']

def get_atr_variation_stats(target_date=None):
    """
    Aggregates ATR variation stats for Sectors and Industries.
    Criteria: Abs(Close - PrevClose) > ATR14
    
    Returns: DataFrame with columns [Sector, Industry, Ticker, Close, PctChange, ATR14, DeltaATR, AboveATR]
    """
    db = next(get_db())
    try:
        if not target_date:
            target_date = db.query(func.max(ConstituentPrice.date)).scalar()
        
        if not target_date:
            return pd.DataFrame()

        # Fetch data for target_date and prev_date (for pct_change and prev_close)
        # Actually, we can fetch just T and assume we can calc change from close/prev or if valid.
        # But easier to fetch today's data w/ ATR.
        
        # We need: Ticker, Sector, Industry, Close (Today), ATR14 (Today), PrevClose (to calc Delta)
        # PrevClose might not be in the same row.
        # Alternatively, if we just want "Change", we can use (Open vs Close)? No, request is "Variation", usually Close-Close.
        
        # Optimized query: Fetch T and T-1 for all constituents.
        # To simplify: Let's fetch the last 2 records for every constituent that has data on target_date.
        
        # Step 1: Get all constituents with data on target_date
        # (Using a subquery or join might be heavy on SQLite if naive. Let's do a join.)
        
        # Query: Get PriceData for Target Date
        query = db.query(
            Constituent.ticker, 
            Sector.name.label('Sector'), 
            Constituent.industry.label('Industry'),
            ConstituentPrice.close,
            ConstituentPrice.atr14,
            ConstituentPrice.constituent_id
        ).select_from(ConstituentPrice).join(Constituent).join(Sector).filter(
            ConstituentPrice.date == target_date
        )
        
        df_today = pd.read_sql(query.statement, db.bind)
        
        if df_today.empty:
            return pd.DataFrame()
        
        # Step 2: Get PriceData for Previous Date available for these constituents
        # It's expensive to do "Previous Date" per ticker in SQL.
        # Global previous date is easier.
        prev_global_date = db.query(func.max(ConstituentPrice.date)).filter(ConstituentPrice.date < target_date).scalar()
        
        if not prev_global_date:
            return pd.DataFrame()
            
        query_prev = db.query(
            ConstituentPrice.constituent_id,
            ConstituentPrice.close.label('prev_close')
        ).filter(
            ConstituentPrice.date == prev_global_date
        )
        
        df_prev = pd.read_sql(query_prev.statement, db.bind)
        
        # Merge
        merged = pd.merge(df_today, df_prev, on='constituent_id', how='left')
        merged = merged.dropna(subset=['prev_close', 'atr14']) # Need both to calc
        
        # Calculate Metrics
        # User request: Only count POSITIVE variation (Strength)
        merged['change'] = (merged['close'] - merged['prev_close'])
        # User defined criteria: Variation > 0.7 * ATR and Positive
        merged['is_above_atr'] = merged['change'] > (0.7 * merged['atr14'])
        merged['is_below_atr'] = merged['change'] < (-0.7 * merged['atr14'])
        merged['is_volatile'] = merged['change'].abs() > (0.7 * merged['atr14'])
        merged['signal_strength'] = merged['change'] / merged['atr14'] # Ratio
        
        return merged
        
    finally:
        db.close()

def fetch_sector_from_yahoo(tickers: list) -> dict:
    """
    Fetches sector information from Yahoo Finance for a list of tickers
    and maps them to the app's sector names.
    """
    if not tickers:
        return {}

    # Yahoo Sector -> App Sector Mapping
    yahoo_to_app_map = {
        'Basic Materials': 'Materials',
        'Communication Services': 'Communication Services',
        'Consumer Cyclical': 'Consumer Discretionary',
        'Consumer Defensive': 'Consumer Staples',
        'Energy': 'Energy',
        'Financial Services': 'Financials',
        'Healthcare': 'Health Care',
        'Industrials': 'Industrials',
        'Real Estate': 'Real Estate',
        'Technology': 'Technology',
        'Utilities': 'Utilities'
    }

    found_sectors = {}
    
    print(f"Fetching Yahoo info for: {tickers}")
    
    for t in tickers:
        try:
            # yfinance.Ticker info property fetches data (can be slow for many tickers)
            # Maybe optimize with Ticker(str) but we need individual info.
            info = yf.Ticker(t).info
            y_sector = info.get('sector')
            
            if y_sector:
                # Try to map, fallback to original if not in map (or maybe ignore?)
                # If mapped key exists
                if y_sector in yahoo_to_app_map:
                    found_sectors[t] = yahoo_to_app_map[y_sector]
                else:
                    # Provide a best guess or just return what Yahoo gave if it matches key logic
                    # Check if y_sector is exactly one of our app keys?
                    # For now, let's just stick to the map to ensure it fits in charts.
                    print(f"  Warning: Yahoo sector '{y_sector}' for {t} not in map.")
        except Exception as e:
            print(f"  Error fetching {t} from Yahoo: {e}")
            
    return found_sectors

def get_industries_for_tickers(tickers: list) -> dict:
    """
    Identifies the industry for a list of tickers.
    Returns: dict {ticker: industry_name}
    """
    db = next(get_db())
    try:
        # Clean inputs
        clean_tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        if not clean_tickers:
            return {}
            
        results = db.query(Constituent.ticker, Constituent.industry)\
            .filter(Constituent.ticker.in_(clean_tickers))\
            .all()
            
        return {r[0]: r[1] for r in results}
    finally:
        db.close()

def get_industry_counts() -> dict:
    """
    Returns the total number of constituents per industry in the database.
    Returns: dict {industry_name: count}
    """
    db = next(get_db())
    try:
        results = db.query(Constituent.industry, func.count(Constituent.id))\
            .join(Sector)\
            .filter(Sector.type == 'cap')\
            .group_by(Constituent.industry)\
            .all()
            
        return {r[0]: r[1] for r in results if r[0]} # Ensure industry is not None
    finally:
        db.close()

def fetch_industry_from_yahoo(tickers: list) -> dict:
    """
    Fetches industry information from Yahoo Finance for a list of tickers.
    """
    if not tickers:
        return {}

    found_industries = {}
    
    print(f"Fetching Yahoo industry info for: {tickers}")
    
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            y_industry = info.get('industry')
            
            if y_industry:
                found_industries[t] = y_industry
        except Exception as e:
            print(f"  Error fetching {t} from Yahoo: {e}")
            
    return found_industries

def get_available_dates():
    """Returns sorted list of dates available in ConstituentPrice."""
    db = next(get_db())
    try:
        results = db.query(ConstituentPrice.date).distinct().order_by(ConstituentPrice.date.desc()).all()
        return [r[0] for r in results]
    finally:
        db.close()

def get_top_performers(target_date, top_n=20):
    """
    Returns top gainers and losers for each sector on target_date.
    Calculates return as (Close_Target / Close_Prev) - 1.
    """
    db = next(get_db())
    try:
        # 1. Find the previous available date before target_date
        prev_date_row = db.query(ConstituentPrice.date)\
            .filter(ConstituentPrice.date < target_date)\
            .order_by(ConstituentPrice.date.desc())\
            .first()
        
        if not prev_date_row:
            return None, None
            
        prev_date = prev_date_row[0]
        
        # 2. Fetch prices for both dates
        # Subquery for prices at target date
        p_target = db.query(ConstituentPrice.constituent_id, ConstituentPrice.close.label('close_t'))\
            .filter(ConstituentPrice.date == target_date).subquery()
            
        # Subquery for prices at previous date
        p_prev = db.query(ConstituentPrice.constituent_id, ConstituentPrice.close.label('close_p'))\
            .filter(ConstituentPrice.date == prev_date).subquery()
            
        # Main query
        results = db.query(
            Sector.name.label('sector'),
            Constituent.ticker,
            p_target.c.close_t,
            p_prev.c.close_p
        ).join(Constituent, Sector.id == Constituent.sector_id)\
         .join(p_target, Constituent.id == p_target.c.constituent_id)\
         .join(p_prev, Constituent.id == p_prev.c.constituent_id)\
         .filter(Sector.type == 'cap')\
         .all()
         
        if not results:
            return {}, {}
            
        df = pd.DataFrame(results, columns=['Sector', 'Ticker', 'Close_T', 'Close_P'])
        df['Return'] = (df['Close_T'] / df['Close_P'] - 1) * 100
        
        # Initialize maps with all sectors from config to ensure they appear as columns
        # Use abbreviations for column headers
        gainers_map = {SECTOR_ABBR.get(s, s): [""] * top_n for s in SECTORS_CONFIG.keys()}
        losers_map = {SECTOR_ABBR.get(s, s): [""] * top_n for s in SECTORS_CONFIG.keys()}
        
        for sector_name in df['Sector'].unique():
            abbr = SECTOR_ABBR.get(sector_name, sector_name)
            if abbr not in gainers_map: continue 
            
            group = df[df['Sector'] == sector_name]
            sorted_group = group.sort_values(by='Return', ascending=False)
            
            # Gainers
            top_gainers = sorted_group.head(top_n)
            g_list = [f"{r.Ticker} ({r.Return:.2f}%)" for _, r in top_gainers.iterrows()]
            # Pad or truncate to top_n
            gainers_map[abbr] = (g_list + [""] * top_n)[:top_n]
            
            # Losers
            top_losers = group.sort_values(by='Return', ascending=True).head(top_n)
            l_list = [f"{r.Ticker} ({r.Return:.2f}%)" for _, r in top_losers.iterrows()]
            losers_map[abbr] = (l_list + [""] * top_n)[:top_n]
            
        return gainers_map, losers_map
        
    finally:
        db.close()

def get_market_top_gainers(target_date, top_n=50):
    """Returns top N gainers across all sectors for target_date."""
    db = next(get_db())
    try:
        prev_date_row = db.query(ConstituentPrice.date)\
            .filter(ConstituentPrice.date < target_date)\
            .order_by(ConstituentPrice.date.desc())\
            .first()
        
        if not prev_date_row:
            return pd.DataFrame()
            
        prev_date = prev_date_row[0]
        
        # 1. Fetch target date results with constituent_id for merging
        results = db.query(
            Constituent.id.label('constituent_id'),
            Constituent.ticker,
            Sector.name.label('sector'),
            Constituent.industry,
            ConstituentPrice.close.label('close_t')
        ).join(Constituent, Constituent.id == ConstituentPrice.constituent_id)\
         .join(Sector, Sector.id == Constituent.sector_id)\
         .filter(ConstituentPrice.date == target_date)\
         .filter(Sector.type == 'cap')\
         .all()
         
        if not results:
            return pd.DataFrame()
            
        df_target = pd.DataFrame(results, columns=['constituent_id', 'Ticker', 'Sector', 'Industry', 'Close_T'])
        
        # 2. Fetch previous date prices
        prev_results = db.query(
            ConstituentPrice.constituent_id,
            ConstituentPrice.close.label('close_p')
        ).filter(ConstituentPrice.date == prev_date).all()
        
        df_prev = pd.DataFrame(prev_results, columns=['constituent_id', 'close_p'])
        
        # 3. Merge and Calculate
        df = pd.merge(df_target, df_prev, on='constituent_id', how='inner')
        df['Return'] = (df['Close_T'] / df['close_p'] - 1) * 100
        
        return df.sort_values(by='Return', ascending=False).head(top_n)
        
    finally:
        db.close()

def analyze_themes_with_ai(target_date, top_gainers_df):
    """Uses Gemini to identify themes among top gainers."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Erro: GEMINI_API_KEY não encontrada no arquivo .env."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Prepare data for prompt
    stocks_list = []
    for _, row in top_gainers_df.iterrows():
        stocks_list.append(f"- {row.Ticker} ({row.Sector} / {row.Industry}): +{row.Return:.2f}%")
        
    stocks_str = "\n".join(stocks_list)
    
    prompt = f"""
    Act as an expert US equity market analyst. 
    Analyze the following list of the top 50 gaining stocks in the US market for {target_date}.
    
    Stocks List:
    {stocks_str}
    
    Task:
    1. Identify any common investment themes, sector catalysts, or macroeconomic narratives (e.g., earnings beats, interest rate expectations, commodity price movements, breakthrough technology news).
    2. Be neutral and objective. If no clear theme exists, state that the moves appear idiosyncratic.
    3. For each identified theme, cite at least one specific news event or reputable source (e.g., Bloomberg, Reuters, WSJ, CNBC, or official company press releases) from that period that corroborates the price action.
    4. Conclude with a brief summary of the market sentiment (e.g., narrow leadership, broad-based rally, or specific sector strength).
    
    Respond in ENGLISH in a professional and concise format using Markdown.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro ao gerar análise: {str(e)}"

def get_atr_performers(target_date, top_n=20):
    """
    Returns top gainers and losers based on (Close_T - Close_P) / ATR14.
    """
    db = next(get_db())
    try:
        # 1. Find the previous available date before target_date
        prev_date_row = db.query(ConstituentPrice.date)\
            .filter(ConstituentPrice.date < target_date)\
            .order_by(ConstituentPrice.date.desc())\
            .first()
        
        if not prev_date_row:
            return None, None
            
        prev_date = prev_date_row[0]
        
        # 2. Fetch prices and ATR for target date
        p_target = db.query(
            ConstituentPrice.constituent_id, 
            ConstituentPrice.close.label('close_t'),
            ConstituentPrice.atr14
        ).filter(ConstituentPrice.date == target_date).subquery()
            
        # 3. Fetch prices for previous date
        p_prev = db.query(
            ConstituentPrice.constituent_id, 
            ConstituentPrice.close.label('close_p')
        ).filter(ConstituentPrice.date == prev_date).subquery()
            
        # 4. Main query joining with Sector/Constituent
        results = db.query(
            Sector.name.label('sector'),
            Constituent.ticker,
            p_target.c.close_t,
            p_target.c.atr14,
            p_prev.c.close_p
        ).join(Constituent, Sector.id == Constituent.sector_id)\
         .join(p_target, Constituent.id == p_target.c.constituent_id)\
         .join(p_prev, Constituent.id == p_prev.c.constituent_id)\
         .filter(Sector.type == 'cap')\
         .all()
         
        if not results:
            return {}, {}
            
        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['Sector', 'Ticker', 'Close_T', 'ATR14', 'Close_P'])
        # Avoid division by zero or NaN ATR
        df = df[df['ATR14'] > 0]
        df['Signal'] = (df['Close_T'] - df['Close_P']) / df['ATR14']
        
        # Use abbreviations for column headers
        gainers_map = {SECTOR_ABBR.get(s, s): [""] * top_n for s in SECTORS_CONFIG.keys()}
        losers_map = {SECTOR_ABBR.get(s, s): [""] * top_n for s in SECTORS_CONFIG.keys()}
        
        for sector_name in df['Sector'].unique():
            abbr = SECTOR_ABBR.get(sector_name, sector_name)
            if abbr not in gainers_map: continue 
            
            group = df[df['Sector'] == sector_name]
            
            # Gainers (Highest Signal)
            top_gainers = group.sort_values(by='Signal', ascending=False).head(top_n)
            g_list = [f"{r.Ticker} ({r.Signal:.2f}x)" for _, r in top_gainers.iterrows()]
            gainers_map[abbr] = (g_list + [""] * top_n)[:top_n]
            
            # Losers (Lowest Signal)
            top_losers = group.sort_values(by='Signal', ascending=True).head(top_n)
            l_list = [f"{r.Ticker} ({r.Signal:.2f}x)" for _, r in top_losers.iterrows()]
            losers_map[abbr] = (l_list + [""] * top_n)[:top_n]
            
        return gainers_map, losers_map
        
    finally:
        db.close()

def get_market_top_atr_performers(target_date, top_n=50):
    """Returns top N performers by (Variation / ATR) across all sectors."""
    db = next(get_db())
    try:
        prev_date_row = db.query(ConstituentPrice.date)\
            .filter(ConstituentPrice.date < target_date)\
            .order_by(ConstituentPrice.date.desc())\
            .first()
        
        if not prev_date_row:
            return pd.DataFrame()
            
        prev_date = prev_date_row[0]
        
        # 1. Fetch target date results
        results = db.query(
            Constituent.id.label('constituent_id'),
            Constituent.ticker,
            Sector.name.label('sector'),
            Constituent.industry,
            ConstituentPrice.close.label('close_t'),
            ConstituentPrice.atr14
        ).join(Constituent, Constituent.id == ConstituentPrice.constituent_id)\
         .join(Sector, Sector.id == Constituent.sector_id)\
         .filter(ConstituentPrice.date == target_date)\
         .filter(Sector.type == 'cap')\
         .all()
         
        if not results:
            return pd.DataFrame()
            
        df_target = pd.DataFrame(results, columns=['constituent_id', 'Ticker', 'Sector', 'Industry', 'Close_T', 'ATR14'])
        
        # 2. Fetch previous date prices
        prev_results = db.query(
            ConstituentPrice.constituent_id,
            ConstituentPrice.close.label('close_p')
        ).filter(ConstituentPrice.date == prev_date).all()
        
        df_prev = pd.DataFrame(prev_results, columns=['constituent_id', 'close_p'])
        
        # 3. Merge and Calculate
        df = pd.merge(df_target, df_prev, on='constituent_id', how='inner')
        df = df[df['ATR14'] > 0]
        df['Signal'] = (df['Close_T'] - df['close_p']) / df['ATR14']
        df['Return'] = (df['Close_T'] / df['close_p'] - 1) * 100
        
        return df.sort_values(by='Signal', ascending=False).head(top_n)
        
    finally:
        db.close()

def analyze_atr_themes_with_ai(target_date, top_atr_df):
    """Uses Gemini to identify themes among top ATR performers."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Erro: GEMINI_API_KEY não encontrada."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Prepare data for prompt
    stocks_list = []
    for _, row in top_atr_df.iterrows():
        stocks_list.append(f"- {row.Ticker} ({row.Sector} / {row.Industry}): {row.Signal:.2f}x ATR (+{row.Return:.2f}%)")
        
    stocks_str = "\n".join(stocks_list)
    
    prompt = f"""
    Act as an expert US equity analyst specializing in Momentum and Market Volatility.
    The following list shows the 50 stocks with the most significant price moves relative to their 14-day ATR (Average True Range) on {target_date}.
    
    This metric (Variation/ATR) highlights "Price Anomalies" - stocks moving significantly beyond their normal daily range.
    
    Anomalous Moves:
    {stocks_str}
    
    Task:
    1. Objectively analyze whether these anomalies are driven by a common theme, a sector-wide catalyst, or macroeconomic events (e.g., earnings reports, clinical trial data, regulatory news, or rotation into specific market caps).
    2. For each theme identified, cite at least one specific news source or event (Bloomberg, CNBC, Reuters, WSJ) that explains the price action for the affected stocks.
    3. Determine if the overall market sentiment reflected in these moves suggests an aggressive "Risk-on" environment or isolated occurrences.
    
    Respond in ENGLISH in a technical and clear format using Markdown.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro na análise: {str(e)}"
