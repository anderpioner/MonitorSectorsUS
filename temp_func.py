
def get_stocks_up_history(sector_name, lookback_window=84, threshold=0.25, days_history=365):
    """
    Calculates the number of stocks in a sector that are up >= threshold over the lookback_window.
    Returns a DataFrame: Index=Date, Columns=['Count'].
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
        start_date = date.today() - timedelta(days=days_history + lookback_window + 10) # Buffer
        
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
        
        # 7. Count Matches
        # Sum boolean (True=1) row-wise
        daily_counts = (df_pct >= threshold).sum(axis=1)
        
        # 8. Trim to requested history
        result_df = daily_counts.to_frame(name='Count')
        # Filter for last days_history
        disp_start = pd.Timestamp(date.today() - timedelta(days=days_history))
        result_df = result_df[result_df.index >= disp_start]
        
        return result_df
        
    finally:
        db.close()
