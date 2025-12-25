
def calculate_sector_up_metric(sector_id, db_session, start_date=None, lookback_window=84, threshold=0.25):
    """
    Calculates the 'Stocks > 25% in 84d' metric and stores it in BreadthMetric table.
    Metric Name: 'pct_up_25_84d'
    """
    try:
        # Get Constituents
        constituents = db_session.query(Constituent.id).filter(Constituent.sector_id == sector_id).all()
        if not constituents:
            return
            
        c_ids = [c.id for c in constituents]
        
        # Determine fetch start date
        # We need lookback_window prior history
        if start_date:
            fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_window + 20)
        else:
            fetch_start = date.today() - pd.Timedelta(days=3650 + lookback_window + 20) # 10 years + buffer
            
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
        
        # Calculate Daily Stats
        daily_counts = (df_pct >= threshold).sum(axis=1)
        valid_counts = df_pct.notna().sum(axis=1)
        
        # Filter dates to process (if start_date provided)
        if start_date:
             # Ensure we cover the requested start_date
             process_mask = df_pct.index >= pd.to_datetime(start_date)
             daily_counts = daily_counts[process_mask]
             valid_counts = valid_counts[process_mask]
        
        # Prepare Upsert
        metrics_to_upsert = []
        
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        # Note: If reusing mysql/postgres logic, adjust import. Using standard list for now.
        
        for dt, count in daily_counts.items():
            total = valid_counts.loc[dt]
            if total > 0:
                val = (count / total) * 100
                metrics_to_upsert.append({
                    'sector_id': sector_id,
                    'date': dt.date(),
                    'metric': 'pct_up_25_84d',
                    'value': float(val)
                })
                
        # Bulk Upsert
        if metrics_to_upsert:
            # We use core sqlalchemy insert with ON CONFLICT DO UPDATE
            stmt = insert(BreadthMetric).values(metrics_to_upsert)
            stmt = stmt.on_conflict_do_update(
                index_elements=['sector_id', 'date', 'metric'],
                set_={'value': stmt.excluded.value}
            )
            db_session.execute(stmt)
            db_session.commit()
            
        print(f"  > Calculate 'pct_up_25_84d': Updated {len(metrics_to_upsert)} records.")
        
    except Exception as e:
        print(f"Error calculating up metric: {e}")
        db_session.rollback()
