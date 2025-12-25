
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
        if metrics_to_upsert:
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            stmt = insert(BreadthMetric).values(metrics_to_upsert)
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
