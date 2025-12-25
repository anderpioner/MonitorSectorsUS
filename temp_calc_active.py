
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
            stmt = insert(BreadthMetric).values(metrics_to_upsert)
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
