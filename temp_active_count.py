
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
