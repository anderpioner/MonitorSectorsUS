import data_service as ds
import os
import pandas as pd
from database import get_db
from models import Sector, PriceData, ConstituentPrice

DB_PATH = "./sectors_v6.db"

def test_database_flow():
    print("1. Removing existing DB if any...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("   Deleted old DB.")
    
    print("\n2. Initializing Sector DB...")
    ds.initialize_sectors_in_db()
    
    db = next(get_db())
    sectors = db.query(Sector).all()
    print(f"   Sectors found: {len(sectors)}")
    # Should be 22 now (11 cap + 11 equal) if all mapped
    assert len(sectors) >= 22
    
    print("\n3. Updating Sector Data (Fetching from yfinance - small period)...")
    # Using '1mo' to be fast
    ds.update_sector_data(period="1mo")
    
    price_count = db.query(PriceData).count()
    print(f"   Price records found: {price_count}")
    assert price_count > 0
    
    print("\n4. Reading Data from DB (Equal Weight)...")
    df = ds.get_sector_data_from_db(period_days=30, weight_type='equal')
    print("   Data Frame Head (Equal):")
    print(df.head())
    assert not df.empty
    assert 'RSPT' in df.columns or 'RSPG' in df.columns # Check for some equal weight ticker
    
    print("\n5. Testing Performance Matrix...")
    # We only fetched 1mo of data in step 3, so matrix for long periods (252) will be NaN or empty if checked strictly.
    # Let's fetch more data first to test properly.
    print("   Fetching more data (1y)...")
    ds.update_sector_data(period="1y")
    
    matrix = ds.get_sector_performance_matrix(weight_type='cap', periods=[5, 10, 20])
    print("   Matrix Head:")
    print(matrix.head())
    assert not matrix.empty
    
    print("\n6. Testing Momentum Ranking...")
    mom_df = ds.get_momentum_ranking(weight_type='cap')
    print("   Momentum Head:")
    print(mom_df.head())
    assert not mom_df.empty
    assert 'Score' in mom_df.columns
    assert 'R(5-1)' in mom_df.columns
    
    print("\nSUCCESS: Database flow verified!")
    
    print("\n7. Testing Constituent Import & Breadth...")
    # Add dummy constituents for Technology
    dummy_tickers = ['AAPL', 'MSFT', 'NVDA'] # Only 3 for speed
    print(f"   Importing {dummy_tickers} to Technology...")
    ds.import_constituents("Technology", dummy_tickers)
    
    print("   Updating Constituents Data & Breadth...")
    ds.update_constituents_data(sector_name="Technology")
    
    print("   Checking ConstituentPrice table...")
    db = next(get_db())
    c_prices = db.query(ConstituentPrice).limit(5).all()
    print(f"   Sample ConstituentPrice rows: {len(c_prices)}")
    assert len(c_prices) > 0
    print(f"   Sample Row: Above MA50={c_prices[0].above_ma50}, Price={c_prices[0].close}")
    
    print("   Fetching Breadth Data...")
    df_breadth = ds.get_breadth_data("Technology", metric='pct_above_ma50', days=30)
    print("   Breadth Head:")
    print(df_breadth.head())
    assert not df_breadth.empty
    
    # Clean up dummy data if needed, but test DB is deleted anyway on run

if __name__ == "__main__":
    test_database_flow()
