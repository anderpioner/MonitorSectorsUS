
import data_service as ds
import pandas as pd

def check_active_count():
    sector_name = 'Technology Sector'
    print(f"Checking active_count for {sector_name}...")
    
    df = ds.get_breadth_data(sector_name, metric='active_count', days=365)
    
    if df is not None and not df.empty:
        print("Data found:")
        print(df.tail())
        print(f"Min value: {df['Value'].min()}")
        print(f"Max value: {df['Value'].max()}")
    else:
        print("No 'active_count' data found.")

if __name__ == "__main__":
    check_active_count()
