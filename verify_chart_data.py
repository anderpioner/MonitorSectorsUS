import data_service as ds
import pandas as pd

print("Testing Momentum History Fetch...")
try:
    # Test for XLE (Energy) which we know has data
    df = ds.get_momentum_history('XLE', period_days=100)
    
    if df.empty:
        print("FAIL: DataFrame is empty for XLE.")
    else:
        print("SUCCESS: Data fetched.")
        print("--- Head ---")
        print(df.head())
        print(f"Rows: {len(df)}")
        
        if 'Score' not in df.columns:
             print("FAIL: 'Score' column missing.")
        else:
             print("SUCCESS: 'Score' column present.")

except Exception as e:
    print(f"ERROR: {e}")
