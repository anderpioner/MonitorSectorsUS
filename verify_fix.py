import data_service as ds
import pandas as pd

print("Testing Momentum Ranking...")
try:
    df_mom = ds.get_momentum_ranking(weight_type='cap')
    if df_mom.empty:
        print("FAIL: Momentum DataFrame is empty.")
    else:
        if 'Last Price' in df_mom.columns and 'Date' in df_mom.columns:
            print("SUCCESS: Momentum DataFrame has 'Last Price' and 'Date'.")
            print(df_mom[['Sector', 'Last Price', 'Date', 'Score']].head())
        else:
            print(f"FAIL: Missing columns. Found: {df_mom.columns}")
except Exception as e:
    print(f"ERROR Momentum: {e}")

print("\nTesting Performance Matrix...")
try:
    df_perf = ds.get_sector_performance_matrix(weight_type='cap')
    if df_perf.empty:
        print("FAIL: Matrix DataFrame is empty.")
    else:
        if 'Last Price' in df_perf.columns and 'Date' in df_perf.columns:
            print("SUCCESS: Matrix DataFrame has 'Last Price' and 'Date'.")
            print(df_perf[['Last Price', 'Date', '5d', '252d']].head())
        else:
            print(f"FAIL: Missing columns. Found: {df_perf.columns}")
except Exception as e:
    print(f"ERROR Matrix: {e}")
