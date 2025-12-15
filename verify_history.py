import data_service as ds
import pandas as pd

print("Testing Momentum Ranking History...")
try:
    df_mom = ds.get_momentum_ranking(weight_type='cap')
    if df_mom.empty:
        print("FAIL: Momentum DataFrame is empty.")
    else:
        # Check Columns
        req_cols = ['Score', 'Score -5d', 'Score -20d', 'Score -50d']
        missing = [c for c in req_cols if c not in df_mom.columns]
        
        if missing:
            print(f"FAIL: Missing columns: {missing}")
        else:
            print("SUCCESS: Historical columns present.")
            print("\n--- Top 5 Tickers with History ---")
            print(df_mom[req_cols].head())
            
            # Sanity Check
            # Ensure not all None
            nulls = df_mom['Score -50d'].isna().sum()
            print(f"\nNulls in Score -50d: {nulls} / {len(df_mom)}")

except Exception as e:
    print(f"ERROR: {e}")
