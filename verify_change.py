import data_service as ds
import pandas as pd

print("Testing Momentum Change...")
try:
    df_mom = ds.get_momentum_ranking(weight_type='cap')
    if df_mom.empty:
        print("FAIL: Momentum DataFrame is empty.")
    else:
        if 'Score Chg (5d)' in df_mom.columns:
            print("SUCCESS: 'Score Chg (5d)' column present.")
            print("\n--- Check Values ---")
            top_5 = df_mom[['Score', 'Score -5d', 'Score Chg (5d)']].head()
            print(top_5)
            
            # Verify calculation manually for first row
            row = top_5.iloc[0]
            s = row['Score']
            s5 = row['Score -5d']
            chg = row['Score Chg (5d)']
            
            # Scores are scaled by 100 in DF, but Chg is Ratio.
            # Data Service Implementation:
            # score_chg_5d = score / score_5d (using RAW values before scaling)
            # DF contains: Score * 100, Score-5d * 100
            # (S*100) / (S5*100) = S/S5 = Chg.
            # So calc should hold on scaled values too.
            
            calc = s / s5 if s5 != 0 else 0
            print(f"\nManual Calc: {s} / {s5} = {calc}")
            print(f"Stored Chg: {chg}")
            
            if abs(calc - chg) < 0.001:
                print("VERIFIED: Calculation is correct.")
            else:
                print("FAIL: Calculation mishmatch.")
        else:
            print("FAIL: Column missing.")
            
except Exception as e:
    print(f"ERROR: {e}")
