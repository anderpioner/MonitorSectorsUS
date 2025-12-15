import data_service as ds
import pandas as pd

# Baseline from Step 75
baseline = {
    'XLK': 1.825231,
    'XLE': 1.795311,
    'XLY': 1.582509,
    'XLV': 1.466103,
    'XLC': 1.226639
}

print("Running Momentum Ranking Calculation...")
try:
    df = ds.get_momentum_ranking(weight_type='cap')
    print("\n--- Current Top 5 ---")
    print(df[['Score']].head())
    
    print("\n--- Comparison with Baseline ---")
    for ticker, base_score in baseline.items():
        if ticker in df.index:
            curr_score = df.loc[ticker, 'Score']
            diff = curr_score - base_score
            print(f"{ticker}: Base={base_score:.6f}, Curr={curr_score:.6f}, Diff={diff:.6f}")
        else:
            print(f"{ticker}: Not found in current results!")

except Exception as e:
    print(f"Error: {e}")
