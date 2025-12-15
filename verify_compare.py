import data_service as ds

print("Verifying Comparison Chart Helpers...")

# 1. Test get_all_sector_options
opts = ds.get_all_sector_options()
print(f"Found {len(opts)} sector options.")
if len(opts) > 0:
    print(f"First Option: {opts[0]}")
    # Verify both CW and EW are present
    has_cap = any("(Cap)" in o['name'] for o in opts)
    has_eq = any("(Eq)" in o['name'] for o in opts)
    if has_cap and has_eq:
        print("SUCCESS: Both Cap and Eq options found.")
    else:
        print("FAIL: Missing Cap or Eq options.")

# 2. Test get_price_history
ticker = opts[0]['ticker']
print(f"Fetching price history for {ticker}...")
p_hist = ds.get_price_history(ticker, period_days=100)
if not p_hist.empty:
    print(f"SUCCESS: Fetched {len(p_hist)} price points.")
    print("Head:", p_hist.head(2))
else:
    print("FAIL: Price history empty.")
