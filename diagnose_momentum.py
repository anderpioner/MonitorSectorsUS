import data_service as ds
import pandas as pd

# Fetch data directly to inspect prices
# Get standard Cap Weighted data to find RSPG (Wait, RSPG is Equal Weight Energy? Let's check config)
# config: 'Energy': {'cap': 'XLE', 'equal': 'RSPG'}
# So if user is looking at Equal Weight, they see RSPG. If Cap, they see XLE.
# User said "RSPG". So they must be in Equal Weight mode? Or just looking at the ticker?
# Let's assume Equal Weight mode for RSPG.

print("Fetching data for RSPG (Equal Weight Energy)...")
df = ds.get_sector_data_from_db(period_days=200, weight_type='equal')

if 'RSPG' not in df.columns:
    print("RSPG not found in Equal Weight data. Checking Cap Weight just in case...")
    df_cap = ds.get_sector_data_from_db(period_days=200, weight_type='cap')
    if 'RSPG' in df_cap.columns:
        print("Found RSPG in Cap Weight data (Unlikely).")
        df = df_cap
    else:
        print("RSPG not found. Listing columns:")
        print(df.columns)
        exit()

series = df['RSPG'].dropna()
print(f"\n--- RSPG Price History (Last 10) ---")
print(series.tail(10))

def calc_score_debug(series, idx_loc, label):
    try:
        base = idx_loc
        
        # P indexes
        i_1 = base - 1
        i_5 = base - 5
        i_10 = base - 10
        i_20 = base - 20
        i_40 = base - 40
        
        p_base = series.iloc[base] # Price at 'As Of' date
        p_1 = series.iloc[i_1]
        p_5 = series.iloc[i_5]
        p_10 = series.iloc[i_10]
        p_20 = series.iloc[i_20]
        p_40 = series.iloc[i_40]
        
        # Dates
        d_base = series.index[base].date()
        d_1 = series.index[i_1].date()
        d_5 = series.index[i_5].date()
        d_10 = series.index[i_10].date()
        d_20 = series.index[i_20].date()
        d_40 = series.index[i_40].date()
        
        print(f"\n--- Calculation {label} (As of {d_base}) ---")
        print(f"P_Base ({d_base}): {p_base:.2f}")
        print(f"P_1    ({d_1}): {p_1:.2f}")
        print(f"P_5    ({d_5}): {p_5:.2f}")
        print(f"P_10   ({d_10}): {p_10:.2f}")
        print(f"P_20   ({d_20}): {p_20:.2f}")
        print(f"P_40   ({d_40}): {p_40:.2f}")

        r_5_1 = (p_1 / p_5) - 1
        r_10_5 = (p_5 / p_10) - 1
        r_20_10 = (p_10 / p_20) - 1
        r_40_20 = (p_20 / p_40) - 1
        
        print(f"R(5-1)   [T-5 to T-1]: {r_5_1*100:.2f}%")
        print(f"R(10-5)  [T-10 to T-5]: {r_10_5*100:.2f}%")
        print(f"R(20-10) [T-20 to T-10]: {r_20_10*100:.2f}%")
        print(f"R(40-20) [T-40 to T-20]: {r_40_20*100:.2f}%")
        
        score = (0.3 * r_5_1) + (0.3 * r_10_5) + (0.2 * r_20_10) + (0.2 * r_40_20)
        print(f"Score: {score*100:.4f}")
        
    except IndexError as e:
        print(f"Error in debug: {e}")

# Calculate Today
calc_score_debug(series, -1, "TODAY")

# Calculate 5 Days Ago
calc_score_debug(series, -6, "5 DAYS AGO")
