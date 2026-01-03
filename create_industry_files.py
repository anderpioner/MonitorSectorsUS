import pandas as pd
import os

target_dir = r"c:\D\Python\MonitorSectors\tickers\Utilities Sector"
industries = [
    "Utilities - Independent Power Producers",
    "Utilities - Diversified",
    "Utilities - Regulated Gas",
    "Utilities - Regulated Water",
    "Utilities - Regulated Electric"
]

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f"Creating Excel files in: {target_dir}")

for industry in industries:
    filename = f"{industry}.xlsx"
    filepath = os.path.join(target_dir, filename)
    
    # Create an empty DataFrame
    df = pd.DataFrame(columns=["Ticker"])
    
    try:
        df.to_excel(filepath, index=False)
        print(f"Created: {filename}")
    except Exception as e:
        print(f"Error creating {filename}: {e}")

print("Done.")
