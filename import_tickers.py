import pandas as pd
import os
import data_service as ds

# Mapping of CSV filenames to Sector Names in DB
file_mapping = {
    "BasicMaterials o200k.csv": "Materials",
    "ComunicationService o200k.csv": "Communication Services",
    "ConsumerCyclical o200k.csv": "Consumer Discretionary",
    "ConsumerDefensive o200k.csv": "Consumer Staples",
    "Energy o 200k.csv": "Energy",
    "Financial o200k.csv": "Financials",
    "Helthcare o200k.csv": "Health Care",
    "Industrials o200k.csv": "Industrials",
    "Realstate o200k.csv": "Real Estate",
    "Technology o200k.csv": "Technology",
    "Utilities o200k.csv": "Utilities"
}

def run_import():
    total_tickers = 0
    
    print("Starting ticker import process...")
    
    for filename, sector_name in file_mapping.items():
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found. Skipping {sector_name}.")
            continue
            
        try:
            print(f"Reading {filename} for {sector_name}...")
            df = pd.read_csv(filename)
            
            if "Ticker" not in df.columns:
                print(f"Error: 'Ticker' column not found in {filename}.")
                continue
                
            tickers = df["Ticker"].dropna().unique().tolist()
            # Clean tickers (remove whitespace)
            tickers = [t.strip() for t in tickers if isinstance(t, str)]
            
            print(f"Found {len(tickers)} tickers for {sector_name}. Importing...")
            ds.import_constituents(sector_name, tickers)
            
            total_tickers += len(tickers)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nImport finished. Total tickers found: {total_tickers}")
    
    print("\nStarting Data Update (This may take a while)...")
    try:
        ds.update_constituents_data()
        print("\nData Update Complete!")
    except Exception as e:
        print(f"Error during data update: {e}")

if __name__ == "__main__":
    run_import()
