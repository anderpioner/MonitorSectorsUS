import os
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import get_db, init_db
from models import Sector, Constituent

# Root directory for tickers
TICKERS_DIR = r"c:\D\Python\MonitorSectors\tickers"

def import_new_structure():
    print("Starting Bulk Import...")
    
    db = next(get_db())
    
    try:
        # 1. Clear existing constituents
        print("Clearing existing constituents...")
        try:
            # We explicitly delete from constituents. 
            db.execute(text("DELETE FROM constituent_prices"))
            db.execute(text("DELETE FROM constituents"))
            db.commit()
            print("  Existing data cleared.")
        except Exception as e:
            print(f"  Error clearing data: {e}")
            db.rollback()
            return

        # 2. Iterate and Import
        total_imported = 0
        
        # Cache sectors to avoid repetitive queries
        sectors = db.query(Sector).all()
        sector_map = {s.name: s.id for s in sectors}
        
        # Walk through the directory
        for root, dirs, files in os.walk(TICKERS_DIR):
            for file in files:
                if not file.endswith(".xlsx"):
                    continue
                
                # Identify Sector from parent folder name
                sector_name = os.path.basename(root)
                
                # Identify Industry from filename
                industry_name = os.path.splitext(file)[0]
                
                if sector_name not in sector_map:
                    print(f"  Warning: Sector '{sector_name}' not found in DB. Skipping {file}.")
                    continue
                
                sector_id = sector_map[sector_name]
                file_path = os.path.join(root, file)
                
                try:
                    # Read Excel
                    # Assuming data is in the first sheet.
                    # User likely kept the header "Ticker" but maybe pasted raw data.
                    # We'll try to sniff.
                    df = pd.read_excel(file_path)
                    
                    if df.empty:
                        print(f"  Skipping empty file: {file}")
                        continue
                    
                    # Normalize columns
                    df.columns = [str(c).strip() for c in df.columns]
                    
                    ticker_col = None
                    if "Ticker" in df.columns:
                        ticker_col = "Ticker"
                    elif len(df.columns) > 0:
                        # Fallback: use first column
                        ticker_col = df.columns[0]
                    
                    if not ticker_col:
                         print(f"  Could not identify ticker column in {file}. Skipping.")
                         continue
                         
                    # Extract tickers
                    tickers = df[ticker_col].dropna().astype(str).tolist()
                    
                    # Clean and Filter
                    clean_tickers = []
                    for t in tickers:
                        t = t.strip().upper()
                        # Ignore "Symbols from TC2000" and blank/short
                        if not t or t == "SYMBOLS FROM TC2000" or len(t) > 10: 
                            continue
                        clean_tickers.append(t)
                        
                    # Insert into DB
                    if not clean_tickers:
                        continue
                        
                    print(f"  Importing {len(clean_tickers)} tickers for {sector_name} / {industry_name}...")
                    
                    new_objects = []
                    for t in clean_tickers:
                        new_objects.append(Constituent(
                            sector_id=sector_id,
                            ticker=t,
                            industry=industry_name
                        ))
                    
                    db.bulk_save_objects(new_objects)
                    db.commit() # Commit per file to save progress
                    total_imported += len(clean_tickers)
                    
                except Exception as ex:
                    print(f"  Error processing {file}: {ex}")
                    db.rollback()

        print(f"\nImport Finished. Total Tickers Imported: {total_imported}")
        
    except Exception as e:
        print(f"Critical Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    import_new_structure()
