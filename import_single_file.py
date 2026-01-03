import sys
import os
import pandas as pd
from database import get_db
from models import Sector, Constituent

def import_file(file_path):
    # Derive metadata
    folder = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    sector_name = os.path.basename(folder)
    industry_name = os.path.splitext(filename)[0]
    
    print(f"Target: Sector='{sector_name}', Industry='{industry_name}'")
    print(f"Reading {file_path}...")
    
    db = next(get_db())
    try:
        sector = db.query(Sector).filter(Sector.name == sector_name).first()
        if not sector:
            print(f"Error: Sector '{sector_name}' not found in database.")
            return

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return
            
        df.columns = [str(c).strip() for c in df.columns]
        
        # Find ticker col
        col = None
        if "Ticker" in df.columns: col = "Ticker"
        elif len(df.columns) > 0: col = df.columns[0]
        
        if not col:
            print("Error: No ticker column found.")
            return
            
        tickers = df[col].dropna().astype(str).tolist()
        print(f"Found {len(tickers)} rows.")
        
        count = 0
        skipped = 0
        
        for t in tickers:
            t = t.strip().upper()
            if not t or "TC2000" in t or len(t) > 10: 
                continue
            
            # Check if exists
            exists = db.query(Constituent).filter(Constituent.ticker == t).first()
            if not exists:
                db.add(Constituent(sector_id=sector.id, ticker=t, industry=industry_name))
                count += 1
            else:
                skipped += 1
                
        db.commit()
        print(f"Import Result: {count} imported, {skipped} skipped (already existed).")
        
    except Exception as e:
        print(f"System Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # Hardcoded for this specific request to ensure no path issues, 
    # but structured to be reusable if needed.
    target_file = r"c:\D\Python\MonitorSectors\tickers\Utilities Sector\Utilities - Regulated Water.xlsx"
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    import_file(target_file)
