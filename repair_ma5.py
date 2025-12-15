import data_service as ds
import time

def repair_ma5():
    # Sectors identified with 100% missing MA5
    sectors_to_repair = [
        "Industrials", 
        "Materials", 
        "Real Estate", 
        "Utilities",
        "Technology" 
    ]
    
    print(f"Starting MA5 Repair for: {sectors_to_repair}")
    
    for sector in sectors_to_repair:
        print(f"\n--- Repairing {sector} ---")
        try:
            # this will fetch data (10y), calc MA5, and upsert
            ds.update_constituents_data(sector_name=sector)
            print(f"Successfully repaired {sector}")
        except Exception as e:
            print(f"Failed to repair {sector}: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    repair_ma5()
