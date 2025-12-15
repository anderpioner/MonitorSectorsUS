import data_service as ds
import time

def resume_updates():
    # List of sectors that were missing metrics
    missing_sectors = [
        "Financials",
        "Health Care",
        "Industrials",
        "Materials",
        "Real Estate",
        "Utilities"
    ]
    
    print(f"Resuming updates for: {missing_sectors}")
    
    for sector in missing_sectors:
        print(f"\n--- Updating {sector} ---")
        try:
            ds.update_constituents_data(sector_name=sector)
            print(f"Successfully updated {sector}")
        except Exception as e:
            print(f"Failed to update {sector}: {e}")
        
        # Pause to be nice to the API
        time.sleep(5)

if __name__ == "__main__":
    resume_updates()
