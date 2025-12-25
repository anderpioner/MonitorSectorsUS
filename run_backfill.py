
from data_service import update_constituents_data

if __name__ == "__main__":
    print("Starting backfill for Health Care...")
    # This will use the default '10y' period we recently set
    update_constituents_data(sector_name="Health Care")
    print("Backfill complete.")
