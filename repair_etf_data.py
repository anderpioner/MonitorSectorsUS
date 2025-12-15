import data_service as ds

print("Updating ETF Data (PriceData) to sync with Constituents...")
# Update last 1 month to be safe and cover the gap
ds.update_sector_data(period="1mo")
print("Update complete.")
