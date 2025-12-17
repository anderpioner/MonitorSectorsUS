from data_service import update_constituents_data
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

def test_healthcare_update():
    print("Attempting to update Health Care sector...")
    try:
        update_constituents_data(sector_name='Health Care')
        print("Success!")
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_healthcare_update()
