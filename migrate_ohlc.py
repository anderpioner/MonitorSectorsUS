from database import get_db
from sqlalchemy import text

def migrate_ohlc():
    print("Migrating database for High/Low columns...")
    db = next(get_db())
    try:
        new_columns = [
            ("high", "FLOAT"),
            ("low", "FLOAT")
        ]
        
        for col_name, col_type in new_columns:
            try:
                print(f"Adding column {col_name}...")
                db.execute(text(f"ALTER TABLE constituent_prices ADD COLUMN {col_name} {col_type}"))
                print(f"Added {col_name}.")
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    print(f"Column {col_name} already exists.")
                else:
                    print(f"Error adding {col_name}: {e}")
        
        db.commit()
        print("Migration complete.")
        
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    migrate_ohlc()
