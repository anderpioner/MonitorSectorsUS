from database import get_db, Base, engine
from sqlalchemy import text

def migrate_indicators():
    print("Migrating database for new indicators...")
    db = next(get_db())
    try:
        # List of new columns to add
        new_columns = [
            ("ema8", "FLOAT"),
            ("ema20", "FLOAT"),
            ("ema50", "FLOAT"),
            ("atr14", "FLOAT")
        ]
        
        for col_name, col_type in new_columns:
            try:
                print(f"Adding column {col_name}...")
                db.execute(text(f"ALTER TABLE constituent_prices ADD COLUMN {col_name} {col_type}"))
                print(f"Added {col_name}.")
            except Exception as e:
                # SQLite doesn't support IF NOT EXISTS for ADD COLUMN easily, 
                # so we catch the error if it exists (OperationalError: duplicate column name)
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
    migrate_indicators()
