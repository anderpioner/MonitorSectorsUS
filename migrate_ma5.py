import sqlite3

DB_NAME = "sectors_v6.db"

def migrate():
    print(f"Migrating {DB_NAME}...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # Add ma5 column
        print("Adding column: ma5")
        cursor.execute("ALTER TABLE constituent_prices ADD COLUMN ma5 FLOAT")
        
        # Add above_ma5 column
        print("Adding column: above_ma5")
        cursor.execute("ALTER TABLE constituent_prices ADD COLUMN above_ma5 INTEGER")
        
        conn.commit()
        print("Migration successful columns added.")
        
    except sqlite3.OperationalError as e:
        print(f"Migration error (columns might already exist): {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
