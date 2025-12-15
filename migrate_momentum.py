from database import engine
from sqlalchemy import text

def add_momentum_column():
    with engine.connect() as conn:
        try:
            # Check if column exists
            result = conn.execute(text("PRAGMA table_info(price_data)"))
            columns = [row[1] for row in result]
            
            if 'momentum_score' not in columns:
                print("Adding momentum_score column to price_data...")
                conn.execute(text("ALTER TABLE price_data ADD COLUMN momentum_score FLOAT"))
                print("Column added successfully.")
            else:
                print("Column momentum_score already exists.")
        except Exception as e:
            print(f"Error migrating: {e}")

if __name__ == "__main__":
    add_momentum_column()
