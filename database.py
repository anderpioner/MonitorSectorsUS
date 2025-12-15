from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.orm import declarative_base
import os

DATABASE_URL = "sqlite:///./sectors_v6.db"

# Increase timeout to wait for locks to clear (30s)
# check_same_thread=False is needed for SQLite with FastAPI/Streamlit multi-threading
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30})

# Enable Write-Ahead Logging (WAL) for better concurrency
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
