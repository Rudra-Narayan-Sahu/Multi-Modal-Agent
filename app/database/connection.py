import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import logfire

from app.database.models import Base

# Database URL configuration
DEFAULT_DB_PATH = Path("processed_data") / "memory.db"
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

raw_db_url = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.resolve()}")

# Normalize postgres:// to postgresql:// if needed
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

is_sqlite = raw_db_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(
    raw_db_url,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)
    logfire.info(f"Database initialized: {raw_db_url.split('@')[-1] if '@' in raw_db_url else raw_db_url}")


@contextmanager
def get_db_session() -> Session:
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    """FastAPI dependency for yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

