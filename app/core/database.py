from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logger import logger

# Create SQLAlchemy engine
try:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )
    logger.info("Database engine created successfully.")
except Exception as e:
    logger.error(f"Error creating database engine: {e}")
    raise e

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Automap base to reflect existing database schema
Base = automap_base()

def init_db():
    try:
        # Reflect the tables from the database
        Base.prepare(autoload_with=engine)
        logger.info(f"Database schema reflected successfully. Available tables: {list(Base.classes.keys())}")
    except Exception as e:
        logger.error(f"Error reflecting database schema: {e}")
        # Not raising here to avoid crashing startup if DB is temporarily unavailable
        # We can try to reflect later or handle it per request.

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
