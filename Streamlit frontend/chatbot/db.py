# chatbot/db.py
import os
from sqlalchemy import create_engine

def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set. Put it in .env")
    # pool_pre_ping=True helps keep connections healthy
    engine = create_engine(url, pool_pre_ping=True, future=True)
    return engine
