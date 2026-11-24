from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import Base

# The pgvector Vector column type converts lists to pgvector's text format itself, so no
# extra asyncpg codec is registered (doing both makes asyncpg reject the converted value).
engine = create_async_engine(get_settings().database_url)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Enable pgvector and create tables and indexes if missing. Safe to run repeatedly.

    Fine while the schema is young; switch to Alembic migrations once there is data
    you don't want to rebuild.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
