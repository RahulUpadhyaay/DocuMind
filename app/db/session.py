from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# Determine if SSL is required for cloud database providers
async_connect_args = {}
sync_connect_args = {}

db_url_lower = settings.DATABASE_URL.lower()
if any(cloud in db_url_lower for cloud in ["render.com", "neon.tech", "supabase", "aws", "sslmode=require"]):
    async_connect_args["ssl"] = "require"

# Async Engine for FastAPI Request Lifecycle
async_engine = create_async_engine(
    settings.async_database_url,
    connect_args=async_connect_args,
    echo=(settings.ENVIRONMENT == "development"),
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,
    pool_timeout=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Synchronous Engine for Celery Worker Tasks
sync_engine = create_engine(
    settings.sync_database_url,
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,
    pool_timeout=10,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db():
    """Dependency for injecting async database session into FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
