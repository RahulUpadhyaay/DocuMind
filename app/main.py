import os
from starlette.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import app_logger
from app.db.session import async_engine
from app.db.base import Base
import app.db.models  # Register models with Base
from app.api.v1.tenants import router as tenants_router
from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info(f"Starting {settings.PROJECT_NAME} in [{settings.ENVIRONMENT}] mode...")
    
    # Auto-initialize PostgreSQL extensions and tables if not present
    try:
        async with async_engine.begin() as conn:
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
            await conn.run_sync(Base.metadata.create_all)
            app_logger.info("Database extensions and schemas verified successfully.")
    except Exception as e:
        app_logger.warning(f"Database auto-migration notice: {e}")

    yield

    app_logger.info("Shutting down application...")
    await async_engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Production-Ready Multi-Tenant Knowledge Base Microservice with Hybrid Search, Strict Citations, and SSE Streaming.",
    lifespan=lifespan,
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(tenants_router, prefix=settings.API_V1_STR)
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(query_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health"])
async def health_check():
    """Service health check endpoint for Render/Railway/AWS orchestration."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "reranker_provider": settings.RERANKER_PROVIDER,
    }


FRONTEND_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "index.html")


@app.get("/", tags=["Dashboard"])
async def dashboard():
    """Serves the interactive Enterprise Document Intelligence & Citation Playground."""
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "documentation": "/docs",
        "api_v1": settings.API_V1_STR,
    }
