import hashlib
import secrets
from fastapi import Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.db.session import get_db
from app.db.models import Tenant

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key(prefix: str = "rag") -> tuple[str, str]:
    """Generates a raw API key and its secure SHA-256 hash."""
    random_token = secrets.token_urlsafe(32)
    raw_key = f"{prefix}_{random_token}"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return raw_key, key_hash


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def get_current_tenant(
    api_key: str = Security(API_KEY_HEADER),
    db: AsyncSession = Depends(get_db),
) -> Tenant:
    """Authenticates tenant via X-API-Key header. Ensures strict multi-tenant isolation."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header in request",
        )

    # Check for Master Admin API Key
    if api_key == settings.MASTER_API_KEY:
        # Check if default admin tenant exists or create on the fly
        stmt = select(Tenant).where(Tenant.name == "Default Master Tenant")
        result = await db.execute(stmt)
        tenant = result.scalar_one_or_none()
        if not tenant:
            tenant = Tenant(
                name="Default Master Tenant",
                api_key_hash=hash_api_key(settings.MASTER_API_KEY),
                is_active=True,
            )
            db.add(tenant)
            await db.commit()
            await db.refresh(tenant)
        return tenant

    key_hash = hash_api_key(api_key)
    stmt = select(Tenant).where(Tenant.api_key_hash == key_hash, Tenant.is_active == True)
    result = await db.execute(stmt)
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API Key",
        )

    return tenant
