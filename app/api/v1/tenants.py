from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
import uuid

from app.db.session import get_db
from app.db.models import Tenant
from app.core.security import generate_api_key, get_current_tenant
from app.core.config import settings

router = APIRouter(prefix="/tenants", tags=["Tenants"])


class CreateTenantRequest(BaseModel):
    name: str


class TenantResponse(BaseModel):
    id: uuid.UUID
    name: str
    api_key: str | None = None
    created_at: str


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    payload: CreateTenantRequest,
    x_master_key: str = Header(..., description="Master Admin API Key"),
    db: AsyncSession = Depends(get_db),
):
    """Admin-only endpoint to onboard new clients/tenants and issue an isolated API Key."""
    if x_master_key != settings.MASTER_API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Master API Key")

    raw_key, key_hash = generate_api_key(prefix=payload.name.lower().replace(" ", "_")[:8])
    tenant = Tenant(
        name=payload.name,
        api_key_hash=key_hash,
        is_active=True,
    )
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        api_key=raw_key,
        created_at=tenant.created_at.isoformat(),
    )


@router.get("/me", response_model=TenantResponse)
async def get_tenant_info(tenant: Tenant = Depends(get_current_tenant)):
    """Returns profile of currently authenticated tenant."""
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        api_key=None,  # Never reveal key hash
        created_at=tenant.created_at.isoformat(),
    )
