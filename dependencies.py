import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config import supabase_client

security = HTTPBearer()

ELEVATED_ROLES = {"hr_admin", "it_admin", "super_admin"}

@dataclass
class AuthenticatedUser:
    id: str
    email: Optional[str] = None
    role: str = "employee"
    full_name: Optional[str] = None
    department: Optional[str] = None
    office_location: Optional[str] = None
    employment_type: Optional[str] = None
    raw_user: Any = None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> AuthenticatedUser:
    token = credentials.credentials
    try:
        res = await asyncio.to_thread(supabase_client.auth.get_user, token)
        if not res or not res.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = res.user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _fetch_profile():
        return (
            supabase_client.table("users")
            .select("is_active, role, full_name, department, office_location, employment_type")
            .eq("id", user.id)
            .execute()
        )

    role = "employee"
    full_name = None
    department = None
    office_location = None
    employment_type = None

    try:
        profile_res = await asyncio.to_thread(_fetch_profile)
        if profile_res.data:
            profile = profile_res.data[0]
            if profile.get("is_active") is False:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is inactive. Please contact an administrator.",
                )
            role = profile.get("role") or role
            full_name = profile.get("full_name")
            department = profile.get("department")
            office_location = profile.get("office_location")
            employment_type = profile.get("employment_type")

        meta = user.user_metadata if isinstance(user.user_metadata, dict) else {}
        if not full_name:
            full_name = meta.get("full_name")
        if not department:
            department = meta.get("department")
        if not office_location:
            office_location = meta.get("office_location")
        if not employment_type:
            employment_type = meta.get("employment_type")
        if role == "employee" and meta.get("role"):
            role = meta.get("role")
        if not full_name and user.email and "@" in user.email:
            full_name = user.email.split("@")[0]
    except HTTPException:
        raise
    except Exception:
        pass

    return AuthenticatedUser(
        id=user.id,
        email=user.email,
        role=role,
        full_name=full_name,
        department=department,
        office_location=office_location,
        employment_type=employment_type,
        raw_user=user,
    )

async def require_elevated_role(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    if current_user.role not in ELEVATED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. Role '{current_user.role}' is not authorized to ingest documents. Elevated roles required: {', '.join(sorted(ELEVATED_ROLES))}.",
        )
    return current_user
