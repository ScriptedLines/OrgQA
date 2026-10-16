import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from config import logger, supabase_client
from dependencies import AuthenticatedUser, get_current_user
from schemas import AuthLoginRequest, AuthResponse, AuthSignUpRequest

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=AuthResponse)
async def login(req: AuthLoginRequest):
    def _login():
        return supabase_client.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password,
        })

    try:
        res = await asyncio.to_thread(_login)
        if not res or not res.session or not res.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )
        user = res.user
        meta = user.user_metadata or {}
        role = meta.get("role", "employee")
        full_name = meta.get("full_name") or req.email.split("@")[0]
        department = meta.get("department", "General")

        def _fetch_profile():
            return supabase_client.table("users").select("*").eq("id", user.id).execute()

        try:
            profile_res = await asyncio.to_thread(_fetch_profile)
            if profile_res.data:
                profile = profile_res.data[0]
                role = profile.get("role") or role
                full_name = profile.get("full_name") or full_name
                department = profile.get("department") or department
        except Exception:
            pass

        return AuthResponse(
            access_token=res.session.access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "email": user.email,
                "role": role,
                "full_name": full_name,
                "department": department,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Login failure for {req.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed. Please verify your email and password.",
        )

@router.post("/signup", response_model=AuthResponse)
async def signup(req: AuthSignUpRequest):
    role = req.role if req.role in ("employee", "hr_admin", "it_admin") else "employee"
    full_name = req.full_name or req.email.split("@")[0]
    department = req.department or "General"

    def _signup():
        return supabase_client.auth.sign_up({
            "email": req.email,
            "password": req.password,
            "options": {
                "data": {
                    "full_name": full_name,
                    "role": role,
                    "department": department,
                }
            },
        })

    try:
        res = await asyncio.to_thread(_signup)
        if not res or not res.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register user account.",
            )
        user = res.user
        now = datetime.now(timezone.utc).isoformat()

        def _upsert_user():
            return (
                supabase_client.table("users")
                .upsert({
                    "id": user.id,
                    "email": req.email,
                    "full_name": full_name,
                    "role": role,
                    "department": department,
                    "created_at": now,
                })
                .execute()
            )

        try:
            await asyncio.to_thread(_upsert_user)
        except Exception as e:
            logger.warning(f"Could not upsert profile in users table for {user.id}: {e}")

        access_token = res.session.access_token if res.session else ""
        return AuthResponse(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user.id,
                "email": user.email,
                "role": role,
                "full_name": full_name,
                "department": department,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Signup error for {req.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}",
        )

@router.get("/me")
async def get_me(current_user: AuthenticatedUser = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "full_name": current_user.full_name,
        "department": current_user.department,
        "office_location": current_user.office_location,
        "employment_type": current_user.employment_type,
    }

