import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from config import logger, supabase_client
from dependencies import AuthenticatedUser, get_current_user
from schemas import (
    AuthLoginRequest,
    AuthResponse,
    AuthSignUpRequest,
    ResendConfirmationRequest,
    SignUpResponse,
    UpdateProfileRequest,
)

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

@router.post("/signup", response_model=SignUpResponse)
async def signup(req: AuthSignUpRequest):
    def _signup():
        credentials = {
            "email": req.email,
            "password": req.password,
        }
        if req.redirect_to:
            credentials["options"] = {"email_redirect_to": req.redirect_to}
        return supabase_client.auth.sign_up(credentials)

    try:
        res = await asyncio.to_thread(_signup)
        if not res or not res.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register user account with Supabase.",
            )
        user = res.user
        now = datetime.now(timezone.utc).isoformat()

        def _upsert_user():
            return (
                supabase_client.table("users")
                .upsert({
                    "id": user.id,
                    "email": req.email,
                    "full_name": "",
                    "role": "employee",
                    "department": "",
                    "created_at": now,
                })
                .execute()
            )

        try:
            await asyncio.to_thread(_upsert_user)
        except Exception as e:
            logger.warning(f"Could not upsert profile in users table for {user.id}: {e}")

        if res.session and res.session.access_token:
            return SignUpResponse(
                access_token=res.session.access_token,
                token_type="bearer",
                user={
                    "id": user.id,
                    "email": user.email,
                    "role": "employee",
                    "full_name": user.email.split("@")[0],
                    "department": "General",
                },
                requires_confirmation=False,
                email=user.email,
                message="Account registered successfully.",
            )

        return SignUpResponse(
            access_token="",
            token_type="bearer",
            user={
                "id": user.id,
                "email": user.email,
                "role": "employee",
                "full_name": user.email.split("@")[0],
                "department": "General",
            },
            requires_confirmation=True,
            email=user.email,
            message="Confirmation link sent to your email. Click the link in the email to activate your account.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Signup error for {req.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}",
        )

@router.post("/resend-confirmation")
async def resend_confirmation(req: ResendConfirmationRequest):
    def _resend():
        credentials = {
            "email": req.email,
            "type": "signup",
        }
        if req.redirect_to:
            credentials["options"] = {"email_redirect_to": req.redirect_to}
        return supabase_client.auth.resend(credentials)

    try:
        await asyncio.to_thread(_resend)
        return {"message": "Confirmation link resent successfully."}
    except Exception as e:
        logger.warning(f"Resend confirmation failure for {req.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to resend confirmation email. Please wait before retrying.",
        )

@router.put("/profile")
async def update_profile(req: UpdateProfileRequest, current_user: AuthenticatedUser = Depends(get_current_user)):
    role = req.role if req.role in ("employee", "hr_admin", "it_admin") else "employee"
    user_email = current_user.email or ""
    if not user_email and current_user.raw_user:
        user_email = getattr(current_user.raw_user, "email", "") or ""

    fallback_name = user_email.split("@")[0] if "@" in user_email else "User"
    full_name = req.full_name.strip() or fallback_name
    department = req.department.strip() or "General"
    now = datetime.now(timezone.utc).isoformat()

    def _update():
        update_payload = {
            "full_name": full_name,
            "department": department,
            "role": role,
            "updated_at": now,
        }
        try:
            up_res = (
                supabase_client.table("users")
                .update(update_payload)
                .eq("id", current_user.id)
                .execute()
            )
            if up_res.data:
                return up_res
        except Exception as update_err:
            logger.warning(f"Direct update failed for {current_user.id}: {update_err}")

        upsert_payload = {
            "id": current_user.id,
            "full_name": full_name,
            "department": department,
            "role": role,
            "updated_at": now,
            "is_active": True,
        }
        if user_email:
            upsert_payload["email"] = user_email
        return (
            supabase_client.table("users")
            .upsert(upsert_payload, on_conflict="id")
            .execute()
        )

    def _sync_metadata():
        try:
            supabase_client.auth.admin.update_user_by_id(
                current_user.id,
                {
                    "user_metadata": {
                        "full_name": full_name,
                        "department": department,
                        "role": role,
                    }
                },
            )
        except Exception as meta_err:
            logger.warning(f"Could not sync auth metadata for {current_user.id}: {meta_err}")

    try:
        await asyncio.to_thread(_update)
        await asyncio.to_thread(_sync_metadata)
        return {
            "id": current_user.id,
            "email": user_email,
            "full_name": full_name,
            "department": department,
            "role": role,
        }
    except Exception as e:
        logger.error(f"Profile update failure for {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile details: {str(e)}",
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
