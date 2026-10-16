"""Registration, login, logout, and email OTP password recovery endpoints."""
import os
import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from app.services import auth_service
from app.utils.limiter import limiter


router = APIRouter(prefix="/auth", tags=["Authentication"])


def _validate_email(value: str) -> str:
    value = value.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise ValueError("Enter a valid email address.")
    return value


class Credentials(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)
    captcha_token: str | None = Field(default=None, max_length=4096)

    _email_validator = field_validator("email")(_validate_email)


class EmailRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    captcha_token: str | None = Field(default=None, max_length=4096)

    _email_validator = field_validator("email")(_validate_email)


class VerifyOtpRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    otp: str = Field(pattern=r"^\d{6}$")

    _email_validator = field_validator("email")(_validate_email)


class PasswordResetRequest(BaseModel):
    reset_token: str = Field(min_length=20, max_length=4096)
    new_password: str = Field(min_length=10, max_length=128)


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def _set_access_cookie(response: Response, token: str):
    response.set_cookie(
        "houseiq_access", token, max_age=auth_service.ACCESS_TTL_SECONDS,
        httponly=True, secure=os.getenv("APP_ENV", "development").lower() == "production",
        samesite="lax", path="/",
    )


@router.get("/config")
def auth_config():
    return {
        "recaptcha_site_key": os.getenv("RECAPTCHA_SITE_KEY", ""),
        "recaptcha_enabled": bool(
            os.getenv("RECAPTCHA_SITE_KEY")
            and os.getenv("RECAPTCHA_PROJECT_ID")
            and os.getenv("RECAPTCHA_API_KEY")
        ),
    }


@router.post("/register", status_code=201)
@limiter.limit("5/hour")
def register(request: Request, response: Response, body: Credentials):
    user, token = auth_service.register(body.email, body.password, body.captcha_token, _client_ip(request))
    _set_access_cookie(response, token)
    return {"user": user}


@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, response: Response, body: Credentials):
    user, token = auth_service.login(body.email, body.password, body.captcha_token, _client_ip(request))
    _set_access_cookie(response, token)
    return {"user": user}


@router.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("houseiq_access", "")
    if not token:
        authorization = request.headers.get("authorization", "")
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    auth_service.revoke_session(token)
    response.delete_cookie("houseiq_access", path="/")
    return {"message": "Signed out."}


@router.get("/me")
def me(user: dict = Depends(auth_service.current_user)):
    return {"user": user}


@router.post("/forgot-password")
@limiter.limit("3/hour")
def forgot_password(request: Request, body: EmailRequest):
    return auth_service.request_password_reset(body.email, body.captcha_token, _client_ip(request))


@router.post("/verify-reset-otp")
@limiter.limit("10/hour")
def verify_reset_otp(request: Request, body: VerifyOtpRequest):
    return {"reset_token": auth_service.verify_reset_otp(body.email, body.otp)}


@router.post("/reset-password")
@limiter.limit("5/hour")
def reset_password(request: Request, body: PasswordResetRequest):
    return auth_service.reset_password(body.reset_token, body.new_password)
