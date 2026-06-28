"""
auth.py — Authentication with Supabase + JWT
"""
import os
import bcrypt
import jwt
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx

SUPABASE_URL        = os.getenv("SUPABASE_URL", "")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")
_JWT_DEFAULT        = "stockpulse-secret-change-this"
JWT_SECRET          = os.getenv("JWT_SECRET", _JWT_DEFAULT)
_ENVIRONMENT        = os.getenv("ENVIRONMENT", "development")
if _ENVIRONMENT == "production" and JWT_SECRET == _JWT_DEFAULT:
    raise RuntimeError(
        "JWT_SECRET must be set to a strong random secret in production. "
        "Run: python -c \"import secrets; print(secrets.token_hex(32))\" and add it to .env"
    )
JWT_EXPIRE_HOURS    = 24

security = HTTPBearer(auto_error=False)

HEADERS = {
    "apikey": SUPABASE_SECRET_KEY,
    "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

# Shared, connection-pooled client — avoids a fresh TCP+TLS handshake to
# Supabase on every login/register/lookup call.
_http_client = httpx.AsyncClient(
    timeout=10,
    limits=httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=30),
)

# ── Database helpers ──────────────────────────────────────

async def db_get_user(email: str) -> Optional[dict]:
    resp = await _http_client.get(
        f"{SUPABASE_URL}/rest/v1/users",
        headers=HEADERS,
        params={"email": f"eq.{email}", "limit": "1"},
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    return data[0] if data else None

async def db_create_user(email: str, password_hash: str, full_name: str) -> dict:
    resp = await _http_client.post(
        f"{SUPABASE_URL}/rest/v1/users",
        headers=HEADERS,
        json={"email": email, "password_hash": password_hash, "full_name": full_name},
    )
    if resp.status_code in (200, 201):
        data = resp.json()
        return data[0] if isinstance(data, list) else data
    if resp.status_code == 409:
        raise HTTPException(status_code=409, detail="Email already registered")
    raise HTTPException(status_code=500, detail=f"Database error: {resp.text[:100]}")

# ── Password helpers ──────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# ── JWT helpers ───────────────────────────────────────────

def create_token(user_id: str, email: str, full_name: str) -> str:
    payload = {
        "sub":       user_id,
        "email":     email,
        "full_name": full_name,
        "exp":       datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ── FastAPI dependency ────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not logged in")
    return decode_token(credentials.credentials)

# ── Auth handlers ─────────────────────────────────────────

async def register_user(email: str, password: str, full_name: str) -> dict:
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Supabase not configured in .env")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    existing = await db_get_user(email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    pw_hash = hash_password(password)
    user    = await db_create_user(email, pw_hash, full_name)
    token   = create_token(str(user["id"]), email, full_name)
    return {"token": token, "email": email, "full_name": full_name}

async def login_user(email: str, password: str) -> dict:
    if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Supabase not configured in .env")
    user = await db_get_user(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(str(user["id"]), email, user.get("full_name", ""))
    return {"token": token, "email": email, "full_name": user.get("full_name", "")}