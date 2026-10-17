"""Small SQLite-backed authentication service using standard-library crypto."""
import base64
import hashlib
import hmac
import json
import os
import secrets
import smtplib
import sqlite3
import time
from contextlib import contextmanager
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi import HTTPException, Request as FastAPIRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("HOUSEIQ_AUTH_DB", PROJECT_ROOT / "data" / "houseiq_auth.sqlite3"))
RESET_TTL_SECONDS = 10 * 60
ACCESS_TTL_SECONDS = 60 * 60
OTP_TTL_SECONDS = 10 * 60


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _secret() -> bytes:
    secret = os.getenv("AUTH_SECRET_KEY", "")
    if len(secret.encode("utf-8")) < 32:
        raise HTTPException(status_code=503, detail="Authentication is not configured. Set AUTH_SECRET_KEY to a random value of at least 32 characters.")
    return secret.encode("utf-8")


def _token(payload: dict) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode("ascii")
    signature = _b64(hmac.new(_secret(), signing_input, hashlib.sha256).digest())
    return f"{header}.{body}.{signature}"


def _decode(token: str, expected_type: str = "access") -> dict:
    try:
        header, body, signature = token.split(".")
        decoded_header = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
        if decoded_header != {"alg": "HS256", "typ": "JWT"}:
            raise ValueError("unsupported JWT header")
        signing_input = f"{header}.{body}".encode("ascii")
        expected = _b64(hmac.new(_secret(), signing_input, hashlib.sha256).digest())
        if not hmac.compare_digest(expected, signature):
            raise ValueError("bad signature")
        claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if claims.get("typ") != expected_type or int(claims.get("exp", 0)) <= int(time.time()):
            raise ValueError("expired or wrong token type")
        return claims
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Session is invalid or expired. Please sign in again.") from exc


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
        token_version INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL
    )""")
    user_columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
    if "google_sub" not in user_columns:
        connection.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub "
        "ON users(google_sub) WHERE google_sub IS NOT NULL"
    )
    connection.execute("""CREATE TABLE IF NOT EXISTS password_otps (
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        otp_hash TEXT NOT NULL, expires_at INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        consumed INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS used_reset_tokens (
        jti TEXT PRIMARY KEY, expires_at INTEGER NOT NULL
    )""")
    connection.commit()
    return connection


@contextmanager
def _db():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _normalise_email(email: str) -> str:
    return email.strip().lower()


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${_b64(salt)}${_b64(digest)}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt, expected = stored.split("$")
        if algorithm != "scrypt":
            return False
        actual = _password_hash(password, base64.urlsafe_b64decode(salt + "=" * (-len(salt) % 4))).split("$")[2]
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def verify_captcha(captcha_token: str | None, remote_ip: str | None = None, expected_action: str | None = None) -> bool:
    site_key = os.getenv("RECAPTCHA_SITE_KEY", "").strip()
    project_id = os.getenv("RECAPTCHA_PROJECT_ID", "").strip()
    api_key = os.getenv("RECAPTCHA_API_KEY", "").strip()
    if not site_key and not project_id and not api_key:
        return os.getenv("APP_ENV", "development").lower() != "production"
    if not site_key or not project_id or not api_key or not captcha_token:
        return False
    payload = {
        "event": {
            "token": captcha_token,
            "siteKey": site_key,
            "userIpAddress": remote_ip,
            "expectedAction": expected_action,
        }
    }
    try:
        url = f"https://recaptchaenterprise.googleapis.com/v1/projects/{quote(project_id, safe='')}/assessments"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Goog-Api-Key": api_key},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
        token_properties = result.get("tokenProperties") or {}
        score = float((result.get("riskAnalysis") or {}).get("score", 0))
        threshold = float(os.getenv("RECAPTCHA_MIN_SCORE", "0.5"))
        action_matches = expected_action is None or token_properties.get("action") == expected_action
        return token_properties.get("valid") is True and action_matches and score >= threshold
    except Exception:
        return False


def _require_captcha(captcha_token: str | None, remote_ip: str | None, expected_action: str):
    if not verify_captcha(captcha_token, remote_ip, expected_action):
        raise HTTPException(status_code=400, detail="CAPTCHA verification failed. Please try again.")


def _send_otp(email: str, otp: str):
    host = os.getenv("SMTP_HOST", "")
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_FROM", username)
    if not all((host, username, password, sender)):
        raise HTTPException(status_code=503, detail="Email delivery is not configured. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and SMTP_FROM.")
    message = EmailMessage()
    message["Subject"] = "Your HouseIQ password reset code"
    message["From"] = sender
    message["To"] = email
    message.set_content(f"Your HouseIQ password reset code is {otp}. It expires in 10 minutes. If you did not request this, ignore this email.")
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)


def _email_settings_ready():
    required = ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM")
    if any(not os.getenv(name) for name in required):
        raise HTTPException(status_code=503, detail="Email delivery is not configured. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and SMTP_FROM.")
    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="SMTP_PORT must be a valid TCP port.") from exc


def _issue_access(user_id: int, email: str, token_version: int = 0) -> str:
    now = int(time.time())
    return _token({"sub": str(user_id), "email": email, "ver": token_version, "typ": "access", "iat": now, "exp": now + ACCESS_TTL_SECONDS})


def register(email: str, password: str, captcha_token: str | None, remote_ip: str | None = None) -> tuple[dict, str]:
    _secret()
    _require_captcha(captcha_token, remote_ip, "register")
    email = _normalise_email(email)
    if len(password) < 10 or len(password) > 128:
        raise HTTPException(status_code=422, detail="Password must be 10–128 characters long.")
    try:
        with _db() as db:
            cursor = db.execute("INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)", (email, _password_hash(password), int(time.time())))
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    return {"id": user_id, "email": email}, _issue_access(user_id, email)


def ensure_demo_user() -> bool:
    """Create or synchronize only the explicitly configured demo account.

    Demo accounts do not use the signup flow, so no email/SMTP verification is
    needed. The app currently has no separate email-verification state.
    """
    email = _normalise_email(os.getenv("HOUSEIQ_DEMO_EMAIL", ""))
    password = os.getenv("HOUSEIQ_DEMO_PASSWORD", "")
    if not email or not password:
        return False
    if len(password) < 10 or len(password) > 128:
        raise ValueError("HOUSEIQ_DEMO_PASSWORD must be 10–128 characters long.")

    with _db() as db:
        user = db.execute(
            "SELECT id,password_hash FROM users WHERE email=?", (email,)
        ).fetchone()
        if user is None:
            db.execute(
                "INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)",
                (email, _password_hash(password), int(time.time())),
            )
        elif not _verify_password(password, user["password_hash"]):
            db.execute(
                "UPDATE users SET password_hash=? WHERE id=?",
                (_password_hash(password), user["id"]),
            )
    return True


def login(email: str, password: str, captcha_token: str | None, remote_ip: str | None = None) -> tuple[dict, str]:
    _require_captcha(captcha_token, remote_ip, "login")
    email = _normalise_email(email)
    with _db() as db:
        user = db.execute("SELECT id,email,password_hash,token_version FROM users WHERE email=?", (email,)).fetchone()
    if user and not user["password_hash"]:
        raise HTTPException(status_code=401, detail="Sign in with Google.")
    if not user or not _verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    return {"id": user["id"], "email": user["email"]}, _issue_access(user["id"], user["email"], user["token_version"])


def _verify_google_id_token(credential: str) -> dict:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(status_code=503, detail="Google Sign-In is not configured.")
    try:
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2 import id_token

        claims = id_token.verify_oauth2_token(credential, GoogleRequest(), audience=client_id)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Google Sign-In dependency is unavailable.") from exc
    except ValueError as exc:
        reason = str(exc).lower()
        if "audience" in reason or "recipient" in reason:
            detail = "Google ID token audience does not match this app."
        elif "expired" in reason:
            detail = "Google ID token has expired."
        else:
            detail = "Google ID token is invalid. Sign in with Google again."
        raise HTTPException(status_code=401, detail=detail) from exc
    except Exception as exc:
        # Certificate fetch/network errors are transient; never include token data.
        raise HTTPException(status_code=503, detail="Google Sign-In could not be verified right now. Try again.") from exc

    if claims.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(status_code=401, detail="Google ID token issuer is invalid.")
    audience = claims.get("aud")
    if client_id not in (audience if isinstance(audience, list) else [audience]):
        raise HTTPException(status_code=401, detail="Google ID token audience does not match this app.")
    try:
        expires_at = int(claims.get("exp", 0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Google ID token expiry is invalid.") from exc
    if expires_at <= int(time.time()):
        raise HTTPException(status_code=401, detail="Google ID token has expired.")
    if claims.get("email_verified") is not True:
        raise HTTPException(status_code=401, detail="Google account email is not verified.")
    email = claims.get("email")
    subject = claims.get("sub")
    if not isinstance(email, str) or not email.strip() or not isinstance(subject, str) or not subject.strip():
        raise HTTPException(status_code=401, detail="Google ID token is missing account details.")
    return {"email": _normalise_email(email), "sub": subject}


def google_login(credential: str) -> tuple[dict, str]:
    """Verify a GIS ID token and sign in or link the verified Google identity."""
    _secret()
    identity = _verify_google_id_token(credential)
    email, subject = identity["email"], identity["sub"]
    now = int(time.time())
    with _db() as db:
        user = db.execute(
            "SELECT id,email,password_hash,token_version,google_sub FROM users WHERE google_sub=?",
            (subject,),
        ).fetchone()
        if user:
            if user["email"] != email:
                email_owner = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
                if email_owner and email_owner["id"] != user["id"]:
                    raise HTTPException(status_code=409, detail="This Google account email is linked to another HouseIQ account.")
                db.execute("UPDATE users SET email=? WHERE id=?", (email, user["id"]))
                user = db.execute(
                    "SELECT id,email,password_hash,token_version,google_sub FROM users WHERE id=?",
                    (user["id"],),
                ).fetchone()
        else:
            user = db.execute(
                "SELECT id,email,password_hash,token_version,google_sub FROM users WHERE email=?",
                (email,),
            ).fetchone()
            if user:
                if user["google_sub"] and user["google_sub"] != subject:
                    raise HTTPException(status_code=409, detail="This HouseIQ account is already linked to a different Google account.")
                db.execute("UPDATE users SET google_sub=? WHERE id=?", (subject, user["id"]))
                user = db.execute(
                    "SELECT id,email,password_hash,token_version,google_sub FROM users WHERE id=?",
                    (user["id"],),
                ).fetchone()
            else:
                cursor = db.execute(
                    "INSERT INTO users(email,password_hash,token_version,created_at,google_sub) VALUES(?,?,0,?,?)",
                    (email, "", now, subject),
                )
                user = db.execute(
                    "SELECT id,email,password_hash,token_version,google_sub FROM users WHERE id=?",
                    (cursor.lastrowid,),
                ).fetchone()
    return {"id": user["id"], "email": user["email"]}, _issue_access(
        user["id"], user["email"], user["token_version"]
    )


def current_user(request: FastAPIRequest) -> dict:
    authorization = request.headers.get("authorization", "")
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else request.cookies.get("houseiq_access", "")
    claims = _decode(token)
    with _db() as db:
        row = db.execute("SELECT id,email,token_version FROM users WHERE id=?", (int(claims["sub"]),)).fetchone()
    if not row or row["email"] != claims.get("email") or row["token_version"] != claims.get("ver", 0):
        raise HTTPException(status_code=401, detail="Session is no longer valid. Please sign in again.")
    return {"id": int(claims["sub"]), "email": claims["email"]}


def revoke_session(token: str | None):
    """Invalidate the account's current JWT version when a user signs out."""
    if not token:
        return
    try:
        claims = _decode(token)
        user_id = int(claims["sub"])
    except (HTTPException, KeyError, TypeError, ValueError):
        return
    with _db() as db:
        db.execute("UPDATE users SET token_version=token_version+1 WHERE id=?", (user_id,))


def request_password_reset(email: str, captcha_token: str | None, remote_ip: str | None = None):
    _require_captcha(captcha_token, remote_ip, "forgot_password")
    _email_settings_ready()
    email = _normalise_email(email)
    with _db() as db:
        user = db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if user:
            now, otp = int(time.time()), f"{secrets.randbelow(1_000_000):06d}"
            db.execute("UPDATE password_otps SET consumed=1 WHERE user_id=? AND consumed=0", (user["id"],))
            cursor = db.execute("INSERT INTO password_otps(user_id,otp_hash,expires_at,created_at) VALUES(?,?,?,?)", (user["id"], hmac.new(_secret(), f"otp:{user['id']}:{otp}".encode(), hashlib.sha256).hexdigest(), now + OTP_TTL_SECONDS, now))
            otp_id = cursor.lastrowid
        else:
            otp = None
            otp_id = None
    if otp:
        try:
            _send_otp(email, otp)
        except (smtplib.SMTPException, OSError, ValueError) as exc:
            with _db() as db:
                db.execute("UPDATE password_otps SET consumed=1 WHERE id=?", (otp_id,))
            raise HTTPException(
                status_code=503,
                detail="We couldn't send the reset email. Check the SMTP host, port, username, and app password, then try again.",
            ) from exc
        except Exception as exc:
            with _db() as db:
                db.execute("UPDATE password_otps SET consumed=1 WHERE id=?", (otp_id,))
            raise HTTPException(
                status_code=503,
                detail="The reset email could not be sent. Check your SMTP settings and try again.",
            ) from exc
    return {"message": "If an account exists for that email, a reset code has been sent. It expires in 10 minutes."}


def verify_reset_otp(email: str, otp: str) -> str:
    email = _normalise_email(email)
    with _db() as db:
        row = db.execute("SELECT o.id,o.user_id,o.otp_hash,o.expires_at,o.attempts,u.email FROM password_otps o JOIN users u ON u.id=o.user_id WHERE u.email=? AND o.consumed=0 ORDER BY o.id DESC LIMIT 1", (email,)).fetchone()
        if not row or row["expires_at"] <= int(time.time()) or row["attempts"] >= 5:
            raise HTTPException(status_code=400, detail="The reset code is invalid or expired.")
        expected = hmac.new(_secret(), f"otp:{row['user_id']}:{otp}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, row["otp_hash"]):
            db.execute("UPDATE password_otps SET attempts=attempts+1 WHERE id=?", (row["id"],))
            invalid_code = True
        else:
            invalid_code = False
            db.execute("UPDATE password_otps SET consumed=1 WHERE id=?", (row["id"],))
    if invalid_code:
        raise HTTPException(status_code=400, detail="The reset code is invalid or expired.")
    now, jti = int(time.time()), secrets.token_urlsafe(18)
    return _token({"sub": str(row["user_id"]), "email": row["email"], "typ": "password_reset", "jti": jti, "iat": now, "exp": now + RESET_TTL_SECONDS})


def reset_password(reset_token: str, password: str):
    if len(password) < 10 or len(password) > 128:
        raise HTTPException(status_code=422, detail="Password must be 10–128 characters long.")
    claims = _decode(reset_token, "password_reset")
    with _db() as db:
        try:
            db.execute("INSERT INTO used_reset_tokens(jti,expires_at) VALUES(?,?)", (claims["jti"], claims["exp"]))
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=400, detail="This reset link has already been used.") from exc
        db.execute("UPDATE users SET password_hash=?, token_version=token_version+1 WHERE id=?", (_password_hash(password), int(claims["sub"])))
    return {"message": "Password reset successfully. Please sign in."}


def revoke_access(user_id: int, token: str):
    claims = _decode(token)
    jti = claims.get("jti")
    if jti:
        with _db() as db:
            db.execute("INSERT OR IGNORE INTO used_reset_tokens(jti,expires_at) VALUES(?,?)", (jti, claims["exp"]))
