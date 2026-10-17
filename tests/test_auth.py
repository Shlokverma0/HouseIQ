"""Integration checks for account login, JWT protection, CAPTCHA, and email OTP reset."""
import asyncio
import smtplib
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.main import app
from app.repositories.model_repository import model_repository
from app.services import auth_service


class SyncASGIClient:
    def __init__(self):
        self.cookies = httpx.Cookies()

    def request(self, method, path, **kwargs):
        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver", cookies=self.cookies) as client:
                response = await client.request(method, path, **kwargs)
                self.cookies.update(response.cookies)
                return response
        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-only-houseiq-auth-key-1234567890")
    monkeypatch.delenv("RECAPTCHA_SITE_KEY", raising=False)
    monkeypatch.delenv("RECAPTCHA_SECRET_KEY", raising=False)
    monkeypatch.delenv("RECAPTCHA_PROJECT_ID", raising=False)
    monkeypatch.delenv("RECAPTCHA_API_KEY", raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_USERNAME", "test-user")
    monkeypatch.setenv("SMTP_PASSWORD", "test-password")
    monkeypatch.setenv("SMTP_FROM", "houseiq@example.test")
    db_path = Path(__file__).resolve().parents[1] / ".pytest_cache" / f"auth-{uuid4().hex}.sqlite3"
    monkeypatch.setattr(auth_service, "DB_PATH", db_path)
    monkeypatch.setattr(auth_service, "verify_captcha", lambda token, ip=None, expected_action=None: True)
    yield db_path
    db_path.unlink(missing_ok=True)


def _run_startup(monkeypatch):
    from app.main import lifespan

    monkeypatch.setattr(model_repository, "load", lambda: None)

    async def start_and_stop():
        async with lifespan(app):
            pass

    asyncio.run(start_and_stop())


def test_demo_user_is_created_at_startup_and_can_login_without_smtp(auth_env, monkeypatch):
    email = "demo@example.test"
    password = "demo-password-123"
    monkeypatch.setenv("HOUSEIQ_DEMO_EMAIL", email)
    monkeypatch.setenv("HOUSEIQ_DEMO_PASSWORD", password)

    _run_startup(monkeypatch)

    with auth_service._db() as db:
        user = db.execute("SELECT email,password_hash FROM users").fetchone()
        assert user["email"] == email
        assert user["password_hash"].startswith("scrypt$")
        assert auth_service._verify_password(password, user["password_hash"])
    user, _token = auth_service.login(email, password, captcha_token=None)
    assert user["email"] == email


def test_demo_user_startup_does_not_insert_duplicate(auth_env, monkeypatch):
    monkeypatch.setenv("HOUSEIQ_DEMO_EMAIL", "demo@example.test")
    monkeypatch.setenv("HOUSEIQ_DEMO_PASSWORD", "demo-password-123")

    _run_startup(monkeypatch)
    _run_startup(monkeypatch)

    with auth_service._db() as db:
        assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_demo_password_change_updates_only_demo_password_hash(auth_env, monkeypatch):
    monkeypatch.setenv("HOUSEIQ_DEMO_EMAIL", "demo@example.test")
    monkeypatch.setenv("HOUSEIQ_DEMO_PASSWORD", "demo-password-old")
    _run_startup(monkeypatch)
    with auth_service._db() as db:
        db.execute(
            "INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)",
            ("regular@example.test", auth_service._password_hash("regular-password"), 1),
        )
        original_regular_hash = db.execute(
            "SELECT password_hash FROM users WHERE email=?", ("regular@example.test",)
        ).fetchone()[0]
        original_demo_hash = db.execute(
            "SELECT password_hash FROM users WHERE email=?", ("demo@example.test",)
        ).fetchone()[0]

    monkeypatch.setenv("HOUSEIQ_DEMO_PASSWORD", "demo-password-new")
    _run_startup(monkeypatch)

    with auth_service._db() as db:
        demo_hash = db.execute(
            "SELECT password_hash FROM users WHERE email=?", ("demo@example.test",)
        ).fetchone()[0]
        regular_hash = db.execute(
            "SELECT password_hash FROM users WHERE email=?", ("regular@example.test",)
        ).fetchone()[0]
        assert demo_hash != original_demo_hash
        assert auth_service._verify_password("demo-password-new", demo_hash)
        assert not auth_service._verify_password("demo-password-old", demo_hash)
        assert regular_hash == original_regular_hash


def test_demo_user_not_created_when_environment_is_missing(auth_env, monkeypatch):
    monkeypatch.delenv("HOUSEIQ_DEMO_EMAIL", raising=False)
    monkeypatch.delenv("HOUSEIQ_DEMO_PASSWORD", raising=False)

    _run_startup(monkeypatch)

    assert not auth_env.exists()


def _mock_google_verification(monkeypatch, claims=None, error=None):
    from google.oauth2 import id_token

    def verify(credential, request, audience):
        assert credential == "mock-google-id-token"
        assert audience == "houseiq-client-id.apps.googleusercontent.com"
        if error:
            raise error
        return claims

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "houseiq-client-id.apps.googleusercontent.com")
    monkeypatch.setattr(id_token, "verify_oauth2_token", verify)


def _google_claims(**overrides):
    claims = {
        "iss": "https://accounts.google.com",
        "aud": "houseiq-client-id.apps.googleusercontent.com",
        "exp": int(time.time()) + 600,
        "email_verified": True,
        "email": f"google-{uuid4().hex}@example.com",
        "sub": f"google-sub-{uuid4().hex}",
    }
    claims.update(overrides)
    return claims


def test_google_valid_token_creates_session_cookie(auth_env, monkeypatch):
    claims = _google_claims()
    _mock_google_verification(monkeypatch, claims=claims)

    client = SyncASGIClient()
    response = client.post("/auth/google", json={"credential": "mock-google-id-token"})

    assert response.status_code == 200, response.text
    assert response.json()["user"]["email"] == claims["email"]
    assert any(cookie.name == "houseiq_access" for cookie in client.cookies.jar)
    assert client.get("/auth/me").json()["user"]["email"] == claims["email"]


def test_google_login_does_not_require_recaptcha(auth_env, monkeypatch):
    _mock_google_verification(monkeypatch, claims=_google_claims())
    monkeypatch.setattr(auth_service, "verify_captcha", lambda *args, **kwargs: False)

    response = SyncASGIClient().post("/auth/google", json={"credential": "mock-google-id-token"})

    assert response.status_code == 200, response.text


def test_google_wrong_audience_is_rejected(auth_env, monkeypatch):
    _mock_google_verification(monkeypatch, error=ValueError("Wrong recipient"))

    response = SyncASGIClient().post("/auth/google", json={"credential": "mock-google-id-token"})

    assert response.status_code == 401
    assert "audience" in response.json()["detail"].lower()


def test_google_unverified_email_is_rejected(auth_env, monkeypatch):
    _mock_google_verification(monkeypatch, claims=_google_claims(email_verified=False))

    response = SyncASGIClient().post("/auth/google", json={"credential": "mock-google-id-token"})

    assert response.status_code == 401
    assert "email is not verified" in response.json()["detail"]


def test_google_expired_token_is_rejected(auth_env, monkeypatch):
    _mock_google_verification(monkeypatch, error=ValueError("Token expired"))

    response = SyncASGIClient().post("/auth/google", json={"credential": "mock-google-id-token"})

    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_google_login_links_existing_password_account(auth_env, monkeypatch):
    claims = _google_claims(email=f"existing-{uuid4().hex}@example.com")
    _mock_google_verification(monkeypatch, claims=claims)
    with auth_service._db() as db:
        db.execute(
            "INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)",
            (claims["email"], auth_service._password_hash("correct-horse-battery"), int(time.time())),
        )
        password_hash = db.execute("SELECT password_hash FROM users WHERE email=?", (claims["email"],)).fetchone()[0]

    response = SyncASGIClient().post("/auth/google", json={"credential": "mock-google-id-token"})

    assert response.status_code == 200, response.text
    with auth_service._db() as db:
        user = db.execute("SELECT password_hash,google_sub FROM users WHERE email=?", (claims["email"],)).fetchone()
        assert user["password_hash"] == password_hash
        assert user["google_sub"] == claims["sub"]


def test_google_creates_passwordless_account_using_only_verified_token_email(auth_env, monkeypatch):
    claims = _google_claims()
    _mock_google_verification(monkeypatch, claims=claims)

    response = SyncASGIClient().post(
        "/auth/google",
        json={"credential": "mock-google-id-token", "email": "attacker@example.test"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["user"]["email"] == claims["email"]
    with auth_service._db() as db:
        user = db.execute("SELECT password_hash,google_sub FROM users WHERE email=?", (claims["email"],)).fetchone()
        assert user["password_hash"] == ""
        assert user["google_sub"] == claims["sub"]
    password_login = SyncASGIClient().post(
        "/auth/login", json={"email": claims["email"], "password": "anything"}
    )
    assert password_login.status_code == 401
    assert password_login.json()["detail"] == "Sign in with Google."


def test_google_client_id_missing_disables_button_and_endpoint(auth_env, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)

    config = SyncASGIClient().get("/auth/config")
    login_page = SyncASGIClient().get("/login")
    response = SyncASGIClient().post("/auth/google", json={"credential": "mock-google-id-token"})

    assert config.status_code == 200
    assert config.json()["google_client_id"] == ""
    assert 'id="googleLogin" hidden' in login_page.text
    assert response.status_code == 503
    assert response.json()["detail"] == "Google Sign-In is not configured."


def test_register_login_cookie_and_protected_api(auth_env):
    email = f"user-{uuid4().hex}@example.com"
    anonymous = SyncASGIClient()
    denied = anonymous.post("/predict", json={"BHK": 2})
    assert denied.status_code == 401
    assert anonymous.get("/app", follow_redirects=False).status_code == 303
    assert anonymous.get("/login").status_code == 200

    client = SyncASGIClient()
    created = client.post("/auth/register", json={"email": email.upper(), "password": "correct-horse-battery"})
    assert created.status_code == 201, created.text
    cookie = next(cookie for cookie in client.cookies.jar if cookie.name == "houseiq_access")
    assert cookie.has_nonstandard_attr("HttpOnly")
    assert client.get("/auth/me").json()["user"]["email"] == email
    assert client.get("/app").status_code == 200
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").status_code == 401

    logged_in = SyncASGIClient()
    response = logged_in.post("/auth/login", json={"email": email, "password": "correct-horse-battery"})
    assert response.status_code == 200
    assert logged_in.get("/auth/me").status_code == 200


def test_bad_password_and_captcha_failure_are_rejected(auth_env, monkeypatch):
    email = f"captcha-{uuid4().hex}@example.com"
    created = SyncASGIClient().post("/auth/register", json={"email": email, "password": "correct-horse-battery"})
    assert created.status_code == 201

    monkeypatch.setattr(auth_service, "verify_captcha", lambda token, ip=None, expected_action=None: False)
    denied = SyncASGIClient().post("/auth/login", json={"email": email, "password": "correct-horse-battery", "captcha_token": "bad"})
    assert denied.status_code == 400
    monkeypatch.setattr(auth_service, "verify_captcha", lambda token, ip=None, expected_action=None: True)
    bad_password = SyncASGIClient().post("/auth/login", json={"email": email, "password": "wrong-password"})
    assert bad_password.status_code == 401


def test_email_otp_reset_is_expiring_one_time_and_revokes_old_sessions(auth_env, monkeypatch):
    sent = []
    monkeypatch.setattr(auth_service, "_send_otp", lambda email, otp: sent.append((email, otp)))
    email = f"reset-{uuid4().hex}@example.com"
    old_session = SyncASGIClient()
    registered = old_session.post("/auth/register", json={"email": email, "password": "old-password-1234"})
    assert registered.status_code == 201
    requested = old_session.post("/auth/forgot-password", json={"email": email})
    assert requested.status_code == 200
    assert "reset code has been sent" in requested.json()["message"]
    assert sent and email in sent[0][0]
    assert sent[0][1] not in requested.text

    invalid = old_session.post("/auth/verify-reset-otp", json={"email": email, "otp": "000000" if sent[0][1] != "000000" else "000001"})
    assert invalid.status_code == 400
    verified = old_session.post("/auth/verify-reset-otp", json={"email": email, "otp": sent[0][1]})
    assert verified.status_code == 200
    token = verified.json()["reset_token"]
    assert old_session.post("/auth/reset-password", json={"reset_token": token, "new_password": "new-password-5678"}).status_code == 200
    assert old_session.post("/auth/reset-password", json={"reset_token": token, "new_password": "another-password"}).status_code == 400
    assert old_session.get("/auth/me").status_code == 401
    assert old_session.post("/auth/login", json={"email": email, "password": "old-password-1234"}).status_code == 401
    assert old_session.post("/auth/login", json={"email": email, "password": "new-password-5678"}).status_code == 200


def test_forgot_password_does_not_enumerate_unknown_emails(auth_env, monkeypatch):
    sent = []
    monkeypatch.setattr(auth_service, "_send_otp", lambda email, otp: sent.append(email))
    response = SyncASGIClient().post("/auth/forgot-password", json={"email": f"missing-{uuid4().hex}@example.com"})
    assert response.status_code == 200
    assert "If an account exists" in response.json()["message"]
    assert not sent


def test_smtp_failure_returns_actionable_error_instead_of_500(auth_env, monkeypatch):
    email = f"smtp-{uuid4().hex}@example.com"
    client = SyncASGIClient()
    assert client.post("/auth/register", json={"email": email, "password": "correct-horse-battery"}).status_code == 201

    def disconnect(*args, **kwargs):
        raise smtplib.SMTPServerDisconnected("connection unexpectedly closed")

    monkeypatch.setattr(auth_service, "_send_otp", disconnect)
    response = client.post("/auth/forgot-password", json={"email": email})
    assert response.status_code == 503
    assert "Check the SMTP host, port, username, and app password" in response.json()["detail"]
