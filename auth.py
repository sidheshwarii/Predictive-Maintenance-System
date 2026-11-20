"""
auth.py
-------
Minimal authentication for the student-project dashboard: signup, login,
forgot/reset password, and session tokens.

IMPORTANT: users and tokens are kept IN MEMORY (a Python dict), so they
reset every time the server restarts, and passwords are hashed with
stdlib PBKDF2 rather than a production-grade library. This is enough for
a course project demo — swap in a real database and a library like
passlib/bcrypt before using this anywhere real.
"""

import hashlib
import os
import secrets
import time

# username -> {"salt": bytes, "password_hash": bytes, "email": str}
USERS = {}

# session token -> username
SESSIONS = {}

# reset token -> {"username": str, "expires": float}
RESET_TOKENS = {}
RESET_TOKEN_TTL_SECONDS = 15 * 60  # 15 minutes


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)


def create_user(username: str, password: str, email: str) -> None:
    if username in USERS:
        raise ValueError("Username already exists.")
    salt = os.urandom(16)
    USERS[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "email": email,
    }


def verify_user(username: str, password: str) -> bool:
    user = USERS.get(username)
    if not user:
        return False
    return _hash_password(password, user["salt"]) == user["password_hash"]


def create_session(username: str) -> str:
    token = secrets.token_hex(24)
    SESSIONS[token] = username
    return token


def get_username_from_token(token: str):
    return SESSIONS.get(token)


def logout(token: str) -> None:
    SESSIONS.pop(token, None)


def create_reset_token(username_or_email: str):
    """Returns a reset token if the user/email exists, else None."""
    username = None
    if username_or_email in USERS:
        username = username_or_email
    else:
        for uname, data in USERS.items():
            if data["email"].lower() == username_or_email.lower():
                username = uname
                break
    if username is None:
        return None

    token = secrets.token_urlsafe(24)
    RESET_TOKENS[token] = {"username": username, "expires": time.time() + RESET_TOKEN_TTL_SECONDS}
    return token


def reset_password(token: str, new_password: str) -> bool:
    entry = RESET_TOKENS.get(token)
    if not entry or entry["expires"] < time.time():
        RESET_TOKENS.pop(token, None)
        return False
    username = entry["username"]
    salt = os.urandom(16)
    USERS[username]["salt"] = salt
    USERS[username]["password_hash"] = _hash_password(new_password, salt)
    RESET_TOKENS.pop(token, None)
    return True
