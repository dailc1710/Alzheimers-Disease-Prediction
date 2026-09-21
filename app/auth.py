"""Local authentication primitives for the Streamlit application.

The application intentionally keeps authentication small and dependency-free:
passwords are stored as salted PBKDF2-SHA256 hashes in the local state database.
The first account created through the UI is an administrator; administrators
can then create doctor or viewer accounts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT_STATE_DB = Path(__file__).resolve().parent / "data" / "app_state.sqlite3"
STATE_DB_ENV = "ALZHEIMER_STATE_DB"
AUTH_DISABLED_ENV = "ALZHEIMER_AUTH_DISABLED"
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 310_000
ALLOWED_ROLES = {"viewer", "doctor", "admin"}
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,64}$")


@dataclass(frozen=True)
class AuthUser:
    """Authenticated application identity."""

    username: str
    role: str


def state_database_path() -> Path:
    """Return the configured local state database path."""

    configured = os.environ.get(STATE_DB_ENV, "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_STATE_DB


def authentication_disabled() -> bool:
    """Allow explicit authentication bypass for automated tests only."""

    return os.environ.get(AUTH_DISABLED_ENV, "").strip().lower() in {"1", "true", "yes"}


def validate_username(username: str) -> str:
    """Normalize and validate a login name."""

    normalized = username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Username must contain 3-64 letters, digits, dots, underscores, or hyphens."
        )
    return normalized


def validate_password(password: str) -> None:
    """Apply a small but explicit password-quality policy."""

    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters.")
    if not any(character.isalpha() for character in password):
        raise ValueError("Password must contain at least one letter.")
    if not any(character.isdigit() for character in password):
        raise ValueError("Password must contain at least one digit.")


def hash_password(
    password: str,
    *,
    salt: bytes | None = None,
    iterations: int = PASSWORD_ITERATIONS,
) -> str:
    """Return a versioned salted PBKDF2-SHA256 password representation."""

    validate_password(password)
    chosen_salt = secrets.token_bytes(16) if salt is None else salt
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        chosen_salt,
        iterations,
    )
    encoded_salt = base64.urlsafe_b64encode(chosen_salt).decode("ascii")
    encoded_digest = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"{PASSWORD_SCHEME}${iterations}${encoded_salt}${encoded_digest}"


def verify_password(password: str, encoded: str) -> bool:
    """Verify a password without leaking digest equality through timing."""

    try:
        scheme, raw_iterations, raw_salt, raw_digest = encoded.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(raw_iterations)
        if iterations < 100_000 or iterations > 2_000_000:
            return False
        salt = base64.urlsafe_b64decode(raw_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(raw_digest.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, expected)


def _open_connection(database_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(database_path) if database_path is not None else state_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('viewer', 'doctor', 'admin')),
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            created_at_utc TEXT NOT NULL,
            created_by TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


@contextmanager
def _connect(database_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield one initialized connection and always release its Windows file lock."""

    connection = _open_connection(database_path)
    try:
        yield connection
    finally:
        connection.close()


def user_count(database_path: str | Path | None = None) -> int:
    """Return the number of active users."""

    with _connect(database_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM users WHERE active = 1").fetchone()
    return int(row["count"] if row else 0)


def create_user(
    username: str,
    password: str,
    role: str,
    *,
    created_by: str,
    database_path: str | Path | None = None,
) -> AuthUser:
    """Create one active user after validating identity, role, and password."""

    normalized = validate_username(username)
    if role not in ALLOWED_ROLES:
        raise ValueError(f"Unsupported role: {role}")
    password_hash = hash_password(password)
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with _connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO users (username, password_hash, role, active, created_at_utc, created_by)
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (normalized, password_hash, role, timestamp, created_by),
            )
            connection.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("Username already exists.") from exc
    return AuthUser(username=normalized, role=role)


def authenticate(
    username: str,
    password: str,
    *,
    database_path: str | Path | None = None,
) -> AuthUser | None:
    """Authenticate an active user and return their role."""

    try:
        normalized = validate_username(username)
    except ValueError:
        return None
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT username, password_hash, role FROM users WHERE username = ? AND active = 1",
            (normalized,),
        ).fetchone()
    if row is None or not verify_password(password, str(row["password_hash"])):
        return None
    return AuthUser(username=str(row["username"]), role=str(row["role"]))


def list_users(database_path: str | Path | None = None) -> list[dict[str, str]]:
    """Return non-secret account metadata for administration."""

    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT username, role, created_at_utc, created_by
            FROM users
            WHERE active = 1
            ORDER BY username
            """
        ).fetchall()
    return [dict(row) for row in rows]


__all__ = [
    "AUTH_DISABLED_ENV",
    "ALLOWED_ROLES",
    "AuthUser",
    "authenticate",
    "authentication_disabled",
    "create_user",
    "hash_password",
    "list_users",
    "state_database_path",
    "user_count",
    "validate_password",
    "validate_username",
    "verify_password",
]
