"""Auth Module — authentication, authorization, JWT, OAuth, RBAC, rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast


class AuthMethod(StrEnum):
    JWT = "jwt"
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    SESSION = "session"
    BASIC = "basic"
    SAML = "saml"
    OIDC = "oidc"


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_ANALYTICS = "view_analytics"
    EXPORT = "export"


@dataclass
class Role:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    permissions: list[str] = field(default_factory=list)
    description: str = ""
    is_default: bool = False


@dataclass
class User:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    email: str = ""
    username: str = ""
    password_hash: str = ""
    roles: list[str] = field(default_factory=list)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenPair:
    access_token: str = ""
    refresh_token: str = ""
    token_type: str = "Bearer"
    expires_in: int = 3600
    created_at: float = field(default_factory=time.time)


@dataclass
class RateLimitRule:
    name: str = ""
    max_requests: int = 100
    window_seconds: float = 60.0
    per: str = "ip"  # ip, user, api_key


@dataclass
class AuditEntry:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_id: str = ""
    action: str = ""
    resource: str = ""
    timestamp: float = field(default_factory=time.time)
    ip_address: str = ""
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class PasswordHasher:
    """Secure password hashing."""

    @staticmethod
    def hash_password(password: str, salt: str | None = None) -> str:
        if not salt:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return f"{salt}${hashed.hex()}"

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        if "$" not in password_hash:
            return False
        salt, stored_hash = password_hash.split("$", 1)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return hmac.compare_digest(hashed.hex(), stored_hash)


class TokenManager:
    """JWT-like token management."""

    def __init__(
        self, secret: str | None = None, access_ttl: int = 3600, refresh_ttl: int = 86400
    ) -> None:
        # Fail closed: an unset secret must never fall back to a well-known
        # value, or tokens would be forgeable. Generate a random one instead;
        # pass an explicit secret to share tokens across instances.
        if not secret:
            secret = secrets.token_urlsafe(32)
        self.secret = secret
        self.access_ttl = access_ttl
        self.refresh_ttl = refresh_ttl
        self._revoked: set[str] = set()

    def create_tokens(self, user_id: str, roles: list[str] | None = None) -> TokenPair:
        import base64
        import json

        def _make_token(payload: dict, ttl: int) -> str:
            payload["exp"] = time.time() + ttl
            payload["iat"] = time.time()
            payload["jti"] = uuid.uuid4().hex[:8]
            body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
            sig = hmac.new(self.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            return f"{body}.{sig}"

        access = _make_token(
            {"sub": user_id, "type": "access", "roles": roles or []}, self.access_ttl
        )
        refresh = _make_token({"sub": user_id, "type": "refresh"}, self.refresh_ttl)
        return TokenPair(access_token=access, refresh_token=refresh, expires_in=self.access_ttl)

    def verify_token(self, token: str) -> dict[str, Any] | None:
        import base64
        import json

        if token in self._revoked:
            return None
        try:
            body, sig = token.rsplit(".", 1)
            expected = hmac.new(self.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected):
                return None
            payload = json.loads(base64.urlsafe_b64decode(body + "=="))
            if payload.get("exp", 0) < time.time():
                return None
            return cast("dict[str, Any] | None", payload)
        except Exception:
            return None

    def revoke_token(self, token: str) -> None:
        self._revoked.add(token)

    def refresh(self, refresh_token: str) -> TokenPair | None:
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        return self.create_tokens(payload["sub"], payload.get("roles"))


class RBAC:
    """Role-based access control."""

    def __init__(self) -> None:
        self.roles: dict[str, Role] = {}
        self.user_roles: dict[str, list[str]] = {}
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        self.roles["admin"] = Role(
            name="admin",
            permissions=["read", "write", "delete", "admin", "manage_users", "manage_roles"],
            is_default=False,
        )
        self.roles["user"] = Role(name="user", permissions=["read", "write"], is_default=True)
        self.roles["viewer"] = Role(name="viewer", permissions=["read"], is_default=False)
        self.roles["moderator"] = Role(
            name="moderator", permissions=["read", "write", "delete"], is_default=False
        )

    def assign_role(self, user_id: str, role_name: str) -> bool:
        if role_name not in self.roles:
            return False
        self.user_roles.setdefault(user_id, [])
        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)
        return True

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        if user_id in self.user_roles and role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)
            return True
        return False

    def has_permission(self, user_id: str, permission: str) -> bool:
        user_roles = self.user_roles.get(user_id, [])
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role and permission in role.permissions:
                return True
        return False

    def get_user_permissions(self, user_id: str) -> list[str]:
        perms = set()
        for role_name in self.user_roles.get(user_id, []):
            role = self.roles.get(role_name)
            if role:
                perms.update(role.permissions)
        return sorted(perms)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self) -> None:
        self.rules: dict[str, RateLimitRule] = {}
        self._buckets: dict[str, list[float]] = {}

    def add_rule(self, rule: RateLimitRule) -> None:
        self.rules[rule.name] = rule

    def check(self, rule_name: str, identifier: str) -> dict[str, Any]:
        rule = self.rules.get(rule_name)
        if not rule:
            return {"allowed": True, "remaining": -1}
        key = f"{rule_name}:{identifier}"
        now = time.time()
        window_start = now - rule.window_seconds
        requests = self._buckets.get(key, [])
        requests = [t for t in requests if t > window_start]
        remaining = max(0, rule.max_requests - len(requests))
        allowed = len(requests) < rule.max_requests
        if allowed:
            requests.append(now)
        self._buckets[key] = requests
        return {
            "allowed": allowed,
            "remaining": remaining,
            "limit": rule.max_requests,
            "window": rule.window_seconds,
        }


class AuditLogger:
    """Security audit logging."""

    def __init__(self) -> None:
        self.entries: list[AuditEntry] = []

    def log(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool = True,
        ip_address: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            user_id=user_id,
            action=action,
            resource=resource,
            success=success,
            ip_address=ip_address,
            details=details or {},
        )
        self.entries.append(entry)
        return entry

    def get_user_actions(self, user_id: str, limit: int = 50) -> list[AuditEntry]:
        return [e for e in self.entries if e.user_id == user_id][-limit:]

    def get_failed_actions(self, limit: int = 50) -> list[AuditEntry]:
        return [e for e in self.entries if not e.success][-limit:]


class AuthModule:
    """Standalone auth module."""

    NAME = "auth"
    DESCRIPTION = "Authentication, authorization, JWT, OAuth2, RBAC, rate limiting, audit"

    def __init__(self) -> None:
        self.hasher = PasswordHasher()
        self.tokens = TokenManager()
        self.rbac = RBAC()
        self.rate_limiter = RateLimiter()
        self.audit = AuditLogger()
        self.users: dict[str, User] = {}

    def register_user(self, email: str, username: str, password: str) -> User:
        user = User(
            email=email,
            username=username,
            password_hash=self.hasher.hash_password(password),
        )
        self.users[user.id] = user
        self.rbac.assign_role(user.id, "user")
        return user

    def authenticate(self, email: str, password: str) -> TokenPair | None:
        for user in self.users.values():
            if user.email == email and self.hasher.verify_password(password, user.password_hash):
                user.last_login = time.time()
                tokens = self.tokens.create_tokens(user.id, user.roles)
                self.audit.log(user.id, "login", "auth", True)
                return tokens
        return None

    def authorize(self, user_id: str, permission: str) -> bool:
        allowed = self.rbac.has_permission(user_id, permission)
        self.audit.log(user_id, f"authorize:{permission}", "rbac", allowed)
        return allowed

    def execute(self, task: Any) -> dict[str, Any]:
        tags = getattr(task, "tags", [])
        methods = []
        if "jwt" in tags or "api" in tags:
            methods.append({"method": "JWT", "description": "Stateless tokens for APIs"})
        if "oauth" in tags or "sso" in tags:
            methods.append({"method": "OAuth2/OIDC", "description": "SSO, social login"})
        if "session" in tags:
            methods.append({"method": "Session", "description": "Server-side sessions"})
        if not methods:
            methods = [
                {"method": "JWT + Refresh Token", "description": "Recommended for most APIs"},
                {"method": "OAuth2 + PKCE", "description": "For third-party integrations"},
            ]
        return {
            "type": "auth",
            "recommended_methods": methods,
            "rbac_roles": list(self.rbac.roles.keys()),
            "security_features": [
                "password_hashing",
                "token_revocation",
                "rate_limiting",
                "audit_logging",
            ],
            "_confidence": 0.9,
        }
