"""
Optional token auth for the Leo web dashboard.

Set LEO_WEB_TOKEN in .env to require a bearer token on /api/* and /ws.
The HTML shell (/) is always public so the login form can load.
Bind defaults to 127.0.0.1 — set LEO_WEB_HOST=0.0.0.0 for LAN access
(strongly recommended to also set LEO_WEB_TOKEN).
"""

import os
from typing import Optional

from fastapi import HTTPException, Request, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware


def web_token() -> str:
    return os.getenv("LEO_WEB_TOKEN", "").strip()


def web_host() -> str:
    return os.getenv("LEO_WEB_HOST", "127.0.0.1")


def _extract_token(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    header = request.headers.get("X-Leo-Token", "").strip()
    if header:
        return header
    return request.query_params.get("token", "").strip() or None


def _check_token(provided: Optional[str]) -> None:
    expected = web_token()
    if not expected:
        return
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing dashboard token")


class WebAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)
        if path == "/api/health":
            return await call_next(request)
        _check_token(_extract_token(request))
        return await call_next(request)


async def verify_ws_token(ws: WebSocket) -> bool:
    """Return False if connection was rejected (caller should return early)."""
    expected = web_token()
    if not expected:
        return True
    provided = ws.query_params.get("token", "").strip()
    if provided != expected:
        await ws.close(code=4401, reason="Unauthorized")
        return False
    return True