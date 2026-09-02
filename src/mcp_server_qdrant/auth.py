import secrets

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class BearerAuthMiddleware:
    """
    ASGI middleware that requires a matching `Authorization: Bearer <api_key>`
    header on every HTTP request. Non-HTTP scopes (e.g. lifespan) pass through
    untouched.
    """

    def __init__(self, app: ASGIApp, api_key: str):
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        token = (
            auth_header[len("Bearer ") :]
            if auth_header.startswith("Bearer ")
            else ""
        )

        if not token or not secrets.compare_digest(token, self.api_key):
            response = JSONResponse({"error": "Unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
