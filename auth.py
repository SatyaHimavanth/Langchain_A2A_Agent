import os
from dataclasses import dataclass

from a2a.auth.user import UnauthenticatedUser
from a2a.auth.user import User as A2AUser
from a2a.extensions.common import HTTP_EXTENSION_HEADER, get_requested_extensions
from a2a.server.apps.jsonrpc.jsonrpc_app import CallContextBuilder
from a2a.server.context import ServerCallContext


DEFAULT_TOKEN = 'dummy-token-for-extended-card'


@dataclass(frozen=True)
class BearerUser(A2AUser):
    name: str

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self.name


class BearerTokenCallContextBuilder(CallContextBuilder):
    def __init__(self, valid_tokens: set[str]):
        self.valid_tokens = valid_tokens

    @classmethod
    def from_env(cls, env_var: str = 'A2A_AUTH_TOKENS') -> 'BearerTokenCallContextBuilder':
        raw_tokens = os.getenv(env_var, DEFAULT_TOKEN)
        valid_tokens = {token.strip() for token in raw_tokens.split(',') if token.strip()}
        return cls(valid_tokens=valid_tokens)

    def build(self, request) -> ServerCallContext:  # type: ignore[override]
        auth_header = request.headers.get('authorization', '')
        token = self._extract_bearer_token(auth_header)
        is_valid = token in self.valid_tokens if token else False

        state: dict[str, object] = {
            'headers': dict(request.headers),
            'auth_token_valid': is_valid,
        }
        user: A2AUser = BearerUser(name='authorized-client') if is_valid else UnauthenticatedUser()

        return ServerCallContext(
            user=user,
            state=state,
            requested_extensions=get_requested_extensions(
                request.headers.getlist(HTTP_EXTENSION_HEADER)
            ),
        )

    @staticmethod
    def _extract_bearer_token(auth_header: str) -> str | None:
        if not auth_header:
            return None

        scheme, _, token = auth_header.partition(' ')
        if scheme.lower() != 'bearer' or not token:
            return None
        return token.strip()
