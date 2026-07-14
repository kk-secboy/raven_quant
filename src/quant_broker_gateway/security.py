from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections.abc import Callable


class SignatureError(ValueError):
    pass


class HmacRequestVerifier:
    def __init__(
        self,
        secret: str,
        *,
        max_clock_skew_seconds: int = 30,
        nonce_claim: Callable[[str, int, int], bool] | None = None,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("HMAC secret must contain at least 32 characters")
        self.secret = secret.encode()
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.nonce_claim = nonce_claim
        self._nonces: dict[str, int] = {}
        self._lock = threading.Lock()

    def verify(
        self,
        *,
        method: str,
        path: str,
        body: bytes,
        timestamp: str,
        nonce: str,
        signature: str,
        now: int | None = None,
    ) -> None:
        current = int(time.time()) if now is None else now
        try:
            signed_at = int(timestamp)
        except ValueError as exc:
            raise SignatureError("invalid timestamp") from exc
        if abs(current - signed_at) > self.max_clock_skew_seconds:
            raise SignatureError("request timestamp is outside the allowed window")
        if not 16 <= len(nonce) <= 128 or not nonce.isascii():
            raise SignatureError("invalid nonce")
        if len(signature) != 64:
            raise SignatureError("invalid signature")
        message = b".".join(
            [timestamp.encode(), nonce.encode(), method.encode(), path.encode(), body]
        )
        expected = hmac.new(self.secret, message, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise SignatureError("invalid signature")
        with self._lock:
            self._nonces = {
                key: expires for key, expires in self._nonces.items() if expires >= current
            }
            if nonce in self._nonces:
                raise SignatureError("nonce was already used")
            expires = current + self.max_clock_skew_seconds
            if self.nonce_claim is not None and not self.nonce_claim(nonce, current, expires):
                raise SignatureError("nonce was already used")
            self._nonces[nonce] = expires
