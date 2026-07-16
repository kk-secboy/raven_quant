from __future__ import annotations

import random
import threading
import time


class GlobalRateGate:
    """Thread-safe start-rate limiter shared by every download worker."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._cooldown_until = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            allowed = max(now, self._next_allowed, self._cooldown_until)
            self._next_allowed = allowed + self._interval
            delay = allowed - now
        if delay > 0:
            time.sleep(delay)

    def cooldown(self, seconds: float) -> bool:
        """Start one shared cooldown without letting other workers extend it."""

        with self._lock:
            now = time.monotonic()
            if self._cooldown_until > now:
                return False
            jitter = random.uniform(0, min(5.0, seconds * 0.05))
            self._cooldown_until = now + seconds + jitter
            return True
