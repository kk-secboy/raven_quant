"""Windows-side broker gateway for the QuantLab sandbox execution contract."""

from .app import create_app

__all__ = ["create_app"]
