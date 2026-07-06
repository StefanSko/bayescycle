"""Shared bayesite-idata exception types."""

from __future__ import annotations


class AssemblyError(ValueError):
    """Parsed artifacts cannot be assembled into a coherent fit."""
