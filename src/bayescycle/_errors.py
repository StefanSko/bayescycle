"""Shared workflow errors."""

from __future__ import annotations


class WorkflowError(RuntimeError):
    """Raised when a workflow request cannot be prepared or selected."""
