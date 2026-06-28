"""Typed artifact path wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalDataArtifact:
    """Durable canonical ``bayescycle.data.json.v1`` data artifact."""

    path: Path


@dataclass(frozen=True)
class IrArtifact:
    """Durable serialized jaxstanv5 IR artifact."""

    path: Path


@dataclass(frozen=True)
class BackendPrivateArtifact:
    """Backend-native private adapter artifact."""

    path: Path
