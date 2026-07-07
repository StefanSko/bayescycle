#!/usr/bin/env python3
"""Rewrite the five lockstep package versions and their exact sibling pins.

Given ``--version X.Y.Z``, this script rewrites, across the monorepo:

- the ``[project] version`` field in each of the five package
  ``pyproject.toml`` files (bayeswire, jaxstanv5, bayescycle, bayesite-viz,
  bayesite-idata);
- the exact sibling-version pins in ``[project.dependencies]`` /
  ``[project.optional-dependencies]`` that reference another workspace
  package (bayescycle -> bayeswire, bayescycle's inproc extra ->
  jaxstanv5, jaxstanv5 -> bayeswire);
- the runtime ``__version__`` constants in the ``__init__.py`` of the
  packages listed in ``VERSION_CONSTANT_MODULES`` (bayescycle,
  bayesite-viz);
- the two moving pins in
  ``packages/bayescycle/src/bayescycle/backends/bayesite_viz/uvx_runner.py``:
  ``BAYESITE_VIZ_SOURCE``/``BAYESITE_IDATA_SOURCE`` (both rewritten to
  ``==X.Y.Z``) and ``BAYESITE_VIZ_EXCLUDE_NEWER`` (rewritten to the current
  UTC time, RFC 3339, seconds precision).

This script intentionally rewrites KNOWN, exact patterns -- it is not a
general TOML editor. Every rewrite target must already exist verbatim (up to
the version/timestamp value itself) in the file being edited; if a pattern
is not found exactly once, the script fails loudly rather than silently
skipping or duplicating a rewrite. Running it twice with the same
``--version`` is a no-op for the version/pin fields (idempotent);
``BAYESITE_VIZ_EXCLUDE_NEWER`` always moves to the new invocation's
timestamp.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# All five lockstep-versioned packages.
PACKAGES: tuple[str, ...] = (
    "bayeswire",
    "jaxstanv5",
    "bayescycle",
    "bayesite-viz",
    "bayesite-idata",
)

_VERSION_FIELD_RE = re.compile(r'^version = "[^"]*"$', re.MULTILINE)
_VERSION_CONSTANT_RE = re.compile(r'^__version__ = "[^"]*"$', re.MULTILINE)

# Packages whose top-level __init__.py exposes a runtime __version__
# constant that must track the pyproject version, as (package, module dir).
VERSION_CONSTANT_MODULES: tuple[tuple[str, str], ...] = (
    ("bayescycle", "bayescycle"),
    ("bayesite-viz", "bayesite_viz"),
)


class BumpError(RuntimeError):
    """Raised when an expected, exact rewrite pattern is not found."""


def _rewrite_exactly_once(
    text: str, pattern: re.Pattern[str], replacement: str, *, description: str
) -> str:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise BumpError(f"expected exactly one match for {description}, found {len(matches)}")
    # A function replacement (not a string) so the replacement text is never
    # interpreted for backreferences (e.g. a literal backslash in a path).
    return pattern.sub(lambda _match: replacement, text, count=1)


def _bump_pyproject_version(path: Path, version: str) -> None:
    text = path.read_text()
    new_text = _rewrite_exactly_once(
        text,
        _VERSION_FIELD_RE,
        f'version = "{version}"',
        description=f"[project] version in {path}",
    )
    path.write_text(new_text)


def _dependency_pin_pattern(name: str) -> re.Pattern[str]:
    # Matches a bare or already-pinned quoted dependency entry, e.g.
    # "bayeswire" or "bayeswire==0.2.0", but not other occurrences of the
    # name (e.g. in prose comments, which are never quoted this way).
    return re.compile(rf'"{re.escape(name)}(?:==[^"]*)?"')


def _bump_dependency_pin(path: Path, name: str, version: str) -> None:
    text = path.read_text()
    new_text = _rewrite_exactly_once(
        text,
        _dependency_pin_pattern(name),
        f'"{name}=={version}"',
        description=f'dependency pin "{name}" in {path}',
    )
    path.write_text(new_text)


def _bump_uvx_runner_pins(path: Path, version: str, exclude_newer: str) -> None:
    text = path.read_text()
    text = _rewrite_exactly_once(
        text,
        re.compile(r'BAYESITE_VIZ_SOURCE = "bayesite-viz==[^"]*"'),
        f'BAYESITE_VIZ_SOURCE = "bayesite-viz=={version}"',
        description=f"BAYESITE_VIZ_SOURCE in {path}",
    )
    text = _rewrite_exactly_once(
        text,
        re.compile(r'BAYESITE_IDATA_SOURCE = "bayesite-idata==[^"]*"'),
        f'BAYESITE_IDATA_SOURCE = "bayesite-idata=={version}"',
        description=f"BAYESITE_IDATA_SOURCE in {path}",
    )
    text = _rewrite_exactly_once(
        text,
        re.compile(r'BAYESITE_VIZ_EXCLUDE_NEWER = "[^"]*"'),
        f'BAYESITE_VIZ_EXCLUDE_NEWER = "{exclude_newer}"',
        description=f"BAYESITE_VIZ_EXCLUDE_NEWER in {path}",
    )
    path.write_text(text)


def _bump_version_constant(path: Path, version: str) -> None:
    text = path.read_text()
    new_text = _rewrite_exactly_once(
        text,
        _VERSION_CONSTANT_RE,
        f'__version__ = "{version}"',
        description=f"__version__ constant in {path}",
    )
    path.write_text(new_text)


def rfc3339_now() -> str:
    """Return the current UTC time as RFC 3339 with seconds precision."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def bump(version: str, *, repo_root: Path = REPO_ROOT, exclude_newer: str | None = None) -> None:
    """Rewrite all five package versions, their sibling pins, and the uvx pins.

    ``repo_root`` defaults to this script's own repo, and is overridable so
    tests can exercise real file rewrites against a temporary copy instead
    of the real tree. ``exclude_newer`` defaults to the current UTC time;
    callers pass an explicit value only for deterministic tests.
    """
    packages_dir = repo_root / "packages"
    uvx_runner_path = (
        packages_dir
        / "bayescycle"
        / "src"
        / "bayescycle"
        / "backends"
        / "bayesite_viz"
        / "uvx_runner.py"
    )
    stamp = exclude_newer if exclude_newer is not None else rfc3339_now()

    for package in PACKAGES:
        _bump_pyproject_version(packages_dir / package / "pyproject.toml", version)

    _bump_dependency_pin(packages_dir / "bayescycle" / "pyproject.toml", "bayeswire", version)
    _bump_dependency_pin(packages_dir / "bayescycle" / "pyproject.toml", "jaxstanv5", version)
    _bump_dependency_pin(packages_dir / "jaxstanv5" / "pyproject.toml", "bayeswire", version)

    for package, module in VERSION_CONSTANT_MODULES:
        _bump_version_constant(packages_dir / package / "src" / module / "__init__.py", version)

    _bump_uvx_runner_pins(uvx_runner_path, version, stamp)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="lockstep version, e.g. 0.3.0")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        bump(args.version)
    except BumpError as exc:
        print(f"bump_version: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
