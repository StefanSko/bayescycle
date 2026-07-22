"""Guard tests for the uv workspace wiring.

Reads pyproject.toml files with stdlib tomllib only (no third-party TOML or
PEP 508 parsing) and asserts the cross-package dependency shape the
monorepo migration plan requires: workspace-source sibling deps, JAX
confined to bayesjax, and bayesite-viz/bayesite-idata kept out of the
workspace member list and out of every workspace member's dependencies.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = ROOT / "packages"

# All six packages that live under packages/, whether or not they are
# workspace members.
ALL_PACKAGES = (
    "bayeswire",
    "bayesjax",
    "bayescycle",
    "bayescycle-study",
    "bayesite-viz",
    "bayesite-idata",
)

WORKSPACE_MEMBERS = (
    "bayeswire",
    "bayesjax",
    "bayescycle",
    "bayescycle-study",
)

_DEP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _load_pyproject(package: str) -> dict[str, Any]:
    path = PACKAGES_DIR / package / "pyproject.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_root_pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _dependency_name(spec: str) -> str:
    """Extract the distribution name from a PEP 508 dependency string."""
    match = _DEP_NAME_RE.match(spec.strip())
    assert match is not None, f"could not parse dependency name from {spec!r}"
    return match.group(0).lower()


def _project_dependencies(data: dict[str, Any]) -> list[str]:
    project = data.get("project", {})
    deps = project.get("dependencies", [])
    assert isinstance(deps, list)
    return deps


def _project_optional_dependencies(data: dict[str, Any]) -> dict[str, list[str]]:
    project = data.get("project", {})
    extras = project.get("optional-dependencies", {})
    assert isinstance(extras, dict)
    return extras


def _package_version(package: str) -> str:
    data = _load_pyproject(package)
    project = data.get("project", {})
    version = project.get("version")
    assert isinstance(version, str), f"{package} has no [project] version string"
    return version


def test_all_lockstep_package_versions_agree() -> None:
    versions = {package: _package_version(package) for package in ALL_PACKAGES}
    distinct = set(versions.values())
    assert len(distinct) == 1, (
        f"expected one lockstep version across all packages, got {versions!r}"
    )


def test_changelog_has_current_lockstep_version() -> None:
    version = _package_version("bayeswire")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{version}]" in changelog, (
        f"CHANGELOG.md has no section for current lockstep version {version}"
    )


_VERSION_CONSTANT_RE = re.compile(r'^__version__ = "([^"]*)"$', re.MULTILINE)

# Packages that expose a runtime __version__ constant; it must track the
# pyproject version, or `--version` output drifts from the published wheel.
VERSION_CONSTANT_FILES = {
    "bayescycle": PACKAGES_DIR / "bayescycle" / "src" / "bayescycle" / "__init__.py",
    "bayescycle-study": (
        PACKAGES_DIR / "bayescycle-study" / "src" / "bayescycle_study" / "__init__.py"
    ),
    "bayesite-viz": PACKAGES_DIR / "bayesite-viz" / "src" / "bayesite_viz" / "__init__.py",
}


def test_runtime_version_constants_match_pyproject_versions() -> None:
    for package, path in VERSION_CONSTANT_FILES.items():
        match = _VERSION_CONSTANT_RE.search(path.read_text())
        assert match is not None, f"no __version__ constant found in {path}"
        assert match.group(1) == _package_version(package), (
            f"{package}'s __version__ constant is {match.group(1)!r} but its "
            f"pyproject version is {_package_version(package)!r}"
        )


def test_bayescycle_depends_on_exact_bayeswire_pin() -> None:
    version = _package_version("bayeswire")
    data = _load_pyproject("bayescycle")
    deps = _project_dependencies(data)
    assert deps == [f"bayeswire=={version}"], (
        f"expected bayescycle's [project.dependencies] to be exactly "
        f"['bayeswire=={version}'], got {deps!r}"
    )


def test_bayescycle_inproc_extra_is_exact_bayesjax_pin() -> None:
    version = _package_version("bayesjax")
    data = _load_pyproject("bayescycle")
    extras = _project_optional_dependencies(data)
    assert extras.get("inproc") == [f"bayesjax=={version}"], (
        f"expected bayescycle's inproc optional-dependency to be exactly "
        f"['bayesjax=={version}'], got {extras.get('inproc')!r}"
    )


def test_bayesjax_depends_on_exact_bayeswire_pin() -> None:
    version = _package_version("bayeswire")
    data = _load_pyproject("bayesjax")
    deps = _project_dependencies(data)
    bayeswire_specs = [d for d in deps if _dependency_name(d) == "bayeswire"]
    assert len(bayeswire_specs) == 1, (
        f"expected exactly one bayeswire dependency in bayesjax, got {bayeswire_specs!r}"
    )
    assert bayeswire_specs[0] == f"bayeswire=={version}", (
        f"expected bayesjax's bayeswire dependency to be pinned to "
        f"'bayeswire=={version}', got {bayeswire_specs[0]!r}"
    )


def test_root_workspace_members_match_the_lightweight_packages() -> None:
    root = _load_root_pyproject()
    members = root["tool"]["uv"]["workspace"]["members"]
    expected = {f"packages/{name}" for name in WORKSPACE_MEMBERS}
    assert set(members) == expected, (
        f"expected workspace members {sorted(expected)!r}, got {sorted(members)!r}"
    )
    assert "packages/bayesite-viz" not in members
    assert "packages/bayesite-idata" not in members


def test_jax_confined_to_bayesjax() -> None:
    for package in ALL_PACKAGES:
        data = _load_pyproject(package)
        deps = _project_dependencies(data)
        names = {_dependency_name(d) for d in deps}
        if package == "bayesjax":
            continue
        assert "jax" not in names, f"{package} must not depend on jax directly"
        assert "jaxlib" not in names, f"{package} must not depend on jaxlib directly"

    # bayescycle may only reach JAX transitively via its inproc extra's
    # bayesjax dependency, never a direct jax/jaxlib dependency.
    bayescycle = _load_pyproject("bayescycle")
    extras = _project_optional_dependencies(bayescycle)
    inproc_names = {_dependency_name(d) for d in extras.get("inproc", [])}
    assert "jax" not in inproc_names
    assert "jaxlib" not in inproc_names


def test_bayesite_viz_and_idata_not_pinned_by_workspace_members() -> None:
    for package in WORKSPACE_MEMBERS:
        data = _load_pyproject(package)
        deps = _project_dependencies(data)
        extras = _project_optional_dependencies(data)
        all_specs = list(deps) + [spec for group in extras.values() for spec in group]
        names = {_dependency_name(spec) for spec in all_specs}
        assert "bayesite-viz" not in names, f"{package} must not depend on bayesite-viz"
        assert "bayesite-idata" not in names, f"{package} must not depend on bayesite-idata"
