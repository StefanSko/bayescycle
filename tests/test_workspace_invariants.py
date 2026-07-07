"""Guard tests for the uv workspace wiring.

Reads pyproject.toml files with stdlib tomllib only (no third-party TOML or
PEP 508 parsing) and asserts the cross-package dependency shape the
monorepo migration plan requires: workspace-source sibling deps, JAX
confined to jaxstanv5, and bayesite-viz/bayesite-idata kept out of the
workspace member list and out of every workspace member's dependencies.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = ROOT / "packages"

# All five packages that live under packages/, whether or not they are
# workspace members.
ALL_PACKAGES = (
    "bayeswire",
    "jaxstanv5",
    "bayescycle",
    "bayesite-viz",
    "bayesite-idata",
)

WORKSPACE_MEMBERS = (
    "bayeswire",
    "jaxstanv5",
    "bayescycle",
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


def test_bayescycle_depends_on_bare_bayeswire_only() -> None:
    data = _load_pyproject("bayescycle")
    deps = _project_dependencies(data)
    assert deps == ["bayeswire"], (
        f"expected bayescycle's [project.dependencies] to be exactly "
        f"['bayeswire'] with no URL/git specifier, got {deps!r}"
    )


def test_bayescycle_inproc_extra_is_bare_jaxstanv5_only() -> None:
    data = _load_pyproject("bayescycle")
    extras = _project_optional_dependencies(data)
    assert extras.get("inproc") == ["jaxstanv5"], (
        f"expected bayescycle's inproc optional-dependency to be exactly "
        f"['jaxstanv5'] with no URL specifier, got {extras.get('inproc')!r}"
    )


def test_jaxstanv5_depends_on_bare_bayeswire() -> None:
    data = _load_pyproject("jaxstanv5")
    deps = _project_dependencies(data)
    bayeswire_specs = [d for d in deps if _dependency_name(d) == "bayeswire"]
    assert len(bayeswire_specs) == 1, (
        f"expected exactly one bayeswire dependency in jaxstanv5, got {bayeswire_specs!r}"
    )
    assert bayeswire_specs[0] == "bayeswire", (
        f"expected jaxstanv5's bayeswire dependency to be a bare name with no "
        f"URL/git specifier, got {bayeswire_specs[0]!r}"
    )


def test_root_workspace_members_are_exactly_three() -> None:
    root = _load_root_pyproject()
    members = root["tool"]["uv"]["workspace"]["members"]
    expected = {f"packages/{name}" for name in WORKSPACE_MEMBERS}
    assert set(members) == expected, (
        f"expected workspace members {sorted(expected)!r}, got {sorted(members)!r}"
    )
    assert "packages/bayesite-viz" not in members
    assert "packages/bayesite-idata" not in members


def test_jax_confined_to_jaxstanv5() -> None:
    for package in ALL_PACKAGES:
        data = _load_pyproject(package)
        deps = _project_dependencies(data)
        names = {_dependency_name(d) for d in deps}
        if package == "jaxstanv5":
            continue
        assert "jax" not in names, f"{package} must not depend on jax directly"
        assert "jaxlib" not in names, f"{package} must not depend on jaxlib directly"

    # bayescycle may only reach JAX transitively via its inproc extra's
    # jaxstanv5 dependency, never a direct jax/jaxlib dependency.
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
