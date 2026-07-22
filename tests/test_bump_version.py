"""Behavioral tests for scripts/bump_version.py.

These exercise the real script against a temporary copy of a minimal
package layout (real file I/O, no mocking) so the rewrite regexes are
proven against fixture text shaped exactly like the real pyproject.toml /
uvx_runner.py files, without mutating the repo's own files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bump_version import BumpError, bump  # noqa: E402

_BAYESWIRE_PYPROJECT = """\
[project]
name = "bayeswire"
version = "0.2.0"
dependencies = []
"""

_BAYESJAX_PYPROJECT = """\
[project]
name = "bayesjax"
version = "0.1.0"
dependencies = [
    "bayeswire",
    "jax>=0.6.0",
    "blackjax>=1.2.0",
]
"""

_BAYESCYCLE_PYPROJECT = """\
[project]
name = "bayescycle"
version = "0.1.0"
dependencies = [
    "bayeswire",
]

[project.optional-dependencies]
inproc = [
    "bayesjax",
]
"""

_BAYESCYCLE_STUDY_PYPROJECT = """\
[project]
name = "bayescycle-study"
version = "0.1.0"
"""

_BAYESITE_VIZ_PYPROJECT = """\
[project]
name = "bayesite-viz"
version = "0.1.0"
"""

_BAYESITE_IDATA_PYPROJECT = """\
[project]
name = "bayesite-idata"
version = "0.1.0"
"""

_UVX_RUNNER = """\
BAYESITE_VIZ_SOURCE = "bayesite-viz==0.3.0"
BAYESITE_IDATA_SOURCE = "bayesite-idata==0.3.0"
BAYESITE_VIZ_EXCLUDE_NEWER = "2026-07-07T00:00:00Z"
FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION = "2100-01-01T00:00:00Z"
"""


def _write_fixture_repo(root: Path) -> None:
    layout = {
        "bayeswire": _BAYESWIRE_PYPROJECT,
        "bayesjax": _BAYESJAX_PYPROJECT,
        "bayescycle": _BAYESCYCLE_PYPROJECT,
        "bayescycle-study": _BAYESCYCLE_STUDY_PYPROJECT,
        "bayesite-viz": _BAYESITE_VIZ_PYPROJECT,
        "bayesite-idata": _BAYESITE_IDATA_PYPROJECT,
    }
    for package, content in layout.items():
        package_dir = root / "packages" / package
        package_dir.mkdir(parents=True)
        (package_dir / "pyproject.toml").write_text(content)

    uvx_runner_dir = (
        root / "packages" / "bayescycle" / "src" / "bayescycle" / "backends" / "bayesite_viz"
    )
    uvx_runner_dir.mkdir(parents=True)
    (uvx_runner_dir / "uvx_runner.py").write_text(_UVX_RUNNER)

    version_constant = '__version__ = "0.1.0"\n'
    (root / "packages" / "bayescycle" / "src" / "bayescycle" / "__init__.py").write_text(
        version_constant
    )
    study_src_dir = root / "packages" / "bayescycle-study" / "src" / "bayescycle_study"
    study_src_dir.mkdir(parents=True)
    (study_src_dir / "__init__.py").write_text(version_constant)
    viz_src_dir = root / "packages" / "bayesite-viz" / "src" / "bayesite_viz"
    viz_src_dir.mkdir(parents=True)
    (viz_src_dir / "__init__.py").write_text(version_constant)


def _read(root: Path, *parts: str) -> str:
    return (root / "packages" / Path(*parts)).read_text()


def test_bump_rewrites_all_versions_and_sibling_pins(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)

    bump("0.3.0", repo_root=tmp_path, exclude_newer="2030-05-01T12:00:00Z")

    for package in (
        "bayeswire",
        "bayesjax",
        "bayescycle",
        "bayescycle-study",
        "bayesite-viz",
        "bayesite-idata",
    ):
        assert 'version = "0.3.0"' in _read(tmp_path, package, "pyproject.toml")

    bayescycle_text = _read(tmp_path, "bayescycle", "pyproject.toml")
    assert '"bayeswire==0.3.0"' in bayescycle_text
    assert '"bayesjax==0.3.0"' in bayescycle_text

    bayesjax_text = _read(tmp_path, "bayesjax", "pyproject.toml")
    assert '"bayeswire==0.3.0"' in bayesjax_text
    assert '"jax>=0.6.0"' in bayesjax_text
    assert '"blackjax>=1.2.0"' in bayesjax_text

    uvx_runner_text = _read(
        tmp_path, "bayescycle", "src", "bayescycle", "backends", "bayesite_viz", "uvx_runner.py"
    )
    assert 'BAYESITE_VIZ_SOURCE = "bayesite-viz==0.3.0"' in uvx_runner_text
    assert 'BAYESITE_IDATA_SOURCE = "bayesite-idata==0.3.0"' in uvx_runner_text
    assert 'BAYESITE_VIZ_EXCLUDE_NEWER = "2030-05-01T12:00:00Z"' in uvx_runner_text
    # Not one of the moving pins.
    assert 'FIRST_PARTY_EXCLUDE_NEWER_EXEMPTION = "2100-01-01T00:00:00Z"' in uvx_runner_text

    for init_path in (
        ("bayescycle", "src", "bayescycle", "__init__.py"),
        ("bayescycle-study", "src", "bayescycle_study", "__init__.py"),
        ("bayesite-viz", "src", "bayesite_viz", "__init__.py"),
    ):
        assert '__version__ = "0.3.0"' in _read(tmp_path, *init_path)


def test_bump_is_idempotent_for_the_same_version(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)

    bump("0.3.0", repo_root=tmp_path, exclude_newer="2030-05-01T12:00:00Z")
    first_pass = _read(tmp_path, "bayescycle", "pyproject.toml")

    bump("0.3.0", repo_root=tmp_path, exclude_newer="2030-05-01T12:00:00Z")
    second_pass = _read(tmp_path, "bayescycle", "pyproject.toml")

    assert first_pass == second_pass


def test_bump_moves_an_existing_pin_to_a_new_version(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)

    bump("0.3.0", repo_root=tmp_path, exclude_newer="2030-05-01T12:00:00Z")
    bump("0.4.0", repo_root=tmp_path, exclude_newer="2031-06-02T13:00:00Z")

    bayescycle_text = _read(tmp_path, "bayescycle", "pyproject.toml")
    assert '"bayeswire==0.4.0"' in bayescycle_text
    assert "0.3.0" not in bayescycle_text


def test_bump_fails_loudly_when_a_pattern_is_missing(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    # Corrupt the expected pattern: no quoted version field left to find.
    (tmp_path / "packages" / "bayeswire" / "pyproject.toml").write_text(
        '[project]\nname = "bayeswire"\n'
    )

    with pytest.raises(BumpError, match="expected exactly one match"):
        bump("0.3.0", repo_root=tmp_path, exclude_newer="2030-05-01T12:00:00Z")


def test_bump_fails_loudly_on_an_ambiguous_pattern(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    # Corrupt the expected pattern: two version fields now match.
    text = _read(tmp_path, "bayeswire", "pyproject.toml")
    (tmp_path / "packages" / "bayeswire" / "pyproject.toml").write_text(
        text + '\nversion = "9.9.9"\n'
    )

    with pytest.raises(BumpError, match="expected exactly one match"):
        bump("0.3.0", repo_root=tmp_path, exclude_newer="2030-05-01T12:00:00Z")
