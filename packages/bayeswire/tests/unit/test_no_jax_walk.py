"""The no-JAX walk: every bayeswire module imports with backends blocked.

bayeswire is stdlib-only by identity. A JAX (or BlackJAX) import anywhere in
the package is a bug by definition; this walk covers every module so a new
module cannot dodge the boundary.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "src" / "bayeswire"


_BLOCK_BACKEND_IMPORTS = """
import importlib.abc
import sys

class BlockBackendImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ImportError(f"blocked backend import: {fullname}")
        if fullname == "blackjax" or fullname.startswith("blackjax."):
            raise ImportError(f"blocked backend import: {fullname}")
        return None

sys.meta_path.insert(0, BlockBackendImports())
"""


def _every_module_name() -> tuple[str, ...]:
    modules: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT.parent)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return tuple(modules)


def test_module_walk_discovers_the_whole_package() -> None:
    modules = _every_module_name()

    assert "bayeswire" in modules
    assert "bayeswire.ir" in modules
    assert "bayeswire.model.decorator" in modules
    assert len(modules) >= 20


def test_every_module_imports_without_jax_or_blackjax() -> None:
    imports = "\n".join(f"import {name}" for name in _every_module_name())
    checks = textwrap.dedent(
        """
        forbidden = [
            name for name in sys.modules
            if name == "jax" or name.startswith("jax.")
            or name == "blackjax" or name.startswith("blackjax.")
        ]
        assert forbidden == [], forbidden
        site_injected = {"bayeswire", "sitecustomize", "usercustomize"}
        non_stdlib = [
            name for name in sys.modules
            if "." not in name
            and name not in sys.stdlib_module_names
            and not name.startswith("_")
            and name not in site_injected
        ]
        assert non_stdlib == [], non_stdlib
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_BACKEND_IMPORTS + imports + "\n" + checks],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_package_declares_no_runtime_dependencies() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert tuple(pyproject["project"].get("dependencies", ())) == ()
    assert "optional-dependencies" not in pyproject["project"]
