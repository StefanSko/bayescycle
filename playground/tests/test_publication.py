import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "playground" / "site"


def test_version_chip_matches_lockstep_package() -> None:
    version = tomllib.loads((ROOT / "packages" / "bayescycle" / "pyproject.toml").read_text())[
        "project"
    ]["version"]
    assert json.loads((SITE / "VERSION.json").read_text()) == {"version": version}
    assert 'id="playground-version"' in (SITE / "index.html").read_text()


def test_pages_workflow_stages_pinned_assets() -> None:
    workflow = (ROOT / ".github" / "workflows" / "playground-pages.yml").read_text()
    assert "playground/scripts/stage_assets.py" in workflow
    assert "actions/deploy-pages" in workflow
    assert "playground/site" in workflow


def test_release_procedure_registers_browser_edges() -> None:
    releasing = (ROOT / "docs" / "releasing.md").read_text()
    assert "Playground" in releasing
    assert "bayesite_core.wasm" in releasing
    assert "Pyodide" in releasing


def test_application_assets_are_path_relative() -> None:
    source = (SITE / "src" / "app" / "main.mjs").read_text()
    assert 'fetch("/' not in source
    assert "new URL(" in source
