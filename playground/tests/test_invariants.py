from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = REPOSITORY_ROOT / "playground"
SITE_ROOT = PLAYGROUND_ROOT / "site"
VENDOR_ROOT = SITE_ROOT / "vendor"


def test_playground_has_no_node_or_editor_toolchain() -> None:
    assert not list(PLAYGROUND_ROOT.rglob("package.json"))
    assert not list(PLAYGROUND_ROOT.rglob("*.ts"))
    assert not (VENDOR_ROOT / "codemirror").exists()


def test_playground_is_not_a_workspace_member() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as file:
        workspace = tomllib.load(file)["tool"]["uv"]["workspace"]
    assert all("playground" not in member for member in workspace.get("members", []))


def test_native_document_surfaces_are_present() -> None:
    html = (SITE_ROOT / "index.html").read_text()
    for element_id in ("model-source", "observed-data", "design-data", "truth-data"):
        assert f'id="{element_id}"' in html
    assert "codemirror" not in html.lower()


def test_application_does_not_infer_raw_ir_semantics() -> None:
    application_source = "\n".join(
        path.read_text() for path in sorted((SITE_ROOT / "src" / "app").rglob("*.mjs"))
    )
    forbidden = (
        "VectorScatterOp",
        "ScalarIndex",
        "requiredInputs",
        "designDefaults",
        "indexDataNames",
        "containsNode",
    )
    for token in forbidden:
        assert token not in application_source


def test_svg_renderer_theme_variables_are_defined() -> None:
    renderer_source = "\n".join(
        path.read_text()
        for directory in (SITE_ROOT / "src" / "dashboard", SITE_ROOT / "src" / "critique")
        for path in directory.glob("*.mjs")
    )
    stylesheet = (SITE_ROOT / "styles.css").read_text()
    referenced = set(re.findall(r"var\((--[a-z-]+)", renderer_source))
    defined = set(re.findall(r"(--[a-z-]+)\s*:", stylesheet))
    assert referenced <= defined, f"undefined SVG theme variables: {sorted(referenced - defined)}"


def test_engine_manifest_matches_wasm() -> None:
    manifest_path = VENDOR_ROOT / "bayesite" / "ENGINE.json"
    wasm_path = VENDOR_ROOT / "bayesite" / "bayesite_core.wasm"
    manifest = json.loads(manifest_path.read_text())
    assert hashlib.sha256(wasm_path.read_bytes()).hexdigest() == manifest["wasm_sha256"]


def test_pyodide_pin_is_committed() -> None:
    manifest = json.loads((VENDOR_ROOT / "pyodide" / "VENDOR.json").read_text())
    assert {"version", "url", "sha256"} <= manifest.keys()
