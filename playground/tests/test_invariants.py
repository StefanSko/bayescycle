import hashlib
import json
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = REPOSITORY_ROOT / "playground"
VENDOR_ROOT = PLAYGROUND_ROOT / "site" / "vendor"


def test_no_package_json() -> None:
    package_files = list(PLAYGROUND_ROOT.rglob("package.json"))
    assert not package_files, f"package.json is forbidden under playground/: {package_files}"


def test_playground_is_not_a_workspace_member() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as file:
        workspace = tomllib.load(file)["tool"]["uv"]["workspace"]
    members = workspace.get("members", [])
    assert all("playground" not in member for member in members), (
        f"playground must not be a uv workspace member: {members}"
    )


def test_engine_manifest_matches_wasm() -> None:
    manifest_path = VENDOR_ROOT / "bayesite" / "ENGINE.json"
    wasm_path = VENDOR_ROOT / "bayesite" / "bayesite_core.wasm"
    assert manifest_path.is_file(), f"missing engine manifest: {manifest_path}"
    assert wasm_path.is_file(), f"missing engine wasm: {wasm_path}"

    manifest = json.loads(manifest_path.read_text())
    actual_sha256 = hashlib.sha256(wasm_path.read_bytes()).hexdigest()
    assert manifest.get("wasm_sha256") == actual_sha256, (
        "ENGINE.json wasm_sha256 does not match bayesite_core.wasm: "
        f"expected {actual_sha256}, got {manifest.get('wasm_sha256')!r}"
    )


def test_pyodide_vendor_manifest() -> None:
    manifest_path = VENDOR_ROOT / "pyodide" / "VENDOR.json"
    assert manifest_path.is_file(), f"missing Pyodide vendor manifest: {manifest_path}"

    manifest = json.loads(manifest_path.read_text())
    missing_keys = {"version", "sha256"} - manifest.keys()
    assert not missing_keys, f"Pyodide VENDOR.json missing keys: {sorted(missing_keys)}"


def test_codemirror_vendor_manifest() -> None:
    manifest_path = VENDOR_ROOT / "codemirror" / "VENDOR.json"
    assert manifest_path.is_file(), f"missing CodeMirror vendor manifest: {manifest_path}"
