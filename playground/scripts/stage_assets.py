from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPOSITORY_ROOT / "playground" / "site" / "vendor"
BAYESWIRE_SOURCE = REPOSITORY_ROOT / "packages" / "bayeswire" / "src" / "bayeswire"


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read vendor manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise SystemExit(f"vendor manifest must contain a JSON object: {path}")
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_bayeswire() -> int:
    target = VENDOR_ROOT / "bayeswire" / "bayeswire"
    shutil.rmtree(target.parent, ignore_errors=True)

    sources = sorted(
        (
            path
            for path in BAYESWIRE_SOURCE.rglob("*.py")
            if "corpus" not in path.relative_to(BAYESWIRE_SOURCE).parts
        ),
        key=lambda path: path.relative_to(BAYESWIRE_SOURCE).as_posix(),
    )
    manifest = []
    for source in sources:
        relative = source.relative_to(BAYESWIRE_SOURCE)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append((Path("bayeswire") / relative).as_posix())
    (target.parent / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return len(sources)


def is_forbidden_playground_file(path: Path) -> bool:
    return path.name == "package.json" or path.suffix in {".ts", ".mts", ".tsx"}


def remove_forbidden_playground_files(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and is_forbidden_playground_file(path):
            path.unlink()


def pyodide_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe path in Pyodide archive: {member.name}")
        if path.parts and path.parts[0] == "pyodide" and not is_forbidden_playground_file(path):
            members.append(member)
    if not members:
        raise SystemExit("Pyodide archive does not contain a pyodide/ directory")
    return members


def stage_pyodide() -> str:
    target = VENDOR_ROOT / "pyodide"
    remove_forbidden_playground_files(target)
    runtime = target / "pyodide.mjs"
    if runtime.is_file():
        return "skipped (pyodide.mjs already exists)"

    manifest_path = target / "VENDOR.json"
    manifest = read_manifest(manifest_path)
    url = manifest.get("url")
    expected_sha256 = manifest.get("sha256")
    version = manifest.get("version")
    if not all(isinstance(value, str) and value for value in (url, expected_sha256, version)):
        raise SystemExit(f"Pyodide manifest is missing version, sha256, or url: {manifest_path}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        archive_path = temporary / "pyodide.tar.bz2"
        try:
            with urllib.request.urlopen(url) as response, archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        except OSError as error:
            raise SystemExit(f"failed to download Pyodide {version}: {error}") from error

        actual_sha256 = sha256(archive_path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"Pyodide archive sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )

        extraction_root = temporary / "extracted"
        extraction_root.mkdir()
        try:
            with tarfile.open(archive_path, mode="r:bz2") as archive:
                members = pyodide_archive_members(archive)
                archive.extractall(extraction_root, members=members, filter="data")
        except (OSError, tarfile.TarError) as error:
            raise SystemExit(f"failed to extract Pyodide {version}: {error}") from error

        extracted = extraction_root / "pyodide"
        if not (extracted / "pyodide.mjs").is_file():
            raise SystemExit("Pyodide archive is missing pyodide/pyodide.mjs")

        target.mkdir(parents=True, exist_ok=True)
        for path in target.iterdir():
            if path.name == "VENDOR.json":
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        for source in extracted.iterdir():
            destination = target / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

    return f"staged {version}"


def verify_engine() -> str:
    engine_root = VENDOR_ROOT / "bayesite"
    manifest_path = engine_root / "ENGINE.json"
    wasm_path = engine_root / "bayesite_core.wasm"
    manifest = read_manifest(manifest_path)
    expected_sha256 = manifest.get("wasm_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise SystemExit(f"engine manifest is missing wasm_sha256: {manifest_path}")
    try:
        actual_sha256 = sha256(wasm_path)
    except OSError as error:
        raise SystemExit(f"cannot read engine wasm {wasm_path}: {error}") from error
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"engine wasm sha256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def stage_version() -> str:
    package_file = REPOSITORY_ROOT / "packages" / "bayescycle" / "pyproject.toml"
    with package_file.open("rb") as file:
        version = tomllib.load(file)["project"]["version"]
    (REPOSITORY_ROOT / "playground" / "site" / "VERSION.json").write_text(
        json.dumps({"version": version}, separators=(",", ":")) + "\n"
    )
    return str(version)


def main() -> None:
    version = stage_version()
    bayeswire_count = stage_bayeswire()
    pyodide_status = stage_pyodide()
    engine_sha256 = verify_engine()
    print(f"playground: staged lockstep version {version}")
    print(f"bayeswire: staged {bayeswire_count} Python files (corpus excluded)")
    print(f"pyodide: {pyodide_status}")
    print(f"engine: verified {engine_sha256}")


if __name__ == "__main__":
    main()
