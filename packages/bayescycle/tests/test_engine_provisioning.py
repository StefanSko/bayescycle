"""Provisioning of the pinned Bayesite engine release.

Exercises download, sha256 verification, extraction, and idempotent install
against a real stdlib HTTP server (no mocks) serving locally built archives.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import io
import os
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from bayescycle._errors import WorkflowError
from bayescycle.backends.bayesite.provisioning import (
    EngineRelease,
    EngineTarget,
    ProvisionedEngine,
    cache_dir,
    ensure_engine,
    fetch_and_verify,
    platform_target,
)

ArchiveServer = tuple[str, Path, list[int]]


def _make_handler_factory(
    directory: str, requests: list[int]
) -> functools.partial[http.server.SimpleHTTPRequestHandler]:
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            requests.append(1)
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    return functools.partial(_Handler, directory=directory)


@pytest.fixture
def archive_server(tmp_path: Path) -> Iterator[ArchiveServer]:
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    requests: list[int] = []
    handler_factory = _make_handler_factory(str(serve_dir), requests)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", serve_dir, requests
    finally:
        server.shutdown()
        thread.join()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = mode
    tar.addfile(info, io.BytesIO(data))


def _build_tar_gz(serve_dir: Path, version: str, target: str, binary_bytes: bytes) -> str:
    """Build a real release tar.gz under serve_dir/{version}/...; return its sha256."""
    root = f"bayesite-{version}-{target}"
    version_dir = serve_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    archive_path = version_dir / f"{root}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        _add_bytes(tar, f"{root}/bayesite", binary_bytes, mode=0o755)
        _add_bytes(tar, f"{root}/README.md", b"readme")
        _add_bytes(tar, f"{root}/LICENSE", b"license")
        _add_bytes(tar, f"{root}/NOTICE", b"notice")
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def test_fetch_and_verify_downloads_verifies_and_installs(
    archive_server: ArchiveServer, tmp_path: Path
) -> None:
    base_url, serve_dir, requests = archive_server
    version = "v0.1.0"
    target = "x86_64-unknown-linux-musl"
    binary_bytes = b"#!/bin/sh\necho ok\n"
    sha256 = _build_tar_gz(serve_dir, version, target, binary_bytes)

    release = EngineRelease(
        version=version,
        base_url=base_url,
        targets=(EngineTarget(target=target, archive_format="tar.gz", sha256=sha256),),
    )
    dest_dir = tmp_path / "install"

    binary_path = fetch_and_verify(release, target, dest_dir)

    assert binary_path.is_file()
    assert os.access(binary_path, os.X_OK)
    assert binary_path.read_bytes() == binary_bytes
    assert len(requests) == 1


def test_fetch_and_verify_rejects_sha256_mismatch(
    archive_server: ArchiveServer, tmp_path: Path
) -> None:
    base_url, serve_dir, _requests = archive_server
    version = "v0.1.0"
    target = "x86_64-unknown-linux-musl"
    _build_tar_gz(serve_dir, version, target, b"binary bytes")

    release = EngineRelease(
        version=version,
        base_url=base_url,
        targets=(EngineTarget(target=target, archive_format="tar.gz", sha256="0" * 64),),
    )
    dest_dir = tmp_path / "install"

    with pytest.raises(WorkflowError, match="sha256"):
        fetch_and_verify(release, target, dest_dir)

    expected_binary = dest_dir / f"bayesite-{version}-{target}" / "bayesite"
    assert not expected_binary.exists()


def test_fetch_and_verify_is_idempotent(archive_server: ArchiveServer, tmp_path: Path) -> None:
    base_url, serve_dir, requests = archive_server
    version = "v0.1.0"
    target = "x86_64-apple-darwin"
    binary_bytes = b"binary-payload"
    sha256 = _build_tar_gz(serve_dir, version, target, binary_bytes)

    release = EngineRelease(
        version=version,
        base_url=base_url,
        targets=(EngineTarget(target=target, archive_format="tar.gz", sha256=sha256),),
    )
    dest_dir = tmp_path / "install"

    first_path = fetch_and_verify(release, target, dest_dir)
    assert len(requests) == 1

    second_path = fetch_and_verify(release, target, dest_dir)
    assert len(requests) == 1
    assert second_path == first_path


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", "x86_64-unknown-linux-musl"),
        ("Darwin", "x86_64", "x86_64-apple-darwin"),
        ("Darwin", "arm64", "aarch64-apple-darwin"),
        ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
    ],
)
def test_platform_target_known_combinations(system: str, machine: str, expected: str) -> None:
    assert platform_target(system=system, machine=machine) == expected


def test_platform_target_rejects_unsupported_combination() -> None:
    with pytest.raises(WorkflowError, match="i386"):
        platform_target(system="Linux", machine="i386")


def test_platform_target_defaults_resolve_without_arguments() -> None:
    resolved = platform_target()
    assert resolved in {
        "x86_64-unknown-linux-musl",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
    }


def _local_release(archive_server: ArchiveServer, *, version: str = "v9.9.9") -> EngineRelease:
    base_url, serve_dir, _requests = archive_server
    target = platform_target()
    binary_bytes = f"#!/bin/sh\necho {version}\n".encode()
    sha256 = _build_tar_gz(serve_dir, version, target, binary_bytes)
    return EngineRelease(
        version=version,
        base_url=base_url,
        targets=(EngineTarget(target=target, archive_format="tar.gz", sha256=sha256),),
    )


def test_cache_dir_defaults_under_home_cache_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert cache_dir() == tmp_path / ".cache" / "bayescycle"


def test_cache_dir_honors_xdg_cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    xdg = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))

    assert cache_dir() == xdg / "bayescycle"


def test_ensure_engine_fresh_install_prints_notice_and_installs_under_cache_layout(
    archive_server: ArchiveServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release = _local_release(archive_server)
    _base_url, _serve_dir, requests = archive_server
    cache_root = tmp_path / "cache"
    target = platform_target()

    provisioned = ensure_engine(release, cache_root=cache_root)

    assert isinstance(provisioned, ProvisionedEngine)
    assert provisioned.executable.is_file()
    assert provisioned.version == release.version
    assert provisioned.sha256 == release.targets[0].sha256
    expected_prefix = cache_root / "engines" / "bayesite" / release.version / target
    assert str(provisioned.executable).startswith(str(expected_prefix))
    assert len(requests) == 1
    err = capsys.readouterr().err
    assert "provisioning bayesite" in err
    assert release.version in err


def test_ensure_engine_cached_short_circuits_without_download(
    archive_server: ArchiveServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    release = _local_release(archive_server)
    _base_url, _serve_dir, requests = archive_server
    cache_root = tmp_path / "cache"

    first = ensure_engine(release, cache_root=cache_root)
    assert len(requests) == 1
    capsys.readouterr()

    second = ensure_engine(release, cache_root=cache_root)

    assert len(requests) == 1
    assert second.executable == first.executable
    err = capsys.readouterr().err
    assert err == ""


def test_ensure_engine_force_redownloads_and_reverifies(
    archive_server: ArchiveServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    release = _local_release(archive_server)
    _base_url, _serve_dir, requests = archive_server
    cache_root = tmp_path / "cache"

    first = ensure_engine(release, cache_root=cache_root)
    assert len(requests) == 1
    capsys.readouterr()

    second = ensure_engine(release, cache_root=cache_root, force=True)

    assert len(requests) == 2
    assert second.executable == first.executable
    err = capsys.readouterr().err
    assert "provisioning bayesite" in err
