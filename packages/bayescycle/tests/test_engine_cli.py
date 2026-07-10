"""CLI coverage for the `bayescycle engine` verb (ensure/path/info).

Uses a real stdlib HTTP server (no mocks) serving a locally built release
archive, reached only through the documented `--base-url`/`--cache-root`
seams -- no network access and no environment monkeypatching for
provisioning. `PATH` is scrubbed via `monkeypatch.setenv` only where a test
needs to guarantee no real `bayesite` binary is resolvable on this machine.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import io
import json
import sys
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from bayescycle._cli import main
from bayescycle.backends.bayesite.provisioning import PINNED_ENGINE_RELEASE, platform_target

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


def _build_release_archive(serve_dir: Path, version: str, target: str, binary_bytes: bytes) -> str:
    root = f"bayesite-{version}-{target}"
    version_dir = serve_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    archive_path = version_dir / f"{root}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        _add_bytes(tar, f"{root}/bayesite", binary_bytes, mode=0o755)
    return hashlib.sha256(archive_path.read_bytes()).hexdigest()


def test_engine_ensure_honors_base_url_and_cache_root_seams(
    archive_server: ArchiveServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`engine ensure --base-url --cache-root` reaches the local mirror.

    `--base-url` only overrides the mirror host; the pinned release still
    enforces its real, hardcoded sha256 (a locally fabricated archive cannot
    match it). That's checked here as a sha256-verification failure, which
    still proves both seams are wired: the request reached our local server
    (not the real GitHub release) and the notice names our cache_root.
    """
    base_url, serve_dir, requests = archive_server
    target = platform_target()
    _build_release_archive(
        serve_dir,
        PINNED_ENGINE_RELEASE.version,
        target,
        b"#!/bin/sh\necho ok\n",
    )
    cache_root = tmp_path / "cache"

    code = main(
        [
            "engine",
            "ensure",
            "--base-url",
            base_url,
            "--cache-root",
            str(cache_root),
        ]
    )

    assert code == 2
    assert len(requests) == 1
    err = capsys.readouterr().err
    assert "sha256" in err
    assert str(cache_root) in err


def test_engine_ensure_does_not_cache_a_verification_failure(
    archive_server: ArchiveServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_url, serve_dir, requests = archive_server
    target = platform_target()
    _build_release_archive(
        serve_dir,
        PINNED_ENGINE_RELEASE.version,
        target,
        b"#!/bin/sh\necho ok\n",
    )
    cache_root = tmp_path / "cache"
    args = ["engine", "ensure", "--base-url", base_url, "--cache-root", str(cache_root)]

    assert main(args) == 2
    capsys.readouterr()
    assert len(requests) == 1

    assert main(args) == 2
    assert len(requests) == 2


def test_engine_path_reports_workflow_error_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    cache_root = tmp_path / "empty-cache"

    code = main(["engine", "path", "--cache-root", str(cache_root)])

    assert code == 2
    err = capsys.readouterr().err
    assert "bayescycle: " in err
    assert "engine ensure" in err


def test_engine_path_prints_cached_path_without_downloading(
    archive_server: ArchiveServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A binary already cached (by whatever means) is reported, no download."""
    from bayescycle.backends.bayesite.provisioning import EngineRelease, EngineTarget, ensure_engine

    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    base_url, serve_dir, requests = archive_server
    target = platform_target()
    sha256 = _build_release_archive(
        serve_dir,
        PINNED_ENGINE_RELEASE.version,
        target,
        b"#!/bin/sh\necho ok\n",
    )
    cache_root = tmp_path / "cache"
    # Populate the cache directly through the Python API (with a release
    # whose sha256 matches our local fixture) -- this is the same cache
    # layout `bayescycle engine ensure` would produce for the real pinned
    # release, just reached without needing byte-identical mirror content.
    provisioned = ensure_engine(
        EngineRelease(
            version=PINNED_ENGINE_RELEASE.version,
            base_url=base_url,
            targets=(EngineTarget(target=target, archive_format="tar.gz", sha256=sha256),),
        ),
        cache_root=cache_root,
    )
    capsys.readouterr()
    assert len(requests) == 1

    code = main(["engine", "path", "--cache-root", str(cache_root)])

    assert code == 0
    assert capsys.readouterr().out.strip() == str(provisioned.executable)
    assert len(requests) == 1


def _write_fake_engine_with_capabilities(tmp_path: Path) -> Path:
    engine = tmp_path / "fake_bayesite.py"
    capabilities_document = json.dumps(
        {
            "capabilities_format": "bayesite.capabilities.v0",
            "commands": ["sample", "diagnose"],
            "version": "v9.9.9-fake",
            "ir": {"bayeswire_ir": 1},
            "schemas": {"data": "bayescycle.data.json.v1"},
        }
    )
    engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        "if args == ['capabilities']:\n"
        f"    print({capabilities_document!r})\n"
        "    raise SystemExit(0)\n"
        "print('usage: bayesite sample')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)
    return engine


def test_engine_info_prints_structured_capabilities_for_explicit_engine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = _write_fake_engine_with_capabilities(tmp_path)

    code = main(["engine", "info", "--engine", str(engine)])

    assert code == 0
    document = json.loads(capsys.readouterr().out)
    assert document == {
        "capabilities_format": "bayesite.capabilities.v0",
        "commands": ["sample", "diagnose"],
        "version": "v9.9.9-fake",
        "ir": {"bayeswire_ir": 1},
        "schemas": {"data": "bayescycle.data.json.v1"},
    }


def test_engine_info_reports_clear_error_for_stale_engine_without_capabilities(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = tmp_path / "stale_bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\nprint('usage: bayesite sample')\nraise SystemExit(0)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    code = main(["engine", "info", "--engine", str(engine)])

    assert code == 2
    assert "does not support structured capabilities" in capsys.readouterr().err


def test_engine_info_reports_workflow_error_when_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    cache_root = tmp_path / "empty-cache"

    code = main(["engine", "info", "--cache-root", str(cache_root)])

    assert code == 2
    assert "engine ensure" in capsys.readouterr().err
