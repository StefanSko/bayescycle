"""Bumping PINNED_ENGINE_RELEASE via scripts/bump_engine_release.py.

Exercises the sidecar download and in-place rewrite against a real stdlib
HTTP server (no mocks) serving `.sha256` sidecar fixtures, and against a copy
of the real `provisioning.py`, so a change to that file's shape that this
script no longer understands fails the test rather than failing silently in
production.
"""

from __future__ import annotations

import functools
import http.server
import re
import shutil
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from bayescycle.backends.bayesite.provisioning import EngineRelease

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bump_engine_release import (  # noqa: E402
    TargetChecksum,
    bump_engine_release,
    fetch_all_checksums,
    fetch_checksum,
    rewrite_provisioning,
)

_REAL_PROVISIONING_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "bayescycle"
    / "backends"
    / "bayesite"
    / "provisioning.py"
)

_TARGET_ORDER: tuple[str, ...] = (
    "x86_64-unknown-linux-musl",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)

SidecarServer = tuple[str, Path]


def _make_handler_factory(
    directory: str,
) -> functools.partial[http.server.SimpleHTTPRequestHandler]:
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    return functools.partial(_Handler, directory=directory)


@pytest.fixture
def sidecar_server(tmp_path: Path) -> Iterator[SidecarServer]:
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    handler_factory = _make_handler_factory(str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", serve_dir
    finally:
        server.shutdown()
        thread.join()


def _write_sidecar(
    serve_dir: Path, tag: str, target: str, archive_format: str, sha256: str
) -> None:
    archive_name = f"bayesite-{tag}-{target}.{archive_format}"
    tag_dir = serve_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / f"{archive_name}.sha256").write_text(f"{sha256}  {archive_name}\n", encoding="utf-8")


def _fake_checksums(tag: str) -> tuple[TargetChecksum, ...]:
    formats = {"x86_64-pc-windows-msvc": "zip"}
    return tuple(
        TargetChecksum(
            target=target,
            archive_format=formats.get(target, "tar.gz"),
            sha256=f"{index:02x}" * 32,
        )
        for index, target in enumerate(_TARGET_ORDER, start=1)
    )


def test_fetch_checksum_downloads_sidecar(sidecar_server: SidecarServer) -> None:
    base_url, serve_dir = sidecar_server
    tag = "v0.3.0"
    expected = "ab" * 32
    _write_sidecar(serve_dir, tag, "x86_64-apple-darwin", "tar.gz", expected)

    checksum = fetch_checksum(base_url, tag, "x86_64-apple-darwin", "tar.gz")

    assert checksum == expected


def test_fetch_checksum_rejects_malformed_sidecar(sidecar_server: SidecarServer) -> None:
    base_url, serve_dir = sidecar_server
    tag = "v0.3.0"
    target_dir = serve_dir / tag
    target_dir.mkdir(parents=True)
    archive_name = f"bayesite-{tag}-x86_64-apple-darwin.tar.gz"
    (target_dir / f"{archive_name}.sha256").write_text("not-a-checksum\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="did not contain a sha256 checksum"):
        fetch_checksum(base_url, tag, "x86_64-apple-darwin", "tar.gz")


def test_fetch_checksum_reports_missing_sidecar(sidecar_server: SidecarServer) -> None:
    base_url, _serve_dir = sidecar_server
    with pytest.raises(RuntimeError, match="failed to download checksum sidecar"):
        fetch_checksum(base_url, "v0.3.0", "x86_64-apple-darwin", "tar.gz")


def test_fetch_all_checksums_downloads_every_pinned_target(sidecar_server: SidecarServer) -> None:
    base_url, serve_dir = sidecar_server
    tag = "v0.3.0"
    expected = _fake_checksums(tag)
    for checksum in expected:
        _write_sidecar(serve_dir, tag, checksum.target, checksum.archive_format, checksum.sha256)

    checksums = fetch_all_checksums(base_url, tag)

    assert checksums == expected


def test_rewrite_provisioning_updates_version_and_every_target_sha256() -> None:
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    tag = "v9.9.9"
    checksums = _fake_checksums(tag)

    updated = rewrite_provisioning(source, tag, checksums)

    assert 'version="v9.9.9"' in updated
    for checksum in checksums:
        assert f'sha256="{checksum.sha256}"' in updated
    # Rewriting is the only change: line count and every target/format field
    # are untouched, so a real EngineRelease still parses out of the module.
    assert updated.count("EngineTarget(") == source.count("EngineTarget(")
    for target, archive_format in zip(
        _TARGET_ORDER, (c.archive_format for c in checksums), strict=True
    ):
        assert f'target="{target}"' in updated
        assert f'archive_format="{archive_format}"' in updated


def test_rewrite_provisioning_fails_loudly_when_version_field_is_missing() -> None:
    source = 'PINNED_ENGINE_RELEASE = EngineRelease(\n    base_url="x",\n)\n'

    with pytest.raises(RuntimeError, match="PINNED_ENGINE_RELEASE shape"):
        rewrite_provisioning(source, "v9.9.9", _fake_checksums("v9.9.9"))


def test_rewrite_provisioning_fails_loudly_when_a_target_is_missing() -> None:
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    unknown_checksum = TargetChecksum(
        target="riscv64-unknown-linux-musl", archive_format="tar.gz", sha256="cd" * 32
    )

    with pytest.raises(RuntimeError, match="no EngineTarget entry"):
        rewrite_provisioning(source, "v9.9.9", (unknown_checksum,))


def _windows_target_anchor(source: str) -> str:
    """Return the windows `EngineTarget(...)` block in `source`, closing
    `),` line and trailing newline included.

    Derived from the file's current content by target name rather than a
    hardcoded checksum literal, so it keeps matching after each real engine
    bump rewrites every sha256 field in provisioning.py.
    """
    pattern = re.compile(
        r"^[ \t]*EngineTarget\(\n"
        r"(?:(?![ \t]*\),\n).*\n)*?"
        r'[ \t]*target="x86_64-pc-windows-msvc",\n'
        r"(?:(?![ \t]*\),\n).*\n)*"
        r"[ \t]*\),\n",
        re.MULTILINE,
    )
    matches = pattern.findall(source)
    assert len(matches) == 1, (
        f"expected exactly one windows EngineTarget block, found {len(matches)}"
    )
    return matches[0]


def test_rewrite_provisioning_fails_loudly_when_an_extra_target_is_present() -> None:
    """A target added to provisioning.py that this script's `_TARGETS` list (and
    therefore the fetched checksums) doesn't cover must fail loudly instead of
    rewriting only the targets it knows about -- otherwise the new target's
    stale sha256 would be recorded under the new version silently.
    """
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    anchor = _windows_target_anchor(source)
    assert source.count(anchor) == 1
    extra_entry = (
        "        EngineTarget(\n"
        '            target="aarch64-unknown-linux-musl",\n'
        '            archive_format="tar.gz",\n'
        f'            sha256="{"ef" * 32}",\n'
        "        ),\n"
    )
    source_with_extra_target = source.replace(anchor, anchor + extra_entry, 1)
    assert source_with_extra_target.count("EngineTarget(") == source.count("EngineTarget(") + 1

    with pytest.raises(RuntimeError, match="aarch64-unknown-linux-musl"):
        rewrite_provisioning(source_with_extra_target, "v9.9.9", _fake_checksums("v9.9.9"))


def test_rewrite_provisioning_fails_loudly_on_archive_format_mismatch() -> None:
    """Checksum substitution keys on the target name alone, but a checksum is a
    property of one concrete archive. If provisioning.py's EngineTarget entry
    downloads a different archive format than the one the script fetched the
    sidecar for, recording that checksum guarantees a sha256 failure at
    provision time -- fail loudly before substituting anything instead.
    """
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    windows_anchor = _windows_target_anchor(source)
    divergent_anchor = windows_anchor.replace('archive_format="zip"', 'archive_format="tar.gz"')
    assert divergent_anchor != windows_anchor
    divergent_source = source.replace(windows_anchor, divergent_anchor, 1)

    with pytest.raises(RuntimeError, match="x86_64-pc-windows-msvc") as exc_info:
        rewrite_provisioning(divergent_source, "v9.9.9", _fake_checksums("v9.9.9"))
    message = str(exc_info.value)
    assert "zip" in message
    assert "tar.gz" in message


def test_windows_anchor_derivation_survives_an_engine_bump() -> None:
    """Regression guard for the anchor used above: a real engine bump rewrites
    every sha256 in provisioning.py, so the injected-target anchor must be
    derived from the file's current content by target name, not pinned to
    today's checksum literal -- the literal approach broke on the next bump.
    """
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    bumped = rewrite_provisioning(source, "v9.9.9", _fake_checksums("v9.9.9"))
    assert bumped != source

    anchor = _windows_target_anchor(bumped)

    assert bumped.count(anchor) == 1
    assert 'target="x86_64-pc-windows-msvc"' in anchor
    windows_checksum = next(
        c.sha256 for c in _fake_checksums("v9.9.9") if c.target == "x86_64-pc-windows-msvc"
    )
    assert f'sha256="{windows_checksum}"' in anchor


def test_bump_engine_release_rewrites_a_copy_of_the_real_file_in_place(
    sidecar_server: SidecarServer, tmp_path: Path
) -> None:
    base_url, serve_dir = sidecar_server
    tag = "v9.9.9"
    expected = _fake_checksums(tag)
    for checksum in expected:
        _write_sidecar(serve_dir, tag, checksum.target, checksum.archive_format, checksum.sha256)
    provisioning_copy = tmp_path / "provisioning.py"
    shutil.copy(_REAL_PROVISIONING_PATH, provisioning_copy)

    result = bump_engine_release(tag=tag, base_url=base_url, provisioning_path=provisioning_copy)

    assert result == expected
    rewritten = provisioning_copy.read_text(encoding="utf-8")
    assert 'version="v9.9.9"' in rewritten
    for checksum in expected:
        assert f'sha256="{checksum.sha256}"' in rewritten
    namespace: dict[str, object] = {}
    exec(compile(rewritten, str(provisioning_copy), "exec"), namespace)  # noqa: S102
    release = cast(EngineRelease, namespace["PINNED_ENGINE_RELEASE"])
    assert release.version == tag
    assert tuple(t.sha256 for t in release.targets) == tuple(c.sha256 for c in expected)
