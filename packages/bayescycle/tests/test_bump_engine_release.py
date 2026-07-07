"""Bumping PINNED_ENGINE_RELEASE via scripts/bump_engine_release.py.

Exercises the sidecar download and rewrite against a real stdlib HTTP server
(no mocks) serving `.sha256` sidecar fixtures, and against a copy of the real
`provisioning.py`. Assertions are made on PARSED data (the rewritten file's
`ast`, or the real `EngineRelease` object obtained by executing it) rather
than on string fragments, because the rewrite itself now works at the data
level: it regenerates the whole `PINNED_ENGINE_RELEASE = EngineRelease(...)`
assignment from fetched checksums instead of patching individual fields in
place, so a change to that file's shape that this script no longer
understands fails the test rather than failing silently in production.
"""

from __future__ import annotations

import ast
import functools
import http.server
import shutil
import subprocess
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


def _exec_release(source: str, path: Path) -> EngineRelease:
    """Execute a rewritten provisioning.py source and return its `EngineRelease`.

    This runs the module's own top-level code (a real, trusted source file
    from this repo, not user input), the same way Python itself would when
    the file is imported -- it is the most direct way to confirm the
    rewritten text is valid Python that actually defines the constant
    correctly, complementing the `ast`-level checks below.
    """
    namespace: dict[str, object] = {}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102
    return cast(EngineRelease, namespace["PINNED_ENGINE_RELEASE"])


def _parsed_target_pairs(source: str) -> frozenset[tuple[str, str]]:
    """Return the `(target, archive_format)` pairs in a `PINNED_ENGINE_RELEASE`
    assignment, via `ast`, without depending on any bump-script internals."""
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PINNED_ENGINE_RELEASE"
        ):
            call = node.value
            assert isinstance(call, ast.Call)
            for keyword in call.keywords:
                if keyword.arg == "targets":
                    assert isinstance(keyword.value, ast.Tuple)
                    pairs: set[tuple[str, str]] = set()
                    for element in keyword.value.elts:
                        assert isinstance(element, ast.Call)
                        fields = {kw.arg: ast.literal_eval(kw.value) for kw in element.keywords}
                        pairs.add(
                            (cast(str, fields["target"]), cast(str, fields["archive_format"]))
                        )
                    return frozenset(pairs)
    raise AssertionError("no PINNED_ENGINE_RELEASE assignment found")


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


def test_rewrite_provisioning_regenerates_version_and_every_target() -> None:
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    tag = "v9.9.9"
    checksums = _fake_checksums(tag)

    updated = rewrite_provisioning(source, tag, checksums)

    release = _exec_release(updated, _REAL_PROVISIONING_PATH)
    assert release.version == tag
    # base_url is untouched by a bump: carried over from the file verbatim.
    original_release = _exec_release(source, _REAL_PROVISIONING_PATH)
    assert release.base_url == original_release.base_url
    assert tuple((t.target, t.archive_format, t.sha256) for t in release.targets) == tuple(
        (c.target, c.archive_format, c.sha256) for c in checksums
    )
    # Target order is preserved from the file, not `_TARGETS`' order.
    assert tuple(t.target for t in release.targets) == _TARGET_ORDER


def test_rewrite_provisioning_only_touches_the_pinned_release_assignment() -> None:
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    tag = "v9.9.9"

    updated = rewrite_provisioning(source, tag, _fake_checksums(tag))

    # `_PLATFORM_TARGETS` is the next top-level definition after the
    # assignment and never changes on a bump, so everything from it onward
    # must be byte-for-byte identical in `updated`.
    anchor = "_PLATFORM_TARGETS: dict[tuple[str, str], str] = {"
    prefix = source.partition("PINNED_ENGINE_RELEASE = EngineRelease(")[0]
    source_after_anchor = source.partition(anchor)[2]
    updated_after_anchor = updated.partition(anchor)[2]
    assert updated.startswith(prefix)
    assert updated_after_anchor == source_after_anchor


def test_rewritten_file_is_valid_python_and_ruff_formatted(tmp_path: Path) -> None:
    """End-to-end sanity check: the rewritten file must parse and must
    already satisfy `ruff format --check` -- the rendered block matches
    provisioning.py's existing formatting conventions exactly."""
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    tag = "v9.9.9"

    updated = rewrite_provisioning(source, tag, _fake_checksums(tag))

    ast.parse(updated)  # raises SyntaxError if malformed
    rewritten_path = tmp_path / "provisioning.py"
    rewritten_path.write_text(updated, encoding="utf-8")
    result = subprocess.run(
        ["ruff", "format", "--check", str(rewritten_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_rewrite_provisioning_fails_loudly_when_assignment_is_missing() -> None:
    source = "SOMETHING_ELSE = 1\n"

    with pytest.raises(RuntimeError, match="no top-level"):
        rewrite_provisioning(source, "v9.9.9", _fake_checksums("v9.9.9"))


def test_rewrite_provisioning_fails_loudly_when_source_is_not_valid_python() -> None:
    source = "PINNED_ENGINE_RELEASE = EngineRelease(\n    this is not python\n"

    with pytest.raises(RuntimeError, match="not valid Python"):
        rewrite_provisioning(source, "v9.9.9", _fake_checksums("v9.9.9"))


def test_rewrite_provisioning_fails_loudly_when_base_url_field_is_missing() -> None:
    source = 'PINNED_ENGINE_RELEASE = EngineRelease(\n    version="v0.1.0",\n    targets=(),\n)\n'

    with pytest.raises(RuntimeError, match="base_url"):
        rewrite_provisioning(source, "v9.9.9", _fake_checksums("v9.9.9"))


def test_rewrite_provisioning_fails_loudly_when_a_target_is_missing_from_the_file() -> None:
    """A target this script fetched a checksum for, but that provisioning.py
    has no `EngineTarget` entry for, must fail loudly rather than silently
    dropping that target from the rewrite."""
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    checksums = _fake_checksums("v9.9.9") + (
        TargetChecksum(
            target="riscv64-unknown-linux-musl", archive_format="tar.gz", sha256="cd" * 32
        ),
    )

    with pytest.raises(RuntimeError, match="riscv64-unknown-linux-musl"):
        rewrite_provisioning(source, "v9.9.9", checksums)


def test_rewrite_provisioning_fails_loudly_when_an_extra_target_is_present_in_the_file() -> None:
    """A target added to provisioning.py that this script's `_TARGETS` list
    (and therefore the fetched checksums) doesn't cover must fail loudly
    instead of rewriting only the targets it knows about -- otherwise the new
    target's stale sha256 would be recorded under the new version silently.
    """
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    extra_entry = (
        "        EngineTarget(\n"
        '            target="aarch64-unknown-linux-musl",\n'
        '            archive_format="tar.gz",\n'
        f'            sha256="{"ef" * 32}",\n'
        "        ),\n"
    )
    # Insert right after the `targets=(` opening -- a stable structural
    # anchor (field name plus indentation), not a data literal that changes
    # on every bump.
    source_with_extra_target = source.replace("    targets=(\n", "    targets=(\n" + extra_entry, 1)
    assert source_with_extra_target != source

    with pytest.raises(RuntimeError, match="aarch64-unknown-linux-musl"):
        rewrite_provisioning(source_with_extra_target, "v9.9.9", _fake_checksums("v9.9.9"))


def test_rewrite_provisioning_fails_loudly_on_archive_format_mismatch() -> None:
    """Checksum substitution keys on the target name alone, but a checksum is
    a property of one concrete archive. If provisioning.py's `EngineTarget`
    entry downloads a different archive format than the one the script
    fetched the sidecar for, recording that checksum guarantees a sha256
    failure at provision time -- fail loudly before rewriting anything.
    """
    source = _REAL_PROVISIONING_PATH.read_text(encoding="utf-8")
    divergent_source = source.replace(
        '            target="x86_64-pc-windows-msvc",\n            archive_format="zip",',
        '            target="x86_64-pc-windows-msvc",\n            archive_format="tar.gz",',
        1,
    )
    assert divergent_source != source
    assert _parsed_target_pairs(divergent_source) != _parsed_target_pairs(source)

    with pytest.raises(RuntimeError, match="x86_64-pc-windows-msvc") as exc_info:
        rewrite_provisioning(divergent_source, "v9.9.9", _fake_checksums("v9.9.9"))
    message = str(exc_info.value)
    assert "zip" in message
    assert "tar.gz" in message


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
    ast.parse(rewritten)
    format_result = subprocess.run(
        ["ruff", "format", "--check", str(provisioning_copy)],
        capture_output=True,
        text=True,
    )
    assert format_result.returncode == 0, format_result.stdout + format_result.stderr

    release = _exec_release(rewritten, provisioning_copy)
    assert release.version == tag
    assert tuple(t.sha256 for t in release.targets) == tuple(c.sha256 for c in expected)
    assert tuple(t.target for t in release.targets) == _TARGET_ORDER
