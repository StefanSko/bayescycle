"""Bump the pinned Bayesite engine release in provisioning.py.

Given a released bayesite tag, downloads the four `.sha256` checksum
sidecars published alongside that release's archives and replaces the whole
`PINNED_ENGINE_RELEASE = EngineRelease(...)` assignment in
`src/bayescycle/backends/bayesite/provisioning.py` with a freshly rendered
one built from the fetched data. This is the one code path that should ever
change that constant; hand-editing it risks a typo in a checksum that fails
silently until a user's download is rejected.

The rewrite works at the data level, not the string level: the existing
assignment is `ast`-parsed into its `base_url` and `(target, archive_format)`
pairs, that shape is validated against what was fetched, and only then is a
brand-new assignment rendered and spliced in over the old one's exact source
span. Nothing here regexes over checksum literals or scans unbounded source
text -- there are no anchors to go stale.

Fails loudly (raises, non-zero exit) instead of doing a partial rewrite if
`provisioning.py` no longer has the expected shape -- e.g. after a target
was added, removed, or renamed in `EngineRelease.targets`, or if the file's
targets and the fetched checksums disagree on target names or archive
formats.

Usage:
    uv run python scripts/bump_engine_release.py --tag v0.3.0
"""

from __future__ import annotations

import argparse
import ast
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_BASE_URL = "https://github.com/StefanSko/bayesite/releases/download"
_DOWNLOAD_TIMEOUT_SECONDS = 30
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROVISIONING_PATH = (
    _REPO_ROOT / "src" / "bayescycle" / "backends" / "bayesite" / "provisioning.py"
)

# Target/archive-format pairs bayesite currently releases, used to know what
# to fetch. This is the one place that needs a hand-update when a new
# platform target ships; `rewrite_provisioning` fails loudly if this list and
# provisioning.py's `EngineTarget` entries ever disagree on the *set* of
# `(target, archive_format)` pairs, so drift between "what bayesite ships"
# and "what the file declares" can't silently produce a stale or mismatched
# checksum.
_TARGETS: tuple[tuple[str, str], ...] = (
    ("x86_64-unknown-linux-musl", "tar.gz"),
    ("x86_64-apple-darwin", "tar.gz"),
    ("aarch64-apple-darwin", "tar.gz"),
    ("x86_64-pc-windows-msvc", "zip"),
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ASSIGNMENT_NAME = "PINNED_ENGINE_RELEASE"
_RELEASE_CALL_NAME = "EngineRelease"
_TARGET_CALL_NAME = "EngineTarget"


@dataclass(frozen=True)
class TargetChecksum:
    """One target's release archive checksum, fetched from its `.sha256` sidecar."""

    target: str
    archive_format: str
    sha256: str


@dataclass(frozen=True)
class _ParsedTarget:
    """One `EngineTarget(...)` entry's target/format, read out of provisioning.py."""

    target: str
    archive_format: str


def fetch_checksum(base_url: str, tag: str, target: str, archive_format: str) -> str:
    """Download one target's `.sha256` sidecar and return the checksum it names.

    Sidecars are plain `sha256sum`-style text: a hex digest, whitespace, then
    the archive filename.
    """
    archive_name = f"bayesite-{tag}-{target}.{archive_format}"
    url = f"{base_url}/{tag}/{archive_name}.sha256"
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
            body = response.read().decode("utf-8")
    except OSError as exc:
        raise RuntimeError(f"failed to download checksum sidecar: {url}\n{exc}") from exc
    fields = body.split()
    checksum = fields[0] if fields else ""
    if not _SHA256_RE.match(checksum):
        raise RuntimeError(f"sidecar did not contain a sha256 checksum: {url}\ncontents: {body!r}")
    return checksum


def fetch_all_checksums(base_url: str, tag: str) -> tuple[TargetChecksum, ...]:
    """Download the `.sha256` sidecars for every pinned target, in order."""
    return tuple(
        TargetChecksum(
            target=target,
            archive_format=archive_format,
            sha256=fetch_checksum(base_url, tag, target, archive_format),
        )
        for target, archive_format in _TARGETS
    )


def _char_offset(source: str, lineno: int, col_offset: int) -> int:
    """Convert an `ast` position (1-indexed line, UTF-8-byte column) into an
    absolute character offset into `source`.

    `ast` column offsets are UTF-8 byte offsets, not character offsets; this
    always lands on a token boundary, so decoding the truncated byte prefix
    back to `str` is safe.
    """
    lines = source.splitlines(keepends=True)
    preceding = sum(len(line) for line in lines[: lineno - 1])
    line = lines[lineno - 1] if lineno - 1 < len(lines) else ""
    within_line = len(line.encode("utf-8")[:col_offset].decode("utf-8"))
    return preceding + within_line


def _find_pinned_release_assign(source: str) -> ast.Assign:
    """Return the top-level `PINNED_ENGINE_RELEASE = ...` assignment node.

    Raises `RuntimeError` if `source` isn't valid Python or has no such
    assignment -- this is the only place the rewrite looks for the constant,
    so a renamed, removed, or malformed assignment fails loudly instead of
    silently matching nothing.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RuntimeError(f"provisioning.py is not valid Python: {exc}") from exc
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == _ASSIGNMENT_NAME
        ):
            return node
    raise RuntimeError(
        f"provisioning.py has no top-level `{_ASSIGNMENT_NAME} = ...` assignment; "
        "its shape may have changed since this script was written."
    )


def _literal_str(node: ast.expr, field: str) -> str:
    value = ast.literal_eval(node)
    if not isinstance(value, str):
        raise RuntimeError(
            f"`{field}=` is not a string literal; its shape may have changed since "
            "this script was written."
        )
    return value


def _parsed_target(node: ast.expr) -> _ParsedTarget:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _TARGET_CALL_NAME
    ):
        raise RuntimeError(
            f"a `targets=` entry is not an `{_TARGET_CALL_NAME}(...)` call; its shape "
            "may have changed since this script was written."
        )
    target: str | None = None
    archive_format: str | None = None
    for keyword in node.keywords:
        if keyword.arg == "target":
            target = _literal_str(keyword.value, "target")
        elif keyword.arg == "archive_format":
            archive_format = _literal_str(keyword.value, "archive_format")
    if target is None or archive_format is None:
        raise RuntimeError(
            f"an `{_TARGET_CALL_NAME}(...)` entry is missing `target=` or "
            "`archive_format=`; its shape may have changed since this script was written."
        )
    return _ParsedTarget(target=target, archive_format=archive_format)


def _parsed_targets(node: ast.expr) -> tuple[_ParsedTarget, ...]:
    if not isinstance(node, ast.Tuple):
        raise RuntimeError(
            f"`{_ASSIGNMENT_NAME}`'s `targets=` field is not a tuple literal; its shape "
            "may have changed since this script was written."
        )
    return tuple(_parsed_target(element) for element in node.elts)


def _parsed_release(assign: ast.Assign) -> tuple[str, tuple[_ParsedTarget, ...]]:
    """Return `(base_url, targets)` parsed out of the assignment's `EngineRelease(...)` call.

    Reads the call's keyword arguments structurally via `ast`, not by
    matching field text, so a reordered -- but still shape-correct -- file
    parses the same way.
    """
    call = assign.value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == _RELEASE_CALL_NAME
    ):
        raise RuntimeError(
            f"`{_ASSIGNMENT_NAME}` is not assigned an `{_RELEASE_CALL_NAME}(...)` call; "
            "its shape may have changed since this script was written."
        )
    base_url: str | None = None
    targets: tuple[_ParsedTarget, ...] | None = None
    for keyword in call.keywords:
        if keyword.arg == "base_url":
            base_url = _literal_str(keyword.value, "base_url")
        elif keyword.arg == "targets":
            targets = _parsed_targets(keyword.value)
    if base_url is None:
        raise RuntimeError(
            f'`{_ASSIGNMENT_NAME}` has no `base_url="..."` field; its shape may have '
            "changed since this script was written."
        )
    if targets is None:
        raise RuntimeError(
            f"`{_ASSIGNMENT_NAME}` has no `targets=(...)` field; its shape may have "
            "changed since this script was written."
        )
    return base_url, targets


def _validate_targets_match(
    parsed_targets: tuple[_ParsedTarget, ...], checksums: tuple[TargetChecksum, ...]
) -> None:
    """Fail loudly unless the file's targets and the fetched checksums cover
    exactly the same set of `(target, archive_format)` pairs.

    Comparing pairs, not just target names, catches a checksum fetched for
    one archive format being paired with an `EngineTarget` that downloads a
    different one -- recording it would guarantee a sha256 verification
    failure at provision time.
    """
    parsed_by_name = {t.target: t.archive_format for t in parsed_targets}
    fetched_by_name = {c.target: c.archive_format for c in checksums}
    parsed_names = frozenset(parsed_by_name)
    fetched_names = frozenset(fetched_by_name)
    extra = sorted(parsed_names - fetched_names)
    missing = sorted(fetched_names - parsed_names)
    mismatched = sorted(
        name
        for name in parsed_names & fetched_names
        if parsed_by_name[name] != fetched_by_name[name]
    )
    if not (extra or missing or mismatched):
        return
    problems: list[str] = []
    if extra:
        problems.append(
            f"provisioning.py has EngineTarget entries this script fetched no checksum for: {extra}"
        )
    if missing:
        problems.append(
            "this script fetched checksums for targets provisioning.py has no "
            f"EngineTarget entry for: {missing}"
        )
    if mismatched:
        details = ", ".join(
            f"{name} (file={parsed_by_name[name]!r}, fetched={fetched_by_name[name]!r})"
            for name in mismatched
        )
        problems.append(f"archive format mismatch for: {details}")
    raise RuntimeError(
        "provisioning.py's EngineTarget entries don't match the checksums this script "
        "fetched -- " + "; ".join(problems)
    )


def _render_target_block(checksum: TargetChecksum) -> str:
    return (
        "        EngineTarget(\n"
        f'            target="{checksum.target}",\n'
        f'            archive_format="{checksum.archive_format}",\n'
        f'            sha256="{checksum.sha256}",\n'
        "        ),"
    )


def _render_pinned_engine_release(
    *, version: str, base_url: str, targets: tuple[TargetChecksum, ...]
) -> str:
    """Render the full `PINNED_ENGINE_RELEASE = EngineRelease(...)` assignment.

    Matches provisioning.py's existing formatting exactly (4-space top-level
    fields, 8-space `EngineTarget(` entries, 12-space fields, trailing commas
    throughout) so `ruff format --check` sees no diff on an ordinary bump.
    """
    target_blocks = "\n".join(_render_target_block(t) for t in targets)
    return (
        f"{_ASSIGNMENT_NAME} = {_RELEASE_CALL_NAME}(\n"
        f'    version="{version}",\n'
        f'    base_url="{base_url}",\n'
        "    targets=(\n"
        f"{target_blocks}\n"
        "    ),\n"
        ")"
    )


def rewrite_provisioning(source: str, tag: str, checksums: tuple[TargetChecksum, ...]) -> str:
    """Return `source` with `PINNED_ENGINE_RELEASE` replaced by a freshly
    rendered block built from `tag` and `checksums`.

    Regenerates the whole assignment instead of patching individual fields:
    `base_url=` is carried over from the file verbatim, and `targets=`
    entries are emitted in the file's existing order (not `_TARGETS`' order),
    so an ordinary bump's diff is just the changed values. Raises
    `RuntimeError` without modifying anything if the assignment is missing or
    malformed, or if the file's targets and `checksums` don't cover exactly
    the same `(target, archive_format)` pairs.
    """
    assign = _find_pinned_release_assign(source)
    base_url, parsed_targets = _parsed_release(assign)
    _validate_targets_match(parsed_targets, checksums)

    checksum_by_target = {checksum.target: checksum for checksum in checksums}
    ordered_checksums = tuple(checksum_by_target[t.target] for t in parsed_targets)
    rendered = _render_pinned_engine_release(
        version=tag, base_url=base_url, targets=ordered_checksums
    )

    if assign.end_lineno is None or assign.end_col_offset is None:
        raise RuntimeError("could not determine the end of the PINNED_ENGINE_RELEASE assignment")
    start = _char_offset(source, assign.lineno, assign.col_offset)
    end = _char_offset(source, assign.end_lineno, assign.end_col_offset)
    return source[:start] + rendered + source[end:]


def bump_engine_release(
    *,
    tag: str,
    base_url: str = _DEFAULT_BASE_URL,
    provisioning_path: Path = _DEFAULT_PROVISIONING_PATH,
) -> tuple[TargetChecksum, ...]:
    """Fetch checksums for `tag` and rewrite `provisioning_path` in place."""
    checksums = fetch_all_checksums(base_url, tag)
    source = provisioning_path.read_text(encoding="utf-8")
    updated = rewrite_provisioning(source, tag, checksums)
    provisioning_path.write_text(updated, encoding="utf-8")
    return checksums


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="bayesite release tag, e.g. v0.3.0")
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="override the release base URL (default: the bayesite GitHub releases URL)",
    )
    parser.add_argument(
        "--provisioning-path",
        type=Path,
        default=_DEFAULT_PROVISIONING_PATH,
        help="path to provisioning.py (default: the one in this checkout)",
    )
    args = parser.parse_args(argv)

    checksums = bump_engine_release(
        tag=args.tag, base_url=args.base_url, provisioning_path=args.provisioning_path
    )
    print(f"bumped PINNED_ENGINE_RELEASE to {args.tag} in {args.provisioning_path}")
    for checksum in checksums:
        print(f"  {checksum.target}: {checksum.sha256}")


if __name__ == "__main__":
    main()
