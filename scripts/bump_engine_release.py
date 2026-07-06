"""Bump the pinned Bayesite engine release in provisioning.py.

Given a released bayesite tag, downloads the four `.sha256` checksum
sidecars published alongside that release's archives and rewrites
`PINNED_ENGINE_RELEASE` in
`src/bayescycle/backends/bayesite/provisioning.py` in place: the
`version=` field and each target's `sha256=` field. This is the one code
path that should ever change that constant; hand-editing it risks a typo
in a checksum that fails silently until a user's download is rejected.

Fails loudly (raises, non-zero exit) instead of doing a partial rewrite if
`provisioning.py` no longer has the expected shape -- e.g. after a target
was added, removed, or renamed in `EngineRelease.targets`.

Usage:
    uv run python scripts/bump_engine_release.py --tag v0.3.0
"""

from __future__ import annotations

import argparse
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

# Target/archive-format pairs, in the order they appear in
# `PINNED_ENGINE_RELEASE.targets`. Kept in lockstep with provisioning.py by
# hand; `rewrite_provisioning` fails loudly if a target here has no matching
# `EngineTarget` entry in the file.
_TARGETS: tuple[tuple[str, str], ...] = (
    ("x86_64-unknown-linux-musl", "tar.gz"),
    ("x86_64-apple-darwin", "tar.gz"),
    ("aarch64-apple-darwin", "tar.gz"),
    ("x86_64-pc-windows-msvc", "zip"),
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TargetChecksum:
    """One target's release archive checksum, fetched from its `.sha256` sidecar."""

    target: str
    archive_format: str
    sha256: str


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


_VERSION_RE = re.compile(r'(PINNED_ENGINE_RELEASE = EngineRelease\(\s*version=)"([^"]*)"')


def _target_sha256_re(target: str) -> re.Pattern[str]:
    # Non-greedy + DOTALL: matches from this target's `target="..."` field to
    # its own `sha256="..."` field, not a later target's.
    return re.compile(r'(target="' + re.escape(target) + r'".*?sha256=)"([0-9a-f]{64})"', re.DOTALL)


def rewrite_provisioning(source: str, tag: str, checksums: tuple[TargetChecksum, ...]) -> str:
    """Return `source` with `PINNED_ENGINE_RELEASE`'s version and sha256s replaced.

    Raises `RuntimeError` without modifying anything if the expected
    `version=` field or any target's `sha256=` field is missing -- that means
    provisioning.py's shape changed and this script needs updating too,
    rather than writing a corrupted pin.
    """
    if _VERSION_RE.search(source) is None:
        raise RuntimeError(
            "provisioning.py does not match the expected PINNED_ENGINE_RELEASE shape: "
            'no `version="..."` field found under `PINNED_ENGINE_RELEASE = EngineRelease(`.'
        )
    updated = _VERSION_RE.sub(rf'\g<1>"{tag}"', source, count=1)
    for checksum in checksums:
        pattern = _target_sha256_re(checksum.target)
        if pattern.search(updated) is None:
            raise RuntimeError(
                f"provisioning.py has no EngineTarget entry for target={checksum.target!r}; "
                "its shape may have changed since this script was written."
            )
        updated = pattern.sub(rf'\g<1>"{checksum.sha256}"', updated, count=1)
    return updated


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
