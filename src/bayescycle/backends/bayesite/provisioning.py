"""Provisioning for the pinned Bayesite engine release.

Downloads, verifies, and installs a released Bayesite engine binary for the
local platform. This is the single code path CI uses to fetch a pinned
release instead of building bayesite from source; later phases reuse it for
user-facing auto-provisioning.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from bayescycle._errors import WorkflowError

_DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class EngineTarget:
    """One platform build of a Bayesite engine release."""

    target: str
    archive_format: str
    sha256: str


@dataclass(frozen=True)
class EngineRelease:
    """A pinned, released Bayesite engine version across platform targets."""

    version: str
    base_url: str
    targets: tuple[EngineTarget, ...]


PINNED_ENGINE_RELEASE = EngineRelease(
    version="v0.1.0",
    base_url="https://github.com/StefanSko/bayesite/releases/download",
    targets=(
        EngineTarget(
            target="x86_64-unknown-linux-musl",
            archive_format="tar.gz",
            sha256="245d4d06559104e91de6e2880858710a0b0cfa8abccc326c6ddf9ebac8529cde",
        ),
        EngineTarget(
            target="x86_64-apple-darwin",
            archive_format="tar.gz",
            sha256="ab7dfd382f0a0543fceb908be28fdb3271640bac7333c3c72c457b334d6aac9a",
        ),
        EngineTarget(
            target="aarch64-apple-darwin",
            archive_format="tar.gz",
            sha256="8c2e5012ac3c1d055bbdc900313c05427f9aafd99412f89ee318de605e363465",
        ),
        EngineTarget(
            target="x86_64-pc-windows-msvc",
            archive_format="zip",
            sha256="9314a51dbf2def60eb124206c6a48ff7733c8ffa4629a0ebc8ce1dceabae787c",
        ),
    ),
)

_PLATFORM_TARGETS: dict[tuple[str, str], str] = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
}


def platform_target(system: str | None = None, machine: str | None = None) -> str:
    """Resolve the Bayesite release target triple for a platform.

    ``system``/``machine`` default to ``platform.system()``/``platform.machine()``;
    the parameters exist so callers (and tests) can resolve other platforms
    explicitly, without monkeypatching the ``platform`` module.
    """
    resolved_system = system if system is not None else platform.system()
    resolved_machine = machine if machine is not None else platform.machine()
    target = _PLATFORM_TARGETS.get((resolved_system, resolved_machine))
    if target is None:
        supported = ", ".join(f"{s}/{m}" for s, m in sorted(_PLATFORM_TARGETS))
        raise WorkflowError(
            "No pinned Bayesite engine release is available for this platform: "
            f"system={resolved_system!r} machine={resolved_machine!r}\n\n"
            f"Supported platforms: {supported}"
        )
    return target


def fetch_and_verify(release: EngineRelease, target: str, dest_dir: Path) -> Path:
    """Download, sha256-verify, and install the Bayesite binary for ``target``.

    Idempotent: if the binary already exists at the destination, it is
    returned without a network request. Verification failure installs
    nothing.
    """
    engine_target = _find_target(release, target)
    final_install_dir = _install_dir(dest_dir, release, target)
    binary_path = final_install_dir / _binary_name(target)
    if binary_path.is_file():
        return binary_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_url = (
        f"{release.base_url}/{release.version}/"
        f"bayesite-{release.version}-{target}.{engine_target.archive_format}"
    )
    with TemporaryDirectory(dir=dest_dir) as work_dir_name:
        work_dir = Path(work_dir_name)
        archive_path = work_dir / f"archive.{engine_target.archive_format}"
        _download(archive_url, archive_path)
        _verify_sha256(archive_path, engine_target.sha256)

        extracted_root = work_dir / "extracted"
        extracted_root.mkdir()
        _extract(archive_path, engine_target.archive_format, extracted_root)

        extracted_install_dir = extracted_root / f"bayesite-{release.version}-{target}"
        extracted_binary = extracted_install_dir / _binary_name(target)
        if not extracted_binary.is_file():
            raise WorkflowError(
                f"Bayesite archive did not contain the expected binary: {extracted_binary}"
            )
        extracted_binary.chmod(0o755)

        if final_install_dir.exists():
            shutil.rmtree(final_install_dir)
        os.replace(extracted_install_dir, final_install_dir)

    return final_install_dir / _binary_name(target)


def _find_target(release: EngineRelease, target: str) -> EngineTarget:
    for candidate in release.targets:
        if candidate.target == target:
            return candidate
    known = ", ".join(candidate.target for candidate in release.targets)
    raise WorkflowError(
        f"Bayesite release {release.version} has no build for target {target!r}\n\n"
        f"Known targets: {known}"
    )


def _binary_name(target: str) -> str:
    return "bayesite.exe" if "windows" in target else "bayesite"


def _install_dir(dest_dir: Path, release: EngineRelease, target: str) -> Path:
    return dest_dir / f"bayesite-{release.version}-{target}"


def _download(url: str, dest_path: Path) -> None:
    try:
        with (
            urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,  # noqa: S310
            dest_path.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
    except OSError as exc:
        raise WorkflowError(f"failed to download Bayesite engine archive: {url}\n{exc}") from exc


def _verify_sha256(archive_path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise WorkflowError(
            "Bayesite engine archive failed sha256 verification: "
            f"expected {expected_sha256}, got {actual}"
        )


def _extract(archive_path: Path, archive_format: str, dest_root: Path) -> None:
    if archive_format == "tar.gz":
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                _ensure_within(dest_root, member.name)
            tar.extractall(dest_root, filter="data")
    elif archive_format == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                _ensure_within(dest_root, name)
            archive.extractall(dest_root)
    else:
        raise WorkflowError(f"unsupported Bayesite archive format: {archive_format!r}")


def _ensure_within(dest_root: Path, member_name: str) -> None:
    resolved = (dest_root / member_name).resolve()
    if not resolved.is_relative_to(dest_root.resolve()):
        raise WorkflowError(f"Bayesite archive member escapes extraction root: {member_name}")
