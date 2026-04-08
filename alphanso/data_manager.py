"""
Data manager for ALPHANSO nuclear data.

Handles discovery, download, verification, and path resolution for the
nuclear data files required by ALPHANSO. When installed from PyPI, data
is downloaded on first use and cached locally.
"""

import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

DATA_VERSION = "1.1.0"

_RELEASE_URL = (
    f"https://github.com/alphanso-org/alphanso/releases/download/"
    f"data-v{DATA_VERSION}/alphanso-data-v{DATA_VERSION}.tar.gz"
)

# Filled after tarball creation (see release sequence in plan)
_EXPECTED_SHA256 = "3b4953ddd2afbd9f8a5431ad28e96e96a61ba2d03e6c7986dcb83c8d32c1b63c"

_REQUIRED_SUBDIRS = ["an_xs", "decay", "delayed_neutron", "levels", "stopping"]


def _bundled_data_dir() -> Path:
    """Return the path to data bundled inside the package (source checkout)."""
    return Path(__file__).parent / "data"


def _platform_data_dir() -> Path:
    """Return the platformdirs-based cache location for downloaded data."""
    try:
        from platformdirs import user_data_dir
    except ImportError:
        # Fallback if platformdirs is not installed
        base = Path.home() / ".local" / "share"
        return base / "alphanso" / f"data-v{DATA_VERSION}"
    return Path(user_data_dir("alphanso")) / f"data-v{DATA_VERSION}"


def _has_required_subdirs(path: Path) -> bool:
    """Check whether *path* contains the required nuclear-data subdirectories."""
    return all((path / subdir).is_dir() for subdir in _REQUIRED_SUBDIRS)


def get_data_dir() -> Path:
    """
    Return the path to the nuclear data root directory.

    Resolution order:
      1. ``ALPHANSO_DATA_DIR`` environment variable (if set and valid)
      2. Bundled ``alphanso/data/`` (source checkout with data present)
      3. platformdirs location (downloaded data cache)

    This function never triggers a download.
    """
    # 1. Environment variable override
    env = os.environ.get("ALPHANSO_DATA_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
        logger.warning(
            "ALPHANSO_DATA_DIR=%s is not a valid directory; ignoring", env
        )

    # 2. Bundled data (source checkout)
    bundled = _bundled_data_dir()
    if _has_required_subdirs(bundled):
        return bundled

    # 3. Platform cache
    return _platform_data_dir()


def is_data_available() -> bool:
    """Return ``True`` if the nuclear data is present and complete."""
    return _has_required_subdirs(get_data_dir())


def ensure_data() -> Path:
    """
    Ensure nuclear data is available, downloading if necessary.

    Returns the path to the data root directory.
    """
    data_dir = get_data_dir()
    if _has_required_subdirs(data_dir):
        return data_dir

    # Need to download
    _download_and_install()
    data_dir = _platform_data_dir()
    if not _has_required_subdirs(data_dir):
        raise RuntimeError(
            "Nuclear data download appeared to succeed but required "
            f"subdirectories are missing from {data_dir}"
        )
    return data_dir


def get_data_info() -> dict:
    """Return diagnostic information about the data installation."""
    env_var = os.environ.get("ALPHANSO_DATA_DIR")
    bundled = _bundled_data_dir()
    platform = _platform_data_dir()
    resolved = get_data_dir()

    return {
        "data_version": DATA_VERSION,
        "resolved_data_dir": str(resolved),
        "is_available": is_data_available(),
        "is_bundled": _has_required_subdirs(bundled),
        "is_downloaded": _has_required_subdirs(platform),
        "bundled_path": str(bundled),
        "platform_path": str(platform),
        "env_var": env_var,
        "required_subdirs": _REQUIRED_SUBDIRS,
    }


def _download_and_install() -> None:
    """Download, verify, and extract the nuclear data tarball."""
    dest = _platform_data_dir()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ALPHANSO nuclear data v{DATA_VERSION} ...")
    print(f"  URL: {_RELEASE_URL}")
    print(f"  Destination: {dest}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
    try:
        # Download
        try:
            response = urlopen(_RELEASE_URL)
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            block_size = 1024 * 256

            with os.fdopen(tmp_fd, "wb") as out:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(
                            f"\r  Progress: {mb:.1f}/{total_mb:.1f} MB ({pct}%)",
                            end="",
                            flush=True,
                        )
            print()  # newline after progress
        except URLError as exc:
            raise RuntimeError(
                f"Failed to download nuclear data from {_RELEASE_URL}: {exc}"
            ) from exc

        # Verify checksum
        if _EXPECTED_SHA256:
            print("  Verifying checksum ...")
            sha = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 256), b""):
                    sha.update(chunk)
            actual = sha.hexdigest()
            if actual != _EXPECTED_SHA256:
                raise RuntimeError(
                    f"SHA-256 mismatch: expected {_EXPECTED_SHA256}, got {actual}"
                )

        # Extract
        print("  Extracting ...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            # Security: prevent path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise RuntimeError(
                        f"Tarball contains unsafe path: {member.name}"
                    )
            tar.extractall(path=dest)

    finally:
        # Clean up tarball
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Write version stamp
    stamp = dest / ".alphanso-data-version"
    stamp.write_text(DATA_VERSION + "\n")

    print(f"  Nuclear data v{DATA_VERSION} installed to {dest}")
