"""Fetch and verify the pinned Bayesite engine release for the local platform.

CI (and, later, user-facing auto-provisioning) runs this instead of building
bayesite from source. Prints only the resolved binary path to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bayescycle.backends.bayesite.provisioning import (
    PINNED_ENGINE_RELEASE,
    fetch_and_verify,
    platform_target,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=".bayesite-bin",
        help="Directory to install the Bayesite engine binary into (default: .bayesite-bin)",
    )
    args = parser.parse_args()

    binary_path = fetch_and_verify(PINNED_ENGINE_RELEASE, platform_target(), Path(args.out))
    print(binary_path)


if __name__ == "__main__":
    main()
