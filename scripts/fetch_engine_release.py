"""Fetch and verify the pinned Bayesite engine release for the local platform.

CI (and bayescycle's own user-facing auto-provisioning and `bayescycle engine
ensure`) all share this one provisioning code path instead of building
bayesite from source. Prints only the resolved binary path to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bayescycle.backends.bayesite.provisioning import ensure_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=".bayesite-bin",
        help="Cache root to install the Bayesite engine binary under (default: .bayesite-bin)",
    )
    args = parser.parse_args()

    provisioned = ensure_engine(cache_root=Path(args.out))
    print(provisioned.executable)


if __name__ == "__main__":
    main()
