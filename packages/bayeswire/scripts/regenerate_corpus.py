"""Regenerate the corpus IR documents, canonical hashes, and generated tag spec.

Run from the repository root:

    uv run scripts/regenerate_corpus.py

Corpus files pin the wire format. Review every diff this produces, add a
spec changelog entry, and decide whether the change requires a format
version bump before committing it. The evaluation fixtures under
``corpus/fixtures/`` carry JAX-oracle values and are regenerated from a
jaxstanv5 checkout (see ``corpus/README.md``), not by this script.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from bayeswire.ir import canonical_bytes, meta_to_dict, render_ir_v1_tag_spec  # noqa: E402
from conformance.reference_models import reference_model_cases  # noqa: E402

CORPUS_DIR = REPO_ROOT / "corpus"
TAG_SPEC_PATH = REPO_ROOT / "spec" / "ir-v1-tags.md"


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    for case in reference_model_cases():
        document = meta_to_dict(case.meta)
        path = CORPUS_DIR / f"{case.name}.json"
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        hashes[case.name] = hashlib.sha256(canonical_bytes(case.meta)).hexdigest()
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    hashes_path = CORPUS_DIR / "hashes.json"
    hashes_path.write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {hashes_path.relative_to(REPO_ROOT)}")

    TAG_SPEC_PATH.write_text(render_ir_v1_tag_spec(), encoding="utf-8")
    print(f"wrote {TAG_SPEC_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
