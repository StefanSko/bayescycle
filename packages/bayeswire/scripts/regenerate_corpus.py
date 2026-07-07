"""Regenerate the corpus IR documents, hashes, fingerprints, and tag spec.

Run from the repository root:

    uv run scripts/regenerate_corpus.py

Corpus files pin the wire format. Review every diff this produces, add a
spec changelog entry, and decide whether the change requires a format
version bump before committing it. The evaluation fixtures under
``corpus/fixtures/`` carry JAX-oracle values and are regenerated from a
jaxstanv5 checkout (see ``corpus/README.md``), not by this script; the
canonical data documents under ``corpus/data/`` and the fingerprint test
vectors in ``fingerprints.json`` are derived from those fixtures here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from bayeswire.ir import (  # noqa: E402
    canonical_bytes,
    meta_to_dict,
    model_data_fingerprint,
    render_ir_v1_tag_spec,
)
from conformance.reference_models import reference_model_cases  # noqa: E402

CORPUS_DIR = REPO_ROOT / "src" / "bayeswire" / "corpus"
TAG_SPEC_PATH = REPO_ROOT / "spec" / "ir-v1-tags.md"
DATA_DOCUMENT_FORMAT = "bayescycle.data.json.v1"


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    (CORPUS_DIR / "data").mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for case in reference_model_cases():
        document = meta_to_dict(case.meta)
        path = CORPUS_DIR / f"{case.name}.json"
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        hashes[case.name] = hashlib.sha256(canonical_bytes(case.meta)).hexdigest()
        print(f"wrote {path.relative_to(REPO_ROOT)}")

        fixture_path = CORPUS_DIR / "fixtures" / f"{case.name}.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        data_path = CORPUS_DIR / "data" / f"{case.name}.json"
        data_path.write_text(
            json.dumps(
                {"format": DATA_DOCUMENT_FORMAT, "variables": fixture["data"]},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {data_path.relative_to(REPO_ROOT)}")

        # Test vectors over the exact committed bytes of the two documents;
        # run directories fingerprint their own (canonical) model bytes the
        # same way.
        fingerprints[case.name] = model_data_fingerprint(path.read_bytes(), data_path.read_bytes())

    hashes_path = CORPUS_DIR / "hashes.json"
    hashes_path.write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {hashes_path.relative_to(REPO_ROOT)}")

    fingerprints_path = CORPUS_DIR / "fingerprints.json"
    fingerprints_path.write_text(
        json.dumps(fingerprints, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {fingerprints_path.relative_to(REPO_ROOT)}")

    TAG_SPEC_PATH.write_text(render_ir_v1_tag_spec(), encoding="utf-8")
    print(f"wrote {TAG_SPEC_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
