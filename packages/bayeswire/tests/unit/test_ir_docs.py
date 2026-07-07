"""Tests enforcing the generated IR tag/field spec document."""

from __future__ import annotations

from pathlib import Path

from bayeswire.ir import render_ir_v1_tag_spec

# spec/ lives at the monorepo root, two levels above the bayeswire package
# root (packages/bayeswire), which is itself three levels above this file.
PACKAGE_ROOT = Path(__file__).parent.parent.parent
MONOREPO_ROOT = PACKAGE_ROOT.parent.parent
TAG_SPEC_PATH = MONOREPO_ROOT / "spec" / "ir-v1-tags.md"


def test_tag_spec_document_matches_registry() -> None:
    assert TAG_SPEC_PATH.read_text(encoding="utf-8") == render_ir_v1_tag_spec(), (
        "spec/ir-v1-tags.md is out of date. Run scripts/regenerate_corpus.py "
        "and review the diff: any tag or field change is a wire-format change."
    )
