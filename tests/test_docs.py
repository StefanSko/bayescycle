from __future__ import annotations

from pathlib import Path

BAYESITE_VIZ_PIN = "a2809452d1c753885602fae824d789bb627d5cb6"


def test_workflow_walkthrough_uses_pinned_bayesite_viz_git_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "workflow-walkthrough.py").read_text(encoding="utf-8")
    html = (root / "docs" / "workflow-walkthrough.html").read_text(encoding="utf-8")

    for content in (text, html):
        assert "uv run --project ~/bayesite-viz" not in content
        assert "~/bayesite-viz" not in content
        assert BAYESITE_VIZ_PIN in content
        assert "uv run --with" in content
        assert "bayesite-idata" in content
        assert "bayesite-viz" in content
