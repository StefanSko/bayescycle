"""Copy only fixtures/prompts/environment metadata, never hosted arm solutions."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "workflow-value-pilot"
ARMS = ("a-numpyro", "b-bayesjax", "c-bayescycle")


def api(path: str, payload: dict[str, str] | None = None) -> object:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/" + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def main() -> None:
    evaluator = ROOT / "evaluator"
    evaluator.mkdir(exist_ok=False)
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(PREVIOUS / filename, ROOT / filename)
    for arm in ARMS:
        work = ROOT / arm
        work.mkdir()
        shutil.copytree(PREVIOUS / arm / "input", work / "input")
        for stage in ("initial", "revision", "handoff"):
            shutil.copyfile(PREVIOUS / arm / f"{stage}-prompt.md", work / f"{stage}-prompt.md")
    for name in ("INITIAL-BRIEF.md", "REVISION-BRIEF.md", "HANDOFF-BRIEF.md"):
        text = (PREVIOUS / name).read_text()
        text += (
            "\nLOCAL REPLICATION: Do not inspect experiments/workflow-value-pilot "
            "or any previous pilot's solutions, results, figures, reports or logs. "
            "Only its scientific environment is shared automatically through uv. "
            "Do not change the supplied UV_PROJECT_ENVIRONMENT or UV_NO_SYNC. "
            "The local PROTOCOL.md defines this run.\n"
        )
        (ROOT / name).write_text(text)
    for name in ("input-manifest.json", "primary-truth.json"):
        shutil.copyfile(PREVIOUS / "evaluator" / name, evaluator / name)
    tags = api("tags")
    show = api("show", {"model": "gemma4-pi:12b"})
    # Do not collect auth configuration or the large third-party model license.
    model_metadata = {
        key: show[key] for key in ("parameters", "details", "capabilities", "model_info")
    }
    order = random.Random(20260904).sample(list(ARMS), k=len(ARMS))
    metadata = {
        "ollama_tags": tags,
        "selected_model": "gemma4-pi:12b",
        "show": model_metadata,
        "ollama_version": subprocess.check_output(["ollama", "--version"], text=True).strip(),
        "order": order,
        "order_seed": 20260904,
        "shared_environment": str(PREVIOUS / ".venv"),
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434/v1",
        "context_window": 32768,
        "max_output_tokens": 4096,
        "pi_thinking_requested": "medium",
        "reasoning_effort_supported": False,
    }
    (evaluator / "local-environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    smoke = ROOT / "smoke"
    smoke.mkdir()
    (smoke / "probe.txt").write_text("LOCAL_TOOL_PROBE_OK\n")
    (smoke / "prompt.md").write_text(
        "Use the read tool to read probe.txt in the current directory. "
        "Then reply with its exact contents. This is a read-only integration check. "
        "Do not write any files or run commands.\n"
    )
    print("Prepared fresh inputs; no hosted solutions copied. Order:", ", ".join(order))


if __name__ == "__main__":
    main()
