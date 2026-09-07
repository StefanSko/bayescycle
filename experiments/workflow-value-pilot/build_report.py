"""Build a portable, offline HTML report from the saved pilot evidence. No inference."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("a-numpyro", "b-bayesjax", "c-bayescycle")


def main() -> None:
    document_paths = [
        "PROTOCOL.md",
        "INITIAL-BRIEF.md",
        "REVISION-BRIEF.md",
        "HANDOFF-BRIEF.md",
        "evaluator/evaluation.json",
        "evaluator/input-manifest.json",
        "evaluator/primary-truth.json",
        "prepare.py",
        "run_agent.py",
        "evaluate.py",
    ]
    for arm in ARMS:
        document_paths.extend(
            f"{arm}/{name}"
            for name in (
                "initial-prompt.md",
                "revision-prompt.md",
                "handoff-prompt.md",
                "model.py",
                "model_revised.py",
                "analysis.py",
                "analysis_revised.py",
                "STUDY.md",
                "HANDOFF.md",
                "handoff-result.json",
                "input/recovery-truth.json",
            )
        )
        for stage in ("initial", "recovery", "revised", "revised-recovery"):
            document_paths.append(f"{arm}/results/{stage}/result.json")
        for stage in ("initial", "revision", "handoff"):
            document_paths.append(f"evaluator/{arm}/{stage}/execution.json")
    for stage in ("initial", "revised"):
        for name in ("run.json", "model.ir.json", "data.json"):
            document_paths.append(f"c-bayescycle/results/{stage}/{name}")

    documents = {}
    for name in document_paths:
        raw = (ROOT / name).read_bytes()
        documents[name] = {
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    figures = {}
    for arm in ARMS:
        for stage, prior in (("initial", "prior"), ("revised", "revised-prior")):
            trace = "cli-trace.png" if arm == "c-bayescycle" else "trace.png"
            paths = {
                "prior": f"{arm}/results/{prior}/prior_predictive.png",
                "trace": f"{arm}/results/{stage}/{trace}",
                "predictive": f"{arm}/results/{stage}/posterior_predictive.png",
            }
            for kind, name in paths.items():
                raw = (ROOT / name).read_bytes()
                figures[f"{arm}/{stage}/{kind}"] = {
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "src": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
                }

    payload = {
        "evaluation": json.loads(documents["evaluator/evaluation.json"]["text"]),
        "manifest": json.loads(documents["evaluator/input-manifest.json"]["text"]),
        "primary_truth": json.loads(documents["evaluator/primary-truth.json"]["text"]),
        "recovery_truth": json.loads(documents[f"{ARMS[0]}/input/recovery-truth.json"]["text"]),
        "documents": documents,
        "figures": figures,
    }
    # A JSON script element is still HTML raw text: escape '<' so source files
    # containing closing script tags cannot break out of the data element.
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "report.template.html").read_text()
    marker = "__PILOT_PAYLOAD__"
    if template.count(marker) != 1:
        raise ValueError("Expected exactly one data placeholder")
    output = ROOT / "report.html"
    output.write_text(template.replace(marker, encoded))
    print(f"Built {output}: {output.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"Embedded {len(documents)} documents and {len(figures)} figures; no network assets")


if __name__ == "__main__":
    main()
