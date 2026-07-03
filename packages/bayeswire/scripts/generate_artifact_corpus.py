"""Generate the golden run-artifact corpus with a real bayesite binary.

Run from the repository root:

    uv run scripts/generate_artifact_corpus.py --bayesite-bin /path/to/bayesite

For each selected corpus reference model this writes a real run directory
under ``src/bayeswire/corpus/artifacts/<name>/``:

- ``model.ir.json``   canonical bayeswire_ir bytes of the reference model
- ``data.json``       canonical wrapped data document (bayescycle.data.json.v1)
- ``dims.json``       run-directory dims sidecar, when the model declares dims
- ``posterior.ndjson``  bayesite sample output (deterministic seed)
- ``diagnostics.json``  bayesite diagnose output for that fit

plus ``manifest.json`` recording the generator settings. Downstream
exporters (bayesite_idata) consume these files as test fixtures instead of
synthesizing their own; regenerating them is a deliberate corpus change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from bayeswire.ir import canonical_bytes  # noqa: E402
from bayeswire.model import attached_model_dimensions, dimension_metadata_to_dict  # noqa: E402
from conformance.reference_models import ReferenceModelCase, reference_model_cases  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "src" / "bayeswire" / "corpus" / "artifacts"

# Two exporter fixture models: one with a dims sidecar, one without.
ARTIFACT_MODELS = ("eight_schools_non_centered", "varying_intercepts_poisson")

SEED = 20260702
CHAINS = 2
WARMUP = 300
DRAWS = 300


def _fixture_data(name: str) -> dict[str, object]:
    fixture_path = REPO_ROOT / "src" / "bayeswire" / "corpus" / "fixtures" / f"{name}.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    return fixture["data"]


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(command)}\n{result.stderr}")


def _write_artifacts(case: ReferenceModelCase, model_cls: type[object], bayesite: Path) -> None:
    run_dir = ARTIFACT_DIR / case.name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.ir.json"
    model_path.write_bytes(canonical_bytes(case.meta))

    data_path = run_dir / "data.json"
    data_path.write_text(
        json.dumps(
            {"format": "bayescycle.data.json.v1", "variables": _fixture_data(case.name)},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    dimensions = attached_model_dimensions(model_cls)
    dims_document = None if dimensions is None else dimension_metadata_to_dict(dimensions)
    if dims_document is not None and (dims_document["dims"] or dims_document["coords"]):
        (run_dir / "dims.json").write_text(
            json.dumps(
                {"dims_format": "bayescycle-dims-v1", **dims_document},
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    posterior_path = run_dir / "posterior.ndjson"
    posterior_path.unlink(missing_ok=True)
    _run(
        [
            str(bayesite),
            "sample",
            "--model",
            str(model_path),
            "--data",
            str(data_path),
            "--seed",
            str(SEED),
            "--chains",
            str(CHAINS),
            "--warmup",
            str(WARMUP),
            "--draws",
            str(DRAWS),
            "--out",
            str(posterior_path),
        ]
    )

    diagnostics_path = run_dir / "diagnostics.json"
    diagnostics_path.unlink(missing_ok=True)
    _run(
        [
            str(bayesite),
            "diagnose",
            "--fit",
            str(posterior_path),
            "--out",
            str(diagnostics_path),
        ]
    )

    header = json.loads(posterior_path.read_text(encoding="utf-8").splitlines()[0])
    manifest = {
        "model": case.name,
        "seed": SEED,
        "chains": CHAINS,
        "warmup": WARMUP,
        "draws": DRAWS,
        "sample_stats_mode": header.get("sample_stats_mode"),
        "generator": "scripts/generate_artifact_corpus.py",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mode = header.get("sample_stats_mode")
    print(f"wrote {run_dir.relative_to(REPO_ROOT)} (sample_stats_mode={mode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayesite-bin", type=Path, required=True)
    args = parser.parse_args()
    bayesite = args.bayesite_bin.resolve()
    if not bayesite.is_file():
        sys.exit(f"bayesite binary not found: {bayesite}")

    cases = {case.name: case for case in reference_model_cases()}
    for name in ARTIFACT_MODELS:
        case = cases[name]
        _write_artifacts(case, case.model_cls, bayesite)


if __name__ == "__main__":
    main()
