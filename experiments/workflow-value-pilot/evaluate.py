"""Independent numerical checks and factual telemetry for this one pilot."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import arviz as az
import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ("a-numpyro", "b-bayesjax", "c-bayescycle")
PARAMS = ("alpha", "beta", "tau", "z", "sigma")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(arm: str) -> None:
    work = ROOT / arm
    target = ROOT / "evaluator" / arm
    manifest = target / "initial-files.json"
    if manifest.exists():
        raise SystemExit("Initial snapshot already exists")
    hashes = {
        str(p.relative_to(work)): sha256(p)
        for directory in (work / "results", work / "input")
        for p in directory.rglob("*")
        if p.is_file()
    }
    manifest.write_text(json.dumps(hashes, indent=2) + "\n")
    sources = target / "initial-source"
    sources.mkdir()
    for p in work.glob("*.py"):
        shutil.copyfile(p, sources / p.name)
    if (work / "STUDY.md").exists():
        shutil.copyfile(work / "STUDY.md", target / "initial-STUDY.md")


def numerical(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        samples = {name: archive[name] for name in PARAMS}
    shapes_ok = all(
        value.shape == ((4, 1000, 8) if name == "z" else (4, 1000))
        for name, value in samples.items()
    )
    finite = all(np.isfinite(value).all() for value in samples.values())
    data = az.from_dict(posterior=samples)
    beta = samples["beta"]
    mean = float(beta.mean())
    lo, hi = np.quantile(beta, [0.025, 0.975])
    metrics = {
        "mean": mean,
        "sd": float(beta.std(ddof=1)),
        "q025": float(lo),
        "q975": float(hi),
        "rhat": float(az.rhat(data, method="rank")["beta"]),
        "ess_bulk": float(az.ess(data, method="bulk")["beta"]),
        "ess_tail": float(az.ess(data, method="tail")["beta"]),
        "mcse_mean": float(az.mcse(data, method="mean")["beta"]),
    }
    maxima = az.rhat(data, method="rank").to_array().values
    bulk = az.ess(data, method="bulk").to_array().values
    tail = az.ess(data, method="tail").to_array().values
    return {
        "beta": metrics,
        "shapes_ok": shapes_ok,
        "finite": bool(finite),
        "float64": all(value.dtype == np.float64 for value in samples.values()),
        "positive_scales": bool((samples["tau"] > 0).all() and (samples["sigma"] > 0).all()),
        "max_rank_rhat": float(np.max(maxima)),
        "min_bulk_ess": float(np.min(bulk)),
        "min_tail_ess": float(np.min(tail)),
        "chain_diagnostics_pass": bool(
            np.isfinite(maxima).all()
            and np.isfinite(bulk).all()
            and np.isfinite(tail).all()
            and np.max(maxima) <= 1.01
            and np.min(bulk) >= 400
            and np.min(tail) >= 400
        ),
    }


def native_facts(directory: Path, arm: str) -> dict[str, object]:
    filename, key = {
        "a-numpyro": ("sampler_stats.npz", "diverging"),
        "b-bayesjax": ("native_diagnostics.npz", "sampling_is_divergent"),
        "c-bayescycle": ("sample_stats.npz", "diverging"),
    }[arm]
    with np.load(directory / filename, allow_pickle=False) as stats:
        count = int(stats[key].sum())
        shape_ok = stats[key].shape == (4, 1000)
    result: dict[str, object] = {"native_divergences": count, "native_stats_shape_ok": shape_ok}
    if arm == "c-bayescycle":
        rows = [
            json.loads(line) for line in (directory / "posterior.ndjson").read_text().splitlines()
        ]
        fingerprint = (
            "sha256:"
            + hashlib.sha256(
                b"bayescycle-model-data-v1\n"
                + (directory / "model.ir.json").read_bytes()
                + b"\n"
                + (directory / "data.json").read_bytes()
            ).hexdigest()
        )
        result["native_fingerprint_matches"] = (
            rows[0]["model_data_fingerprint"] == fingerprint
            and rows[-1]["trailer"]["model_data_fingerprint"] == fingerprint
        )
        with np.load(directory / "posterior.npz", allow_pickle=False) as samples:
            result["native_posterior_matches_npz"] = len(rows[1:-1]) == 4000 and all(
                np.array_equal(samples[name][row["chain"], row["draw"]], row["values"][name])
                for row in rows[1:-1]
                for name in PARAMS
            )
        result["native_divergences"] = sum(row["diverging"] for row in rows[1:-1])
    return result


def telemetry(arm: str, stage: str) -> dict[str, object]:
    directory = ROOT / "evaluator" / arm / stage
    execution = directory / "execution.json"
    result = json.loads(execution.read_text()) if execution.exists() else {"incomplete": True}
    path = directory / "events.jsonl"
    if not path.exists():
        return result
    tools = 0
    errors = []
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    estimated_cost = 0.0
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "tool_execution_start":
            tools += 1
        if event.get("type") == "tool_execution_end" and event.get("isError"):
            errors.append({"tool": event.get("toolName"), "result": event.get("result")})
        message = event.get("message", {})
        if event.get("type") == "message_end" and message.get("role") == "assistant":
            usage = message.get("usage", {})
            for key in totals:
                totals[key] += usage.get(key, 0)
            estimated_cost += usage.get("cost", {}).get("total", 0.0)
    result.update(
        {
            "tool_calls": tools,
            "flagged_tool_errors": len(errors),
            "usage": totals,
            "reported_list_price_usd": estimated_cost,
        }
    )
    (directory / "flagged-errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    return result


def evaluate() -> None:
    report: dict[str, object] = {}
    comparisons = []
    means = {}
    truth = json.loads((ROOT / "evaluator" / "primary-truth.json").read_text())["beta"]
    for arm in ARMS:
        work = ROOT / arm
        result: dict[str, object] = {}
        for stage in ("initial", "recovery", "revised", "revised-recovery"):
            path = work / "results" / stage / "posterior.npz"
            if not path.exists():
                result[stage] = {"incomplete": True}
                continue
            stats = numerical(path)
            summary = path.parent / "result.json"
            saved = json.loads(summary.read_text()) if summary.exists() else {}
            beta = stats["beta"]
            stats["saved_summary_matches"] = all(
                key in saved.get("beta", {})
                and np.isclose(value, saved["beta"][key], rtol=1e-9, atol=1e-10)
                for key, value in beta.items()
            )
            stats["reported_divergences"] = saved.get("divergences")
            stats.update(native_facts(path.parent, arm))
            stats["divergence_report_matches"] = (
                stats["reported_divergences"] == stats["native_divergences"]
            )
            if stage in ("initial", "revised"):
                stats["truth_in_95_interval_descriptive_only"] = (
                    beta["q025"] <= truth <= beta["q975"]
                )
                means[(arm, stage)] = beta
            else:
                recovery_truth = json.loads((work / "input/recovery-truth.json").read_text())[
                    "beta"
                ]
                stats["truth_in_95_interval_descriptive_only"] = (
                    beta["q025"] <= recovery_truth <= beta["q975"]
                )
            result[stage] = stats
        manifest = ROOT / "evaluator" / arm / "initial-files.json"
        if manifest.exists():
            before = json.loads(manifest.read_text())
            result["initial_files_changed_or_missing"] = [
                name
                for name, digest in before.items()
                if not (work / name).is_file() or sha256(work / name) != digest
            ]
        handoff = work / "handoff-result.json"
        if handoff.exists():
            handed = json.loads(handoff.read_text())
            current = means.get((arm, "revised"))
            result["handoff"] = {
                "reported": handed,
                "independent_beta_matches": current is not None
                and all(
                    key in handed.get("beta", {})
                    and np.isclose(value, handed["beta"][key], rtol=1e-9, atol=1e-10)
                    for key, value in current.items()
                ),
            }
        result["sessions"] = {
            stage: telemetry(arm, stage) for stage in ("initial", "revision", "handoff")
        }
        source = ROOT / "evaluator" / arm / "initial-source"
        result["initial_python_lines"] = sum(
            len(p.read_text().splitlines()) for p in source.glob("*.py")
        )
        result["initial_source_files_changed_or_missing"] = [
            p.name
            for p in source.glob("*.py")
            if not (work / p.name).is_file() or sha256(work / p.name) != sha256(p)
        ]
        report[arm] = result
    for stage in ("initial", "revised"):
        for left, right in itertools.combinations(ARMS, 2):
            if (left, stage) not in means or (right, stage) not in means:
                continue
            a, b = means[(left, stage)], means[(right, stage)]
            tolerance = 4 * math.hypot(a["mcse_mean"], b["mcse_mean"])
            delta = abs(a["mean"] - b["mean"])
            comparisons.append(
                {
                    "stage": stage,
                    "left": left,
                    "right": right,
                    "absolute_mean_difference": delta,
                    "tolerance": tolerance,
                    "within_tolerance": delta <= tolerance,
                }
            )
    output = {
        "arms": report,
        "comparisons": comparisons,
        "limitations": [
            "Single synthetic fixture and one agent per arm",
            "Human control/understanding unmeasured; full skill not executed",
            "Shared concurrent hardware/caches; not a sampler speed benchmark",
            "Flagged tool errors can miss shell commands that mask exit status",
            "LLM price is provider-reported list-price estimate, not a billed subscription cost",
        ],
    }
    (ROOT / "evaluator" / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("snapshot", "evaluate"))
    parser.add_argument("arm", nargs="?", choices=ARMS)
    args = parser.parse_args()
    if args.action == "snapshot":
        if args.arm is None:
            parser.error("snapshot requires an arm")
        snapshot(args.arm)
    else:
        evaluate()
