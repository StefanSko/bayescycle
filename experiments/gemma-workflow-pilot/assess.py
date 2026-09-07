"""Assess partial local-agent evidence without repairing it or running inference."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "workflow-value-pilot"
ARMS = ("a-numpyro", "b-bayesjax", "c-bayescycle")
STAGES = ("initial", "revision", "handoff")
FITS = ("initial", "recovery", "revised", "revised-recovery")

# Reuse the already checked numerical functions, not the hosted agents' solutions.
spec = importlib.util.spec_from_file_location("pilot_numerics", PREVIOUS / "evaluate.py")
assert spec is not None and spec.loader is not None
numerics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numerics)


def session_metrics(arm: str, stage: str) -> dict[str, object]:
    directory = ROOT / "evaluator" / arm / stage
    status = directory / "execution.json"
    result = json.loads(status.read_text()) if status.exists() else {"finished": False}
    if status.exists():
        result["finished"] = True
    events_path = directory / "events.jsonl"
    result["status"] = (
        "finished" if status.exists() else "running" if events_path.exists() else "not_run"
    )
    if not events_path.exists():
        return result
    tools = []
    errors = []
    final_text = []
    output_tokens = 0
    max_input_tokens = 0
    response_limits = 0
    compactions = 0
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A currently running stream may end in a partial JSON line.
        compactions += event.get("type") == "compaction_start"
        if event.get("type") == "tool_execution_start":
            tools.append({"name": event["toolName"], "arguments": event.get("args")})
        if event.get("type") == "tool_execution_end" and event.get("isError"):
            errors.append(
                {"kind": "tool", "tool": event.get("toolName"), "result": event.get("result")}
            )
        message = event.get("message", {})
        if event.get("type") != "message_end" or message.get("role") != "assistant":
            continue
        usage = message.get("usage", {})
        output_tokens += usage.get("output", 0)
        max_input_tokens = max(max_input_tokens, usage.get("input", 0) + usage.get("cacheRead", 0))
        response_limits += message.get("stopReason") == "length"
        if message.get("errorMessage"):
            errors.append({"kind": "provider", "message": message["errorMessage"]})
        # Deliberately omit model thinking from the evaluator's human-facing summary.
        text = "\n".join(c["text"] for c in message.get("content", []) if c.get("type") == "text")
        if text:
            final_text.append(text)
    result.update(
        {
            "tool_calls": len(tools),
            "flagged_errors": len(errors),
            "errors": errors,
            "output_tokens_reported": output_tokens,
            "max_input_tokens_reported": max_input_tokens,
            "length_terminations": response_limits,
            "compactions_observed": compactions,
            "assistant_text_messages": final_text,
        }
    )
    (directory / "tool-calls.json").write_text(json.dumps(tools, indent=2) + "\n")
    return result


def main() -> None:
    hosted = json.loads((PREVIOUS / "evaluator/evaluation.json").read_text())
    arms = {}
    for arm in ARMS:
        work = ROOT / arm
        report = {"sessions": {stage: session_metrics(arm, stage) for stage in STAGES}}
        fits = {}
        for stage in FITS:
            directory = work / "results" / stage
            posterior = directory / "posterior.npz"
            if not posterior.exists():
                fits[stage] = {"present": False}
                continue
            try:
                fit = numerics.numerical(posterior)
                fit["present"] = True
                try:
                    fit.update(numerics.native_facts(directory, arm))
                except (FileNotFoundError, KeyError, ValueError, TypeError) as error:
                    fit["native_telemetry_error"] = f"{type(error).__name__}: {error}"
                summary_path = directory / "result.json"
                if summary_path.exists():
                    saved = json.loads(summary_path.read_text())
                    fit["saved_summary_matches"] = all(
                        key in saved.get("beta", {})
                        and math.isclose(value, saved["beta"][key], rel_tol=1e-9, abs_tol=1e-10)
                        for key, value in fit["beta"].items()
                    )
                    fit["declared_sampler"] = saved.get("sampler")
                    fit["declared_divergences"] = saved.get("divergences")
                else:
                    fit["saved_summary_matches"] = False
                reference = hosted["arms"][arm][stage]["beta"]
                tolerance = 4 * math.hypot(fit["beta"]["mcse_mean"], reference["mcse_mean"])
                delta = abs(fit["beta"]["mean"] - reference["mean"])
                fit["hosted_reference_mean_check"] = {
                    "absolute_difference": delta,
                    "tolerance": tolerance,
                    "within_tolerance": delta <= tolerance,
                    "scope": "primary predeclared screen"
                    if stage in ("initial", "revised")
                    else "descriptive recovery comparison, not calibration",
                }
                fits[stage] = fit
            except (FileNotFoundError, KeyError, ValueError, TypeError, IndexError) as error:
                fits[stage] = {
                    "present": True,
                    "evaluation_error": f"{type(error).__name__}: {error}",
                }
        report["fits"] = fits
        report["study_record_present"] = (work / "STUDY.md").exists()
        report["authored_python_files"] = sorted(
            str(p.relative_to(work)) for p in work.rglob("*.py")
        )
        report["numeric_or_figure_artifacts"] = sorted(
            str(p.relative_to(work))
            for p in work.rglob("*")
            if p.suffix in (".npz", ".npy", ".ndjson", ".png", ".nc")
        )
        report["input_hashes_unchanged"] = all(
            numerics.sha256(work / "input" / name) == digest
            for name, digest in json.loads((ROOT / "evaluator/input-manifest.json").read_text())[
                "sha256"
            ].items()
        )
        arms[arm] = report
    authoring = {}
    for arm in ("a-model", "b-model"):
        source = ROOT / arm / "model.py"
        validation = ROOT / "evaluator" / arm / "validation/result.json"
        authoring[arm] = {
            "session": session_metrics(arm, "authoring"),
            "model_present": source.is_file(),
            "model_sha256": numerics.sha256(source) if source.is_file() else None,
            "evaluator_validation": json.loads(validation.read_text())
            if validation.exists()
            else None,
        }
    output = {
        "arms": arms,
        "exploratory_authoring_probe": authoring,
        "scope": (
            "Partial evidence is not success; "
            "model/settings require source review before accepting a fit."
        ),
        "supervision": (
            "Hosted assistant prepared and evaluates the experiment; "
            "no hosted code repairs supplied to local agents."
        ),
    }
    (ROOT / "evaluator/assessment.json").write_text(json.dumps(output, indent=2) + "\n")
    for arm, report in arms.items():
        initial = report["sessions"]["initial"]
        print(
            arm,
            "finished=",
            initial.get("finished"),
            "tools=",
            initial.get("tool_calls", 0),
            "main_posterior=",
            report["fits"]["initial"]["present"],
            "study_record=",
            report["study_record_present"],
        )


if __name__ == "__main__":
    main()
