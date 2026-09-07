"""One bounded local-only Pi invocation; never falls back to a hosted provider."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arm", choices=("smoke", "a-numpyro", "b-bayesjax", "c-bayescycle", "a-model", "b-model")
    )
    parser.add_argument("stage", choices=("smoke", "initial", "revision", "handoff", "authoring"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    output = ROOT / "evaluator" / args.arm / args.stage
    output.mkdir(parents=True, exist_ok=False)
    prompt = ROOT / args.arm / ("prompt.md" if args.stage == "smoke" else f"{args.stage}-prompt.md")
    tools = (
        "read"
        if args.stage == "smoke"
        else "read,write,edit"
        if args.stage == "authoring"
        else "read,write,edit,bash"
    )
    command = [
        "pi",
        "--mode",
        "json",
        "-p",
        "--provider",
        "ollama",
        "--model",
        "gemma4-pi:12b",
        "--thinking",
        "medium",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-approve",
        "--tools",
        tools,
        "--session",
        str(output / "session.jsonl"),
        "@" + str(prompt),
    ]
    env = os.environ.copy()
    env.update(
        {
            "JAX_ENABLE_X64": "true",
            "MPLBACKEND": "Agg",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "OMP_NUM_THREADS": "1",
            "UV_PROJECT_ENVIRONMENT": str(ROOT.parent / "workflow-value-pilot/.venv"),
            "UV_NO_SYNC": "1",
            "PI_OFFLINE": "1",
        }
    )
    started_at = datetime.now(UTC).isoformat()
    start = time.monotonic()
    timed_out = False
    with (output / "events.jsonl").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT / args.arm,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = process.returncode
    metrics = {
        "arm": args.arm,
        "stage": args.stage,
        "started_at": started_at,
        "wall_seconds": round(time.monotonic() - start, 3),
        "returncode": returncode,
        "timed_out": timed_out,
        "provider": "ollama",
        "model": "gemma4-pi:12b",
        "thinking_requested": "medium",
        "context_window": 32768,
        "max_output_tokens": 4096,
    }
    (output / "execution.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
