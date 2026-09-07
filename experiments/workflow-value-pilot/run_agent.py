"""Run one bounded, fresh Pi session and retain its raw event stream."""

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
    parser.add_argument("arm", choices=("a-numpyro", "b-bayesjax", "c-bayescycle"))
    parser.add_argument("stage", choices=("initial", "revision", "handoff"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    output = ROOT / "evaluator" / args.arm / args.stage
    output.mkdir(parents=True, exist_ok=False)
    prompt = ROOT / args.arm / f"{args.stage}-prompt.md"
    command = [
        "pi",
        "--mode",
        "json",
        "-p",
        "--provider",
        "openai-codex",
        "--model",
        "gpt-6-astra",
        "--thinking",
        "medium",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-approve",
        "--tools",
        "read,write,edit,bash",
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
        }
    )
    start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
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
        "provider": "openai-codex",
        "model": "gpt-6-astra",
        "thinking": "medium",
    }
    (output / "execution.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
