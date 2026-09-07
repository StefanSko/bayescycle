"""Generate fixed pilot inputs once; analysis agents must not read this file."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def dataset(seed: int, beta: float) -> tuple[dict[str, list], dict[str, float | list]]:
    rng = np.random.default_rng(seed)
    clinic = np.repeat(np.arange(8), 20)
    design = np.eye(8)[clinic]
    x = rng.normal(size=clinic.size)
    x = (x - x.mean()) / x.std()
    alpha, tau, sigma = 0.4, 0.45, 0.7
    z = rng.normal(size=8)
    y = alpha + tau * (design @ z) + beta * x + rng.normal(0, sigma, x.size)
    return (
        {"x": x.tolist(), "y": y.tolist(), "clinic_design": design.tolist()},
        {"alpha": alpha, "beta": beta, "tau": tau, "sigma": sigma, "z": z.tolist()},
    )


def main() -> None:
    if (ROOT / "evaluator" / "input-manifest.json").exists():
        raise SystemExit("Fixtures already exist; refusing to overwrite pilot inputs")
    primary, truth = dataset(89131, 0.65)
    recovery, recovery_truth = dataset(89132, 0.4)
    payloads = {
        "data.json": json.dumps(primary, sort_keys=True) + "\n",
        "recovery.json": json.dumps(recovery, sort_keys=True) + "\n",
        "recovery-truth.json": json.dumps(recovery_truth, sort_keys=True) + "\n",
    }
    for arm in ("a-numpyro", "b-bayesjax", "c-bayescycle"):
        inputs = ROOT / arm / "input"
        inputs.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (inputs / name).write_text(payload)
    evaluator = ROOT / "evaluator"
    evaluator.mkdir(exist_ok=True)
    (evaluator / "primary-truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    (evaluator / "input-manifest.json").write_text(
        json.dumps(
            {
                "sha256": {
                    name: hashlib.sha256(payload.encode()).hexdigest()
                    for name, payload in payloads.items()
                },
                "versions": {
                    name: version(name)
                    for name in (
                        "jax",
                        "jaxlib",
                        "numpyro",
                        "blackjax",
                        "arviz",
                        "numpy",
                        "bayeswire",
                        "bayesjax",
                        "bayescycle",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
