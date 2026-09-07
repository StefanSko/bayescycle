"""Evaluator-owned fit of an unmodified, source-reviewed local model file.

The local agent only authored the model. This driver, not that agent, performs
binding, inference and independent numerical checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import jax.numpy as jnp
import numpy as np
from assess import PREVIOUS, ROOT, numerics

SETTINGS = {
    "num_chains": 4,
    "num_warmup": 500,
    "num_samples": 1000,
    "target_acceptance_rate": 0.9,
    "max_tree_depth": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("a-model", "b-model"))
    parser.add_argument("--source-reviewed", action="store_true", required=True)
    args = parser.parse_args()
    output = ROOT / "evaluator" / args.arm / "validation"
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / args.arm / "model.py"
    data_path = ROOT / "a-numpyro/input/data.json"
    provenance = {
        "executor": "evaluator-owned fixed driver; local agent did not run inference",
        "source_reviewed": args.source_reviewed,
        "model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": 4201,
        "settings": SETTINGS,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.monotonic()
    try:
        spec = importlib.util.spec_from_file_location("local_authored_model", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = {
            key: jnp.asarray(value, dtype=jnp.float64)
            for key, value in json.loads(data_path.read_text()).items()
        }
        if args.arm == "a-model":
            from numpyro.infer import MCMC, NUTS

            sampler = MCMC(
                NUTS(module.model, target_accept_prob=0.9, max_tree_depth=10),
                num_chains=4,
                num_warmup=500,
                num_samples=1000,
                chain_method="parallel",
                progress_bar=False,
            )
            sampler.run(jax.random.PRNGKey(4201), **data)
            samples = {
                key: np.asarray(value)
                for key, value in sampler.get_samples(group_by_chain=True).items()
            }
            diverging = np.asarray(sampler.get_extra_fields(group_by_chain=True)["diverging"])
            reference_arm = "a-numpyro"
        else:
            from bayesjax import bind_model
            from bayesjax.inference import sample

            result = sample(bind_model(module.ClinicModel, data), seed=4201, **SETTINGS)
            samples = {key: np.asarray(value) for key, value in result.samples.items()}
            diverging = np.asarray(result.diagnostics.sampling.is_divergent)
            reference_arm = "b-bayesjax"
        np.savez_compressed(output / "posterior.npz", **samples)
        np.savez_compressed(output / "sampling-stats.npz", diverging=diverging)
        metrics = numerics.numerical(output / "posterior.npz")
        metrics["divergences"] = int(diverging.sum())
        reference = json.loads((PREVIOUS / "evaluator/evaluation.json").read_text())
        beta_ref = reference["arms"][reference_arm]["initial"]["beta"]
        beta = metrics["beta"]
        tolerance = float(4 * np.hypot(beta["mcse_mean"], beta_ref["mcse_mean"]))
        delta = abs(beta["mean"] - beta_ref["mean"])
        metrics["hosted_reference_mean_check"] = {
            "absolute_difference": delta,
            "tolerance": tolerance,
            "within_tolerance": delta <= tolerance,
        }
        metrics["execution_succeeded"] = True
        metrics["wall_seconds_including_import_binding_fit_checks"] = time.monotonic() - started
    except Exception as error:
        # Preserve a failed validation; never repair or substitute the agent's model.
        metrics = {
            "execution_succeeded": False,
            "error": f"{type(error).__name__}: {error}",
            "wall_seconds": time.monotonic() - started,
        }
    (output / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
