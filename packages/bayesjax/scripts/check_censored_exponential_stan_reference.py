#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cmdstanpy>=1.3.0",
#   "bayesjax",
# ]
#
# [tool.uv.sources]
# bayesjax = { path = "..", editable = true }
# ///
"""Compare censored Exponential imputation against an equivalent Stan model."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

if TYPE_CHECKING:
    from bayesjax.inference import NutsDiagnosticTrace, SamplerResult
    from bayesjax.model.bound import BoundModel


type FloatVector = tuple[float, ...]
type IntVector = tuple[int, ...]


@dataclass(frozen=True)
class CensoredExponentialConfig:
    """Configuration for the censored Exponential Stan comparison."""

    seed: int
    num_chains: int
    num_warmup: int
    num_samples: int
    max_k: float
    max_rhat: float
    min_ess: float
    target_acceptance_rate: float
    stan_model: Path
    stan_data: Path


@dataclass(frozen=True)
class CensoredExponentialResult:
    """Comparison result for the Exponential rate parameter."""

    parameter: str
    signed_z: float
    k_min: float
    bayesjax_mean: float
    stan_mean: float
    combined_mcse: float
    bayesjax_rhat: float
    stan_rhat: float
    bayesjax_ess: float
    stan_ess: float
    bayesjax_sampling_divergences: int
    stan_sampling_divergences: int


class StanDiagnosticTrace(NamedTuple):
    """Selected Stan sampler diagnostics."""

    is_divergent: jax.Array
    acceptance_rate: jax.Array


class StanDrawResult(NamedTuple):
    """Stan scalar draws and sampler diagnostics."""

    samples: Mapping[str, jax.Array]
    diagnostics: StanDiagnosticTrace


class StanFit(Protocol):
    """Minimal CmdStanMCMC protocol needed by this script."""

    @property
    def column_names(self) -> Sequence[str]:
        """Draw column names."""
        ...

    def draws(self, *, inc_warmup: bool, concat_chains: bool) -> object:
        """Return posterior draws."""
        ...


class StanPosteriorModel(Protocol):
    """Minimal CmdStanModel protocol needed for sampling."""

    def sample(
        self,
        *,
        data: str,
        seed: int,
        chains: int,
        parallel_chains: int,
        iter_warmup: int,
        iter_sampling: int,
        show_progress: bool,
        output_dir: str,
        adapt_delta: float,
    ) -> StanFit:
        """Run Stan NUTS."""
        ...


class StanModelFactory(Protocol):
    """Callable constructor protocol for CmdStanModel."""

    def __call__(self, *, stan_file: str) -> StanPosteriorModel:
        """Construct a Stan model."""
        ...


class CmdStanPyModule(Protocol):
    """Minimal cmdstanpy module protocol."""

    @property
    def CmdStanModel(self) -> StanModelFactory:
        """CmdStanModel constructor."""
        ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))


def _load_json(path: Path) -> Mapping[str, object]:
    return cast(Mapping[str, object], json.loads(path.read_text()))


def _as_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{name} must be numeric")


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    raise ValueError(f"{name} must be an integer")


def _float_sequence(value: object, *, name: str) -> FloatVector:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(_as_float(item, name=f"{name}[]") for item in value)


def _int_sequence(value: object, *, name: str) -> IntVector:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    return tuple(_as_int(item, name=f"{name}[]") for item in value)


def _cmdstan_model(stan_file: Path) -> StanPosteriorModel:
    module = cast(CmdStanPyModule, importlib.import_module("cmdstanpy"))
    return module.CmdStanModel(stan_file=str(stan_file))


def _build_bound(data: Mapping[str, object]) -> BoundModel:
    from bayeswire import Data, Param, PartiallyObserved, model
    from bayeswire.constraints import Positive
    from bayeswire.distributions import Exponential

    from bayesjax.model import bind_model

    @model
    class CensoredExponentialStanReference:
        """Lower-censored Exponential model matching the Stan reference."""

        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        observed_values = Data.vector(n_obs)
        missing_lower = Data.vector(n_mis)
        prior_rate = Data.scalar()

        rate = Param(Exponential(prior_rate), constraint=Positive())
        y = PartiallyObserved.vector(
            Exponential(rate),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_lower=missing_lower,
        )

    n = _as_int(data["N"], name="N")
    n_obs = _as_int(data["N_obs"], name="N_obs")
    n_mis = _as_int(data["N_mis"], name="N_mis")
    observed_idx = jnp.asarray(_int_sequence(data["observed_idx"], name="observed_idx")) - 1
    missing_idx = jnp.asarray(_int_sequence(data["missing_idx"], name="missing_idx")) - 1
    observed_values = jnp.asarray(
        _float_sequence(data["observed_values"], name="observed_values"),
        dtype=jnp.float64,
    )
    missing_lower = jnp.asarray(
        _float_sequence(data["missing_lower"], name="missing_lower"),
        dtype=jnp.float64,
    )
    prior_rate = _as_float(data["prior_rate"], name="prior_rate")
    return bind_model(
        CensoredExponentialStanReference,
        dict(
            n=n,
            n_obs=n_obs,
            n_mis=n_mis,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            observed_values=observed_values,
            missing_lower=missing_lower,
            prior_rate=prior_rate,
        ),
    )


def _block_trace(trace: NutsDiagnosticTrace) -> None:
    trace.is_divergent.block_until_ready()
    trace.acceptance_rate.block_until_ready()
    trace.num_integration_steps.block_until_ready()
    trace.num_trajectory_expansions.block_until_ready()
    trace.energy.block_until_ready()


def _block_result(result: SamplerResult) -> None:
    for value in result.samples.values():
        value.block_until_ready()
    _block_trace(result.diagnostics.warmup)
    _block_trace(result.diagnostics.sampling)


def _draw_column(draws: jax.Array, column_names: tuple[str, ...], *, name: str) -> jax.Array:
    if name not in column_names:
        raise ValueError(f"Missing Stan posterior draws for column: {name}")
    column_index = column_names.index(name)
    return draws[:, :, column_index].T


def _stan_draw_result(fit: StanFit, *, parameter: str) -> StanDrawResult:
    column_names = tuple(fit.column_names)
    draws = jnp.asarray(fit.draws(inc_warmup=False, concat_chains=False))
    return StanDrawResult(
        samples={parameter: _draw_column(draws, column_names, name=parameter)},
        diagnostics=StanDiagnosticTrace(
            is_divergent=_draw_column(draws, column_names, name="divergent__").astype(bool),
            acceptance_rate=_draw_column(draws, column_names, name="accept_stat__"),
        ),
    )


def _run(config: CensoredExponentialConfig) -> CensoredExponentialResult:
    from bayesjax.inference import compile_sampler
    from bayesjax.validation import standardized_discrepancy
    from integration._validation import summarize_scalar_draws

    parameter = "rate"
    data = _load_json(config.stan_data)
    bound = _build_bound(data)
    compiled = compile_sampler(bound, target_acceptance_rate=config.target_acceptance_rate)
    bayesjax_result = compiled.sample(
        seed=config.seed,
        num_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
    )
    _block_result(bayesjax_result)
    bayesjax_summary = summarize_scalar_draws(bayesjax_result.samples, parameter=parameter)

    stan_model = _cmdstan_model(config.stan_model)
    with tempfile.TemporaryDirectory(prefix="censored-exponential-stan-") as output_dir:
        fit = stan_model.sample(
            data=str(config.stan_data),
            seed=config.seed,
            chains=config.num_chains,
            parallel_chains=config.num_chains,
            iter_warmup=config.num_warmup,
            iter_sampling=config.num_samples,
            show_progress=False,
            output_dir=output_dir,
            adapt_delta=config.target_acceptance_rate,
        )
        stan_draws = _stan_draw_result(fit, parameter=parameter)
    stan_summary = summarize_scalar_draws(stan_draws.samples, parameter=parameter)

    if bayesjax_summary.rhat > config.max_rhat:
        raise AssertionError(
            f"bayesjax R-hat for {parameter} is {bayesjax_summary.rhat:.3f}; "
            f"expected <= {config.max_rhat:.3f}"
        )
    if stan_summary.rhat > config.max_rhat:
        raise AssertionError(
            f"Stan R-hat for {parameter} is {stan_summary.rhat:.3f}; "
            f"expected <= {config.max_rhat:.3f}"
        )
    if bayesjax_summary.ess < config.min_ess:
        raise AssertionError(
            f"bayesjax ESS for {parameter} is {bayesjax_summary.ess:.1f}; "
            f"expected >= {config.min_ess:.1f}"
        )
    if stan_summary.ess < config.min_ess:
        raise AssertionError(
            f"Stan ESS for {parameter} is {stan_summary.ess:.1f}; expected >= {config.min_ess:.1f}"
        )
    combined_mcse = math.sqrt(bayesjax_summary.mcse_mean**2 + stan_summary.mcse_mean**2)
    comparison = standardized_discrepancy(
        parameter=parameter,
        summary_name="mean",
        estimate=bayesjax_summary.mean,
        reference=stan_summary.mean,
        mcse=combined_mcse,
    )
    if comparison.k_min > config.max_k:
        raise AssertionError(
            f"bayesjax posterior mean for {parameter} differs from Stan by "
            f"{comparison.k_min:.2f} combined MCSEs; expected <= {config.max_k:.2f}"
        )
    return CensoredExponentialResult(
        parameter=parameter,
        signed_z=comparison.signed_z,
        k_min=comparison.k_min,
        bayesjax_mean=bayesjax_summary.mean,
        stan_mean=stan_summary.mean,
        combined_mcse=comparison.mcse,
        bayesjax_rhat=bayesjax_summary.rhat,
        stan_rhat=stan_summary.rhat,
        bayesjax_ess=bayesjax_summary.ess,
        stan_ess=stan_summary.ess,
        bayesjax_sampling_divergences=int(
            jnp.sum(bayesjax_result.diagnostics.sampling.is_divergent)
        ),
        stan_sampling_divergences=int(jnp.sum(stan_draws.diagnostics.is_divergent)),
    )


def _parse_args() -> CensoredExponentialConfig:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Compare lower-censored Exponential imputation against Stan."
    )
    parser.add_argument("--seed", type=int, default=42_100)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument("--max-k", type=float, default=4.0)
    parser.add_argument("--max-rhat", type=float, default=1.05)
    parser.add_argument("--min-ess", type=float, default=100.0)
    parser.add_argument("--target-acceptance-rate", type=float, default=0.9)
    parser.add_argument(
        "--stan-model",
        type=Path,
        default=root / "reference" / "stan" / "models" / "censored_exponential.stan",
    )
    parser.add_argument(
        "--stan-data",
        type=Path,
        default=root / "reference" / "stan" / "data" / "censored_exponential.json",
    )
    args = parser.parse_args()
    return CensoredExponentialConfig(
        seed=args.seed,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_k=args.max_k,
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        target_acceptance_rate=args.target_acceptance_rate,
        stan_model=args.stan_model,
        stan_data=args.stan_data,
    )


def main() -> int:
    _add_repo_paths()
    config = _parse_args()
    start = time.time()
    result = _run(config)
    elapsed = time.time() - start
    print(
        f"PASS parameter={result.parameter} signed_z={result.signed_z:.3f} "
        f"k_min={result.k_min:.3f} bayesjax_mean={result.bayesjax_mean:.6f} "
        f"stan_mean={result.stan_mean:.6f} combined_mcse={result.combined_mcse:.6f} "
        f"bayesjax_rhat={result.bayesjax_rhat:.4f} stan_rhat={result.stan_rhat:.4f} "
        f"bayesjax_ess={result.bayesjax_ess:.1f} stan_ess={result.stan_ess:.1f} "
        f"bayesjax_sampling_divergences={result.bayesjax_sampling_divergences} "
        f"stan_sampling_divergences={result.stan_sampling_divergences} elapsed={elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
