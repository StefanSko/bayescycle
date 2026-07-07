#!/usr/bin/env bash
# Build the demo run-directory trees that back the bayescycle workflow
# walkthroughs, then regenerate the HTML pages and the SQLite run index.
#
# This drives the *current* CLI (refactored integration modes + append-only
# bayescycle.run.v1 provenance) through two intentionally NON-LINEAR workflows:
#
#   complete/  single-backend (Bayesite) end-to-end loop. A first model with
#              deliberately wide priors fails the prior-predictive gate, so the
#              workflow goes back to square one with a respecified model before
#              the simulation gate and real fit.
#   mixed/     cross-backend loop. Same wide-prior rejection, then a Bayesite
#              `simulate` hands a canonical data artifact to a bayesjax
#              in-process recovery fit, checked back against truth by Bayesite.
#
# Difficulties hit while wiring this up are written up in
# docs/walkthrough-difficulties.md.
#
# Usage:
#   BAYESITE_BIN=/path/to/bayesite/target/release/bayesite \
#   BAYESITE_VIZ=/path/to/bayesite-viz \
#   docs/build-walkthroughs.sh [DEMO_ROOT]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
DEMO="${1:-${BAYESCYCLE_WALKTHROUGH_DIR:-/tmp/bayescycle-walkthrough}}"
WORK="$DEMO/_work"
BAYESITE_BIN="${BAYESITE_BIN:-$(command -v bayesite || true)}"
BAYESITE_VIZ="${BAYESITE_VIZ:-/home/user/bayesite-viz}"
VIZ_PY="${BAYESITE_VIZ_PYTHON:-3.13}"

if [ -z "$BAYESITE_BIN" ] || [ ! -x "$BAYESITE_BIN" ]; then
  echo "set BAYESITE_BIN to a built bayesite engine binary" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1  # keep demo trees free of __pycache__ (difficulties #9)
mkdir -p "$WORK"

# The engine is used directly: bayescycle's preflight reads the engine's
# JSON usage message (walkthrough difficulty #1 is fixed).
ENG="$BAYESITE_BIN"

bc() { ( cd "$PROJECT" && uv run --extra inproc bayescycle "$@" ); }
viz() { ( cd "$PROJECT" && uv run --no-project --with "$BAYESITE_VIZ" --python "$VIZ_PY" -- "$@" ); }

# --- shared model + data --------------------------------------------------
write_models() {
  local dir="$1"
  cat > "$dir/model_v1.py" <<'PY'
"""Iteration 1: priors are far too wide; the prior-predictive gate rejects it."""

from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal


@model
class LinRegWide:
    x = Data.vector()
    alpha = Param(Normal(0.0, 100.0))
    beta = Param(Normal(0.0, 100.0))
    sigma = Param(HalfNormal(50.0), constraint=Positive())
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))
PY
  cat > "$dir/model.py" <<'PY'
"""Iteration 2: priors tightened to a plausible scale after the gate failed."""

from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal


@model
class LinReg:
    x = Data.vector()
    alpha = Param(Normal(0.0, 2.5))
    beta = Param(Normal(0.0, 2.5))
    sigma = Param(HalfNormal(1.0), constraint=Positive())
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))
PY
}

write_data() {
  local dir="$1"
  uv run --python 3.12 python - "$dir" <<'PY'
import json
import random
import sys

dir_ = sys.argv[1]
random.seed(20260628)
alpha, beta, sigma, n = 1.25, 2.40, 0.75, 200
x = [round(random.uniform(-2.0, 2.0), 4) for _ in range(n)]
y = [round(alpha + beta * xi + random.gauss(0.0, sigma), 4) for xi in x]
json.dump({"x": x, "y": y}, open(f"{dir_}/data.json", "w"))
json.dump({"x": x}, open(f"{dir_}/inputs.json", "w"))
json.dump({"alpha": alpha, "beta": beta, "sigma": sigma}, open(f"{dir_}/truth.json", "w"))
json.dump(
    {"targets": [{"name": n_, "truth": n_, "posterior": n_} for n_ in ("alpha", "beta", "sigma")]},
    open(f"{dir_}/targets.json", "w"),
)
json.dump(
    {
        "recover_scenario": "v0-provisional",
        "data": {"x": x},
        "sample": {"chains": 4, "warmup": 400, "draws": 500, "target_accept": 0.9},
        "seed": 7,
    },
    open(f"{dir_}/recover_scenario.json", "w"),
)
json.dump(
    {
        "sbc_scenario": "v0-provisional",
        "data": {"x": x},
        "sample": {"chains": 2, "warmup": 300, "draws": 300},
        "seed": 11,
    },
    open(f"{dir_}/sbc_scenario.json", "w"),
)
print(f"data n={n} truth alpha={alpha} beta={beta} sigma={sigma}")
PY
}

# Render a small prior-predictive figure (left: draws of y vs x, right: marginal)
prior_plot() {
  local ppfile="$1" out="$2" title="$3"
  viz python - "$ppfile" "$out" "$title" <<'PY'
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ppfile, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
lines = [json.loads(line) for line in open(ppfile) if line.strip()]
header = lines[0]
draws = [d for d in lines[1:-1] if "values" in d]
x = header["declared_data"]["x"]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
for d in draws[:60]:
    ax0.plot(x, d["values"]["y"], color="#4061d0", alpha=0.18, lw=0.8)
ax0.set_title("prior predictive draws of y vs x")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ally = [v for d in draws for v in d["values"]["y"]]
ax1.hist(ally, bins=60, color="#4061d0", alpha=0.8)
ax1.set_title("prior predictive marginal of y")
ax1.set_xlabel("y")
fig.suptitle(title)
fig.tight_layout()
fig.savefig(out, dpi=96)
print(f"prior plot -> {out}  y in [{min(ally):.1f}, {max(ally):.1f}]")
PY
}

# bayesite-idata reads the canonical run/data.json directly (walkthrough
# difficulty #6 is fixed): the engine and the exporter both parse the wrapped
# {format, variables} document natively, so the run directory needs no staging.
viz_run() {
  local run="$1" vizdir="$2"
  mkdir -p "$vizdir"
  viz bayesite-idata "$run" -o "$run/fit.nc" --validate require --bayesite "$ENG"
  for verb in trace rank energies forest ess-rhat; do
    viz bayesite-viz "$verb" "$run/fit.nc" -o "$vizdir/$verb.png" >/dev/null
  done
  viz bayesite-viz posterior "$run/fit.nc" --kind hist -o "$vizdir/posterior.png" >/dev/null
  viz bayesite-viz ppc "$run/fit.nc" --kind dist -o "$vizdir/ppc.png" >/dev/null
}

# =========================================================================
# COMPLETE workflow: single backend (Bayesite), non-linear.
# =========================================================================
build_complete() {
  local C="$DEMO/complete"
  rm -rf "$C"
  mkdir -p "$C/viz"
  write_models "$C"
  write_data "$C"

  echo "[complete] iteration 1: wide priors -> prior-predictive gate"
  bc prior-predictive "$C/model_v1.py" --data "$C/inputs.json" \
    -o "$C/run-prior-rejected" --backend bayesite --engine "$ENG" --seed 123 --draws 400
  prior_plot "$C/run-prior-rejected/prior_predictive.ndjson" \
    "$C/viz/prior_predictive_rejected.png" "Iteration 1 (wide priors): prior predictive is absurd"

  echo "[complete] iteration 2: respecified priors"
  bc prior-predictive "$C/model.py" --data "$C/inputs.json" \
    -o "$C/run-prior" --backend bayesite --engine "$ENG" --seed 123 --draws 400
  prior_plot "$C/run-prior/prior_predictive.ndjson" \
    "$C/viz/prior_predictive.png" "Iteration 2 (respecified priors): prior predictive is plausible"

  echo "[complete] simulation gate"
  bc simulate "$C/model.py" --data "$C/inputs.json" --truth "$C/truth.json" \
    -o "$C/run-sim" --backend bayesite --engine "$ENG" --seed 1
  bc sample "$C/model.py" --data "$C/run-sim/simulated_data.json" \
    -o "$C/run-recover-fit" --backend bayesite --engine "$ENG" \
    --seed 2 --chains 4 --warmup 400 --draws 500
  bc recover-check "$C/run-recover-fit" --truth "$C/truth.json" \
    --targets "$C/targets.json" --interval 0.8 --engine "$ENG"
  bc recover "$C/model.py" --scenario "$C/recover_scenario.json" \
    -o "$C/run-recover" --engine "$ENG"
  bc sbc "$C/model.py" --scenario "$C/sbc_scenario.json" \
    -o "$C/run-sbc" --replicates 48 --engine "$ENG"

  echo "[complete] real fit + diagnostics + checks"
  bc sample "$C/model.py" --data "$C/data.json" -o "$C/run" \
    --backend bayesite --engine "$ENG" \
    --seed 123 --chains 4 --warmup 400 --draws 500 --max-treedepth 8 --target-accept 0.9
  bc diagnose "$C/run" --engine "$ENG"
  bc posterior-predictive "$C/run" --seed 456 --engine "$ENG"
  bc posterior-check "$C/run" --seed 789 --engine "$ENG"

  echo "[complete] visualization"
  viz_run "$C/run" "$C/viz"
}

# =========================================================================
# MIXED workflow: Bayesite simulate -> bayesjax in-process recovery fit.
# =========================================================================
build_mixed() {
  local M="$DEMO/mixed"
  rm -rf "$M"
  mkdir -p "$M/viz"
  write_models "$M"
  write_data "$M"

  cat > "$M/workflow.toml" <<'TOML'
[workflow]
mode = "mixed"

[stages.simulate]
backend = "bayesite"

[stages.recover]
backend = "bayesjax"
TOML
  ( cd "$PROJECT" && uv run bayescycle workflow-plan --config "$M/workflow.toml" ) > "$M/plan.json"

  echo "[mixed] iteration 1: wide priors -> prior-predictive gate (bayesjax in-process)"
  bc prior-predictive "$M/model_v1.py" --data "$M/inputs.json" \
    -o "$M/run-prior-rejected" --backend bayesjax --seed 123 --draws 400
  prior_plot "$M/run-prior-rejected/prior_predictive.ndjson" \
    "$M/viz/prior_predictive_rejected.png" "Iteration 1 (wide priors): prior predictive is absurd"

  echo "[mixed] iteration 2: respecified priors (bayesjax in-process)"
  bc prior-predictive "$M/model.py" --data "$M/inputs.json" \
    -o "$M/run-prior" --backend bayesjax --seed 123 --draws 400
  prior_plot "$M/run-prior/prior_predictive.ndjson" \
    "$M/viz/prior_predictive.png" "Iteration 2 (respecified priors): prior predictive is plausible"

  echo "[mixed] cross-backend handoff: bayesite simulate -> bayesjax fit"
  bc simulate "$M/model.py" --data "$M/inputs.json" --truth "$M/truth.json" \
    -o "$M/run-sim" --backend bayesite --engine "$ENG" --seed 1
  bc sample "$M/model.py" --data "$M/run-sim/simulated_data.json" \
    -o "$M/run-recover-fit" --backend bayesjax \
    --seed 2 --chains 4 --warmup 400 --draws 500 --max-treedepth 8 --target-accept 0.9
  bc recover-check "$M/run-recover-fit" --truth "$M/truth.json" \
    --targets "$M/targets.json" --interval 0.8 --engine "$ENG"
  bc diagnose "$M/run-recover-fit" --engine "$ENG"
}

build_complete
build_mixed

echo "[generate] HTML walkthroughs + SQLite index"
( cd "$PROJECT" && uv run --python 3.12 python docs/workflow-walkthrough.py "$DEMO/complete" )
( cd "$PROJECT" && uv run --python 3.12 python docs/mixed-backend-workflow.py "$DEMO/mixed" )
( cd "$PROJECT" && uv run --python 3.12 python docs/run-provenance-db.py "$DEMO" docs/walkthrough-runs.sqlite )

echo "done. demo root: $DEMO"
