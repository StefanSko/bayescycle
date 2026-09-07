#!/usr/bin/env bash
# Run inside this arm with completed initial artifacts and no revised outputs.
set -euo pipefail
cd "$(dirname "$0")"
for path in results/revised results/revised-prior results/revised-recovery logs/revised revision-provenance.json; do
  if [[ -e "$path" ]]; then echo "Refusing existing $path" >&2; exit 1; fi
done
export JAX_ENABLE_X64=true MPLBACKEND=Agg OMP_NUM_THREADS=1
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
export BAYESCYCLE_NO_AUTO_PROVISION=1
mkdir -p logs/revised
trap 'code=$?; echo "exit_code=$code" >> logs/revised/execution-status.log' EXIT
{ time uv run --project .. bayescycle prior-predictive model_revised.py --data design.json -o results/revised-prior --backend bayesjax --seed 4200 --draws 200; } > logs/revised/prior.log 2>&1
uv run --project .. python analysis_revised.py prior > logs/revised/prior-analysis.log 2>&1
{ time uv run --project .. bayescycle sample model_revised.py --data input/recovery.json -o results/revised-recovery --backend bayesjax --seed 4301 --chains 4 --warmup 500 --draws 1000 --target-accept 0.9 --max-treedepth 10; } > logs/revised/recovery.log 2>&1
uv run --project .. python analysis_revised.py recovery > logs/revised/recovery-analysis.log 2>&1
{ time uv run --project .. bayescycle sample model_revised.py --data input/data.json -o results/revised --backend bayesjax --seed 4201 --chains 4 --warmup 500 --draws 1000 --target-accept 0.9 --max-treedepth 10; } > logs/revised/fit.log 2>&1
uv run --project .. python analysis_revised.py revised > logs/revised/fit-analysis.log 2>&1
if ! { time uv run --project .. bayescycle plot trace results/revised -o results/revised/cli-trace.png --no-auto-provision; } > logs/revised/cli-plot.log 2>&1; then
  echo 'CLI visualization failed; see logs/revised/cli-plot.log' >&2
fi
