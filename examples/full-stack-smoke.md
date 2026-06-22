# Bayesite full-stack smoke

*2026-06-22T20:34:47Z by Showboat 0.6.1*
<!-- showboat-id: e8bca9f2-6143-479f-a27c-12613143ff28 -->

This executable smoke test exercises the full local stack: jaxstanv5 model authoring through bayescycle, Bayesite sampling and diagnostics, bayesite-idata export to ArviZ NetCDF, bayesite-viz plots, and Showboat documentation. It writes all generated artifacts under examples/_showboat_work and normalizes absolute CLI paths to repository-relative paths for verification.

```bash
set -euo pipefail
BAYESITE_BIN="${BAYESITE_BIN:-../bayesite/target/debug/bayesite}"
if [ ! -x "$BAYESITE_BIN" ]; then
  cargo build --quiet --manifest-path ../bayesite/Cargo.toml --bin bayesite
fi
BAYESITE_VIZ_SOURCE="${BAYESITE_VIZ_SOURCE:-git+https://github.com/StefanSko/bayesite-viz.git@a280945}"
uv --quiet run bayescycle --version
printf 'bayesite: %s\n' "$(basename "$BAYESITE_BIN")"
uvx --quiet --from "$BAYESITE_VIZ_SOURCE" bayesite-idata --help | sed -n '1p'
uvx --quiet --from "$BAYESITE_VIZ_SOURCE" bayesite-viz --help | sed -n '1p'

```

```output
bayescycle 0.1.0
bayesite: bayesite
Usage: bayesite-idata [OPTIONS] RUN_DIR
Usage: bayesite-viz [OPTIONS] COMMAND [ARGS]...
```

Start with a real jaxstanv5 model.py. The observed variable is vector-valued so posterior-predictive and PPC plots have a non-scalar observed dimension. The Dim metadata lets bayescycle write dims.json for downstream ArviZ coordinates.

```bash
rm -rf examples/_showboat_work
mkdir -p examples/_showboat_work
cat > examples/_showboat_work/model.py <<'PY'
from jaxstanv5 import Dim, Observed, Param, model
from jaxstanv5.distributions import Normal

obs = Dim("obs", coords=["a", "b", "c", "d"])


@model
class SmokeNormal:
    mu = Param(Normal(0.0, 1.0))
    y = Observed(Normal(mu, 1.0), dims=(obs,))
PY
cat > examples/_showboat_work/data.json <<'JSON'
{"y": [0.2, 0.5, 0.9, 1.4]}
JSON
python3 - <<'PY'
from pathlib import Path
for path in sorted(Path("examples/_showboat_work").glob("*.json")):
    print(path.name)
print("model.py")
PY

```

```output
data.json
model.py
```

Next bayescycle executes model.py, asks jaxstanv5 for canonical IR, prepares the run directory, and invokes the Bayesite engine for posterior sampling.

```bash
set -euo pipefail
BAYESITE_BIN="${BAYESITE_BIN:-../bayesite/target/debug/bayesite}"
if [ ! -x "$BAYESITE_BIN" ]; then
  cargo build --quiet --manifest-path ../bayesite/Cargo.toml --bin bayesite
fi
uv --quiet run bayescycle sample \
  examples/_showboat_work/model.py \
  --data examples/_showboat_work/data.json \
  -o examples/_showboat_work/run \
  --engine "$BAYESITE_BIN" \
  -- --seed 7 --chains 2 --warmup 100 --draws 100
find examples/_showboat_work/run -maxdepth 1 -type f -exec basename {} \; | sort
python3 - <<'PY'
import json
from pathlib import Path
run = Path("examples/_showboat_work/run")
ir = json.loads((run / "model.ir.json").read_text())
dims = json.loads((run / "dims.json").read_text())
header = json.loads((run / "posterior.ndjson").read_text().splitlines()[0])
print("ir:", ir["jaxstanv5_ir"])
print("dims:", dims["dims_format"], dims["dims"]["y"], dims["coords"]["obs"])
print("draws:", header["draws_format"], "chains", header["chain_count"], "draws", header["draw_count"])
print("parameters:", ", ".join(header["parameter_order"]))
PY

```

```output
data.json
dims.json
model.ir.json
posterior.ndjson
ir: 1
dims: bayescycle-dims-v1 ['obs'] ['a', 'b', 'c', 'd']
draws: v0-provisional chains 2 draws 200
parameters: mu
```

Bayesite owns sampler diagnostics. Run bayesite diagnose on the posterior stream and print a compact chain/rhat/ESS summary.

```bash
set -euo pipefail
BAYESITE_BIN="${BAYESITE_BIN:-../bayesite/target/debug/bayesite}"
if [ ! -x "$BAYESITE_BIN" ]; then
  cargo build --quiet --manifest-path ../bayesite/Cargo.toml --bin bayesite
fi
"$BAYESITE_BIN" diagnose \
  --fit examples/_showboat_work/run/posterior.ndjson \
  --out examples/_showboat_work/diagnostics.json
python3 - <<'PY'
import json
from pathlib import Path

d = json.loads(Path("examples/_showboat_work/diagnostics.json").read_text())
print("diagnostics:", d["diagnostics_format"])
print("chains:", d["source_chain_count"], "draws:", d["source_draw_count"])
print("mu rhat:", round(d["rhat"]["mu"], 3), "ess:", round(d["ess"]["mu"], 1))
for chain in d["chains"]:
    print(
        f"chain {chain['chain']}: divergences={chain['divergences']} "
        f"mean_accept={chain['mean_accept']:.3f} "
        f"tree_bins={chain['treedepth_histogram'][:3]}"
    )
PY

```

```output
diagnostics: v0-provisional
chains: 2 draws: 200
mu rhat: 1.009 ess: 49.1
chain 0: divergences=0 mean_accept=0.642 tree_bins=[0, 89, 11]
chain 1: divergences=0 mean_accept=0.912 tree_bins=[0, 76, 24]
```

For a model-checking plot, generate posterior predictive replicated y values from the same Bayesite fit stream.

```bash
set -euo pipefail
BAYESITE_BIN="${BAYESITE_BIN:-../bayesite/target/debug/bayesite}"
if [ ! -x "$BAYESITE_BIN" ]; then
  cargo build --quiet --manifest-path ../bayesite/Cargo.toml --bin bayesite
fi
"$BAYESITE_BIN" posterior-predictive \
  --model examples/_showboat_work/run/model.ir.json \
  --data examples/_showboat_work/run/data.json \
  --fit examples/_showboat_work/run/posterior.ndjson \
  --seed 8 \
  --out examples/_showboat_work/run/posterior_predictive.ndjson
python3 - <<'PY'
import json
from pathlib import Path
header = json.loads(Path("examples/_showboat_work/run/posterior_predictive.ndjson").read_text().splitlines()[0])
print("posterior_predictive:", header["posterior_predictive_format"])
print("draws:", header["draw_count"], "sites:", ", ".join(header["site_order"]))
print("site shape:", header["sites"][0]["shape"])
PY

```

```output
posterior_predictive: v0-provisional
draws: 200 sites: y
site shape: [4]
```

Now use bayesite-idata from bayesite-viz to convert the Bayesite run directory to an ArviZ NetCDF/DataTree file. The exporter also consumes the dims.json sidecar, so the observed y coordinate is named obs instead of an auto-generated dimension.

```bash
set -euo pipefail
BAYESITE_BIN="${BAYESITE_BIN:-../bayesite/target/debug/bayesite}"
if [ ! -x "$BAYESITE_BIN" ]; then
  cargo build --quiet --manifest-path ../bayesite/Cargo.toml --bin bayesite
fi
BAYESITE_VIZ_SOURCE="${BAYESITE_VIZ_SOURCE:-git+https://github.com/StefanSko/bayesite-viz.git@a280945}"
fit_path=$(uvx --quiet --from "$BAYESITE_VIZ_SOURCE" bayesite-idata \
  examples/_showboat_work/run \
  -o examples/_showboat_work/fit.nc \
  --validate require \
  --bayesite "$BAYESITE_BIN")
python3 - <<'PY' "$fit_path"
import os
import sys
print(os.path.relpath(sys.argv[1], start=os.getcwd()))
PY
uv --quiet run --with "$BAYESITE_VIZ_SOURCE" python - <<'PY'
import xarray as xr

dt = xr.open_datatree("examples/_showboat_work/fit.nc")
print("groups:", ", ".join(sorted(group.lstrip("/") or "/" for group in dt.groups)))
print("posterior.mu dims:", dt["/posterior"].ds["mu"].dims)
print("observed_data.y dims:", dt["/observed_data"].ds["y"].dims)
print("obs coords:", dt["/observed_data"].ds["obs"].values.tolist())
print("sample_stats:", ", ".join(sorted(dt["/sample_stats"].ds.data_vars)))
PY

```

```output
examples/_showboat_work/fit.nc
groups: /, observed_data, posterior, posterior_predictive, sample_stats
posterior.mu dims: ('chain', 'draw')
observed_data.y dims: ('obs',)
obs coords: ['a', 'b', 'c', 'd']
sample_stats: acceptance_rate, diverging, tree_depth
```

Finally render agent-friendly ArviZ plots through bayesite-viz. The CLI prints absolute paths, so the example normalizes them before recording output.

```bash
set -euo pipefail
BAYESITE_VIZ_SOURCE="${BAYESITE_VIZ_SOURCE:-git+https://github.com/StefanSko/bayesite-viz.git@a280945}"
for verb in trace rank ppc; do
  out="examples/_showboat_work/${verb}.png"
  path=$(PYTHONWARNINGS=ignore MPLBACKEND=Agg uvx --quiet --from "$BAYESITE_VIZ_SOURCE" bayesite-viz "$verb" \
    examples/_showboat_work/fit.nc \
    -o "$out")
  python3 - <<'PY' "$verb" "$path"
import os
import sys
from pathlib import Path
verb = sys.argv[1]
path = Path(sys.argv[2])
rel = os.path.relpath(path, start=os.getcwd())
status = "ok" if path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") else "bad"
print(f"{verb}: {rel} png={status}")
PY
done

```

```output
trace: examples/_showboat_work/trace.png png=ok
rank: examples/_showboat_work/rank.png png=ok
ppc: examples/_showboat_work/ppc.png png=ok
```

The trace and rank plots summarize chain behavior visually; the PPC plot checks replicated y draws against observed y.

```bash {image}
![Trace plot for the Bayesite full-stack smoke](examples/_showboat_work/trace.png)
```

![Trace plot for the Bayesite full-stack smoke](074c899d-2026-06-22.png)

```bash {image}
![Rank plot for the Bayesite full-stack smoke](examples/_showboat_work/rank.png)
```

![Rank plot for the Bayesite full-stack smoke](44380404-2026-06-22.png)

```bash {image}
![Posterior predictive check plot for the Bayesite full-stack smoke](examples/_showboat_work/ppc.png)
```

![Posterior predictive check plot for the Bayesite full-stack smoke](beca592f-2026-06-22.png)
