# ruff: noqa: E501
"""Build the executed mixed-backend workflow walkthrough HTML.

Unlike the older static planning page, this regenerates
``mixed-backend-workflow.html`` from a *real* non-linear mixed-backend run:

  1. Iteration 1 -- a model with deliberately wide priors fails the
     prior-predictive gate (run in-process on bayesjax), so the workflow goes
     back to square one and the model is respecified.
  2. Iteration 2 -- Bayesite ``simulate`` produces a canonical
     ``bayescycle.data.json.v1`` artifact, which a bayesjax in-process recovery
     fit consumes through the adapter boundary, and Bayesite ``recover-check``
     compares the fit back to truth.

The cross-backend handoff is recorded explicitly in each run's append-only
``run.json`` provenance and is queryable in ``walkthrough-runs.sqlite``.

Usage::  python docs/mixed-backend-workflow.py DEMO_MIXED_DIR
"""

from __future__ import annotations

import base64
import html
import json
import os
import sys
from pathlib import Path

DEMO = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("BAYESCYCLE_MIXED_DIR", "/tmp/bayescycle-walkthrough/mixed")
)
VIZ = DEMO / "viz"
OUT = Path(__file__).with_name("mixed-backend-workflow.html")
TRUTH = {"alpha": 1.25, "beta": 2.40, "sigma": 0.75}


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def code(text: str, lang: str = "") -> str:
    cls = f' data-lang="{esc(lang)}"' if lang else ""
    return f"<pre{cls}><code>{esc(text)}</code></pre>"


def json_block(value: object) -> str:
    return code(json.dumps(value, indent=2), "json")


def img(path: Path, alt: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b64}"/>'


def read_ndjson(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def y_range(pp_path: Path) -> tuple[float, float]:
    lines = read_ndjson(pp_path)
    ys = [v for d in lines[1:-1] if "values" in d for v in d["values"]["y"]]
    return min(ys), max(ys)


# --- gather ---------------------------------------------------------------
model_v1 = (DEMO / "model_v1.py").read_text()
model_v2 = (DEMO / "model.py").read_text()
plan = json.loads((DEMO / "plan.json").read_text())
toml_src = (DEMO / "workflow.toml").read_text()

rej_lo, rej_hi = y_range(DEMO / "run-prior-rejected" / "prior_predictive.ndjson")
ok_lo, ok_hi = y_range(DEMO / "run-prior" / "prior_predictive.ndjson")

data = json.loads((DEMO / "data.json").read_text())
obs_lo, obs_hi = min(data["y"]), max(data["y"])

sim_doc = json.loads((DEMO / "run-sim" / "simulated_data.json").read_text())
sim_run = json.loads((DEMO / "run-sim" / "run.json").read_text())
fit_run = json.loads((DEMO / "run-recover-fit" / "run.json").read_text())
fit_header = read_ndjson(DEMO / "run-recover-fit" / "posterior.ndjson")[0]
recovery_check = json.loads((DEMO / "run-recover-fit" / "recovery_check.json").read_text())
diagnostics = json.loads((DEMO / "run-recover-fit" / "diagnostics.json").read_text())


def recovery_table() -> str:
    rows = ""
    for name in recovery_check["target_order"]:
        t = recovery_check["targets"][name]
        contains = recovery_check["interval_contains_truth_by_target"][name]
        mark = '<span class="ok">in</span>' if contains else '<span class="bad">out</span>'
        rows += (
            f"<tr><td><code>{name}</code></td><td>{t['truth']:.3f}</td><td>{t['mean']:.4f}</td>"
            f"<td>{t['lower']:.4f}</td><td>{t['upper']:.4f}</td><td>{mark}</td>"
            f"<td>{t['rank']}/{t['rank_bounds']['max']}</td><td>{t['rhat']:.4f}</td><td>{t['ess']:.0f}</td></tr>"
        )
    pct = int(recovery_check["interval"] * 100)
    return (
        "<table><thead><tr><th>target</th><th>truth</th><th>post. mean</th>"
        f"<th>{pct}% lo</th><th>{pct}% hi</th><th>truth?</th><th>rank</th><th>R-hat</th><th>ESS</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def provenance_table() -> str:
    rows = ""
    for run_dir in sorted(DEMO.glob("run-*")):
        rj = run_dir / "run.json"
        if not rj.is_file():
            continue
        d = json.loads(rj.read_text())
        ins = ", ".join(f"{i['role']}" for i in d["inputs"]) or "&mdash;"
        outs = ", ".join(o["role"] for o in d["outputs"])
        rows += (
            f"<tr><td><code>{esc(run_dir.name)}/</code></td><td><code>{d['kind']}</code></td>"
            f"<td><span class='bk bk-{d['backend']}'>{d['backend']}</span></td>"
            f"<td>{ins}</td><td>{outs}</td></tr>"
        )
    return (
        "<table><thead><tr><th>run dir</th><th>kind</th><th>backend</th><th>inputs</th><th>outputs</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


# --- assemble -------------------------------------------------------------
sim_input_roles = [(i["role"], i.get("format")) for i in sim_run["inputs"]]
fit_input = next(i for i in fit_run["inputs"] if i["role"] == "data")

body = f"""
<section>
<h1>Mixed-backend workflows are explicit &mdash; and non-linear</h1>
<p>Bayescycle separates backend intent from backend-native data files. A workflow
plan is resolved before any run directory is created, and stages exchange
canonical <code>bayescycle.data.json.v1</code> artifacts. This page is generated
from a <strong>real executed run</strong>: a wide-prior model is rejected at the
prior-predictive gate, the model is respecified, then a Bayesite simulation hands
off to an in-process bayesjax recovery fit.</p>
<div class="pipeline">
  <span>model v1<br><small>wide priors</small></span><span class="arr">&rarr;</span>
  <span>prior predictive<br><small>bayesjax</small></span><span class="arr stop">&#10007;</span>
  <span class="redo">respecify<br><small>back to square one</small></span><span class="arr">&rarr;</span>
  <span>simulate<br><small>bayesite</small></span><span class="arr">&rarr;</span>
  <span>recover fit<br><small>bayesjax</small></span><span class="arr">&rarr;</span>
  <span>recover-check<br><small>bayesite</small></span>
</div>
</section>

<section>
<h2>Plan the backends before any run directory exists</h2>
<p>The mixed plan is mixed only because every stage backend is assigned explicitly:</p>
{code(toml_src, "toml")}
{code("uv run bayescycle workflow-plan --config workflow.toml", "bash")}
{json_block(plan)}
<p class="note">Partial assignments (<code>--simulate-backend</code> alone) and backend-specific
options that select no stage (<code>--engine</code> with <code>--backend bayesjax</code>) still
fail before any writes &mdash; see the run-directory contract docs.</p>
</section>

<div class="divider"><span>Iteration 1 &mdash; rejected at the gate</span></div>

<section>
<h2>A wide-prior model fails prior predictive</h2>
{code(model_v1, "python")}
{code("bayescycle prior-predictive model_v1.py --data inputs.json -o run-prior-rejected/ --backend bayesjax --seed 123 --draws 400", "bash")}
<p>The prior predictive distribution of <code>y</code> is wildly implausible:
prior-implied <code>y &isin; [{rej_lo:.0f}, {rej_hi:.0f}]</code> while the observed
data sits in <code>[{obs_lo:.1f}, {obs_hi:.1f}]</code>.</p>
<figure>{img(VIZ / "prior_predictive_rejected.png", "rejected prior predictive")}</figure>
<p class="redo-box"><strong>Decision: reject and respecify.</strong> The priors carry far more
mass than the measurement scale can justify. The workflow loops back to model
declaration rather than proceeding to simulation &mdash; a prior-predictive check
is a gate, not a formality.</p>
</section>

<div class="divider"><span>Iteration 2 &mdash; respecified</span></div>

<section>
<h2>Tightened priors pass the gate</h2>
{code(model_v2, "python")}
{code("bayescycle prior-predictive model.py --data inputs.json -o run-prior/ --backend bayesjax --seed 123 --draws 400", "bash")}
<p>Prior-implied <code>y &isin; [{ok_lo:.0f}, {ok_hi:.0f}]</code> now brackets the
observed range without dwarfing it.</p>
<figure>{img(VIZ / "prior_predictive.png", "respecified prior predictive")}</figure>
</section>

<section>
<h2>Canonical artifact handoff: Bayesite simulate &rarr; bayesjax fit</h2>
<p>Bayesite simulates fake data from a known truth and writes a canonical data document:</p>
{code("bayescycle simulate model.py --data inputs.json --truth truth.json -o run-sim/ --backend bayesite --engine $BAYESITE", "bash")}
<p><code>run-sim/simulated_data.json</code> is <code>{esc(sim_doc.get("format", "?"))}</code>;
its declared inputs ({", ".join(r for r, _ in sim_input_roles)}) and generated
observed <code>y</code> are backend-neutral.</p>
<p>The recovery fit then runs <strong>in-process on bayesjax</strong>, consuming the
canonical artifact through the adapter without ever reading Bayesite-native JSON:</p>
{code("bayescycle sample model.py --data run-sim/simulated_data.json -o run-recover-fit/ --backend bayesjax --seed 2 --chains 4 --warmup 400 --draws 500", "bash")}
<p>The append-only provenance makes the cross-backend edge explicit. The fit's
<code>data</code> input is exactly the simulator's output:</p>
{json_block({"simulate (bayesite) output": "run-sim/simulated_data.json", "fit (bayesjax) input.source_path": fit_input["source_path"].split("/")[-2] + "/" + fit_input["source_path"].split("/")[-1], "input.format": fit_input.get("format"), "input.sha256": fit_input["sha256"][:23] + "..."})}
</section>

<section>
<h2>Recover-check: bayesjax fit vs Bayesite-supplied truth</h2>
<p>Bayesite <code>recover-check</code> reads the bayesjax posterior stream
(<code>{fit_header["chain_count"]}</code> chains &times; <code>{fit_header["draw_count"] // fit_header["chain_count"]}</code> draws)
and compares it back to truth &mdash; no fingerprint barrier because the check
consumes only the fit and the truth:</p>
{code("bayescycle recover-check run-recover-fit/ --truth truth.json --targets targets.json --interval 0.8 --engine $BAYESITE", "bash")}
{recovery_table()}
<p class="note">Factual report: equal-tailed interval containment and ranks, no pass/fail verdict.
R-hat / ESS are from the in-process bayesjax fit;
<code>bayescycle diagnose</code> (Bayesite) recomputes them cross-backend without issue.</p>
</section>

<section>
<h2>Provenance is serializable</h2>
<p>Every run directory carries an append-only <code>run.json</code>
(<code>bayescycle.run.v1</code>). Across this mixed workflow:</p>
{provenance_table()}
<p>These flat records serialize directly into <code>docs/walkthrough-runs.sqlite</code>
(see <code>docs/run-provenance-db.py</code>), where the cross-backend handoff above
appears as a row in <code>workflow_edges</code>:</p>
{code("SELECT p.run_dir, e.producer_role, c.run_dir, e.consumer_role,\n       p.backend || ' -> ' || c.backend AS handoff\nFROM workflow_edges e\nJOIN runs p ON p.id = e.producer_run_id\nJOIN runs c ON c.id = e.consumer_run_id;\n-- mixed/run-sim  simulated_data  mixed/run-recover-fit  data  bayesite -> bayesjax", "sql")}
</section>
"""

doc = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>bayescycle mixed-backend workflow</title>
<style>
body{{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:900px;margin:32px auto;padding:0 20px;color:#20242a}}
h1{{font-size:26px}} h2{{font-size:20px;margin-top:0}}
pre{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:12px;overflow:auto;font-size:13px}}
pre[data-lang]::before{{content:attr(data-lang);display:block;color:#57606a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
section{{margin:26px 0;background:#fff;border:1px solid #e3e6ea;border-radius:12px;padding:18px 22px}}
.ok{{color:#116329;font-weight:700}} .bad{{color:#b42318;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border:1px solid #d0d7de;padding:6px 9px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
thead th{{background:#f6f8fa}}
figure{{margin:10px 0;background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:8px}}
figure img{{width:100%;height:auto;display:block}}
.note{{color:#57606a;font-size:13px}}
.pipeline{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:14px 0}}
.pipeline span{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:6px 10px;font-size:12px;text-align:center}}
.pipeline .arr{{background:none;border:none;color:#0969da;font-size:16px}}
.pipeline .arr.stop{{color:#b42318;font-weight:700}}
.pipeline .redo{{background:#fff3cd;border-color:#e0c869}}
.divider{{text-align:center;color:#0969da;font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:12px;margin:30px 0 0;border-top:1px solid #e3e6ea;padding-top:14px}}
.redo-box{{background:#fff3cd;border:1px solid #e0c869;border-radius:8px;padding:10px 14px}}
.bk{{font-size:11px;padding:2px 7px;border-radius:6px;font-weight:600}}
.bk-bayesite{{background:#dde7ff;color:#0a3069}} .bk-bayesjax{{background:#d7f5dd;color:#0a5128}}
</style></head>
<body>
{body}
<footer style="text-align:center;color:#8a929e;font-size:12px;margin:30px 0">Generated from an executed demo run; all images embedded; self-contained.</footer>
</body></html>
"""

OUT.write_text(doc, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
