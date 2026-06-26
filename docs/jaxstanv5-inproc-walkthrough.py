"""Build a self-contained HTML walkthrough of the bayescycle jaxstanv5 run.

This regenerates ``jaxstanv5-inproc-walkthrough.html`` from a demo directory that
was produced by the steps documented in the walkthrough itself:

    DEMO/
      model.py
      data.json
      run/   (model.ir.json, data.json, dims.json, posterior.ndjson,
              diagnostics.json, posterior_predictive.ndjson)
      viz/   (trace.png, rank.png, energies.png, posterior.png, forest.png,
              ess-rhat.png, ppc.png)

Pass the demo directory as ``argv[1]`` or via ``BAYESCYCLE_WALKTHROUGH_DIR``;
it defaults to the scratch directory used when the walkthrough was authored.
The generated HTML embeds every image as base64 and has no external assets.
"""

from __future__ import annotations

import base64
import html
import json
import math
import os
import sys
from pathlib import Path

DEMO = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get(
        "BAYESCYCLE_WALKTHROUGH_DIR",
        "/tmp/bayescycle-linear-regression-jaxstanv5",
    )
)
RUN = DEMO / "run"
VIZ = DEMO / "viz"
OUT = DEMO / "walkthrough.html"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def code_block(text: str, lang: str = "") -> str:
    cls = f" data-lang=\"{esc(lang)}\"" if lang else ""
    return f'<pre class="code"{cls}><code>{esc(text)}</code></pre>'


def json_block(value: object) -> str:
    return code_block(json.dumps(value, indent=2), "json")


def file_chip(direction: str, path: str, note: str = "") -> str:
    d = direction.lower()
    cls = {
        "write": "chip-write",
        "read": "chip-read",
        "stdout": "chip-stdout",
        "input": "chip-input",
        "derived": "chip-derived",
    }.get(d, "chip-read")
    label = {
        "write": "WRITE",
        "read": "READ",
        "stdout": "STDOUT",
        "input": "INPUT",
        "derived": "DERIVED",
    }.get(d, direction.upper())
    note_html = f' <span class="chip-note">{esc(note)}</span>' if note else ""
    return (
        f'<div class="filechip {cls}">'
        f'<span class="chip-dir">{label}</span>'
        f'<span class="chip-path mono">{esc(path)}</span>'
        f"{note_html}</div>"
    )


def img_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


# ---------------------------------------------------------------------------
# Gather artifacts.
# ---------------------------------------------------------------------------
model_src = (DEMO / "model.py").read_text()

data = json.loads((DEMO / "data.json").read_text())
x = data["x"]
y = data["y"]
data_summary = {
    "n": len(x),
    "alpha_true": 1.25,
    "beta_true": 2.40,
    "sigma_true": 0.75,
    "x_range": [min(x), max(x)],
    "y_range": [round(min(y), 4), round(max(y), 4)],
    "x_head": [round(v, 4) for v in x[:5]],
    "y_head": [round(v, 4) for v in y[:5]],
    "data_size_bytes": (DEMO / "data.json").stat().st_size,
}

posterior_lines = (RUN / "posterior.ndjson").read_text().splitlines()
post_header = json.loads(posterior_lines[0])
post_first_draw = json.loads(posterior_lines[1])
post_trailer = json.loads(posterior_lines[-1])["trailer"]

# Posterior parameter summary from draws.
param_values: dict[str, list[float]] = {"alpha": [], "beta": [], "sigma": []}
divergences = 0
energies: list[float] = []
for line in posterior_lines:
    doc = json.loads(line)
    if "values" not in doc:
        continue
    for name in param_values:
        param_values[name].append(doc["values"][name])
    divergences += int(doc["diverging"])
    energies.append(doc["energy"])

posterior_summary = {}
for name, xs in param_values.items():
    mean = sum(xs) / len(xs)
    sd = (sum((v - mean) ** 2 for v in xs) / (len(xs) - 1)) ** 0.5
    posterior_summary[name] = {
        "mean": mean,
        "sd": sd,
        "q025": quantile(xs, 0.025),
        "median": quantile(xs, 0.5),
        "q975": quantile(xs, 0.975),
    }

diagnostics = json.loads((RUN / "diagnostics.json").read_text())
diag_summary = {
    "diagnostics_format": diagnostics["diagnostics_format"],
    "rhat": diagnostics["rhat"],
    "ess": diagnostics["ess"],
    "chains": [
        {
            "chain": c["chain"],
            "divergences": c["divergences"],
            "step_size": c["step_size"],
            "mean_accept": c["mean_accept"],
            "treedepth_histogram": c["treedepth_histogram"],
        }
        for c in diagnostics["chains"]
    ],
}

ppc_lines = (RUN / "posterior_predictive.ndjson").read_text().splitlines()
ppc_header = json.loads(ppc_lines[0])
ppc_first = json.loads(ppc_lines[1])
ppc_trailer = json.loads(ppc_lines[-1])["trailer"]
ppc_summary = {
    "posterior_predictive_format": ppc_header["posterior_predictive_format"],
    "seed": ppc_header["seed"],
    "source_fit_seed": ppc_header["source_fit_seed"],
    "draw_count": ppc_header["draw_count"],
    "site_order": ppc_header["site_order"],
    "site_y_shape": ppc_header["sites"][0]["shape"],
    "first_draw_y_head": [round(v, 4) for v in ppc_first["values"]["y"][:6]],
    "file_size_bytes": (RUN / "posterior_predictive.ndjson").stat().st_size,
    "line_count": len(ppc_lines),
}


def truncate_record(doc: dict, keep: int = 6) -> dict:
    """Trim long lists in a draw/header for display."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, list) and len(v) > keep:
            out[k] = v[:keep] + [f"... (+{len(v) - keep} more)"]
        elif isinstance(v, dict):
            out[k] = {
                kk: (vv[:keep] + [f"... (+{len(vv) - keep} more)"]
                     if isinstance(vv, list) and len(vv) > keep else vv)
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


# Posterior header is large only in declared metadata; show as-is (small).
post_header_display = post_header
post_first_draw_display = post_first_draw
ppc_header_display = truncate_record(ppc_header, keep=6)
ppc_first_display = truncate_record(ppc_first, keep=6)

# IR is one long line; pretty-print and truncate visually.
ir_obj = json.loads((RUN / "model.ir.json").read_text())
ir_pretty = json.dumps(ir_obj, indent=2)
ir_lines = ir_pretty.splitlines()
if len(ir_lines) > 60:
    ir_pretty = "\n".join(ir_lines[:60] + [f"... (+{len(ir_lines) - 60} more lines)"])

dims_obj = json.loads((RUN / "dims.json").read_text())

# ---------------------------------------------------------------------------
# Visualizations.
# ---------------------------------------------------------------------------
plots = [
    ("trace.png", "Trace", "Per-parameter chains overlaid; well-mixed and stationary."),
    ("rank.png", "Rank", "Rank plots are close to uniform across chains (good mixing)."),
    ("energies.png", "Energy (BFMI)",
     "Marginal vs transition energy overlap indicates healthy NUTS exploration."),
    ("posterior.png", "Posterior marginals",
     "Histogram marginals for alpha, beta, sigma."),
    ("forest.png", "Forest", "Credible intervals per parameter across chains."),
    ("ess-rhat.png", "ESS / R-hat", "Effective sample size and convergence summary."),
    ("ppc.png", "Posterior predictive",
     "Replicated y draws (light) bracket the observed density (dark)."),
]
plot_html = []
for fname, title, caption in plots:
    p = VIZ / fname
    if not p.is_file():
        continue
    uri = img_data_uri(p)
    plot_html.append(
        f'<figure class="plot">'
        f'<h3>{esc(title)}</h3>'
        f'<img alt="{esc(title)}" src="{uri}"/>'
        f'<figcaption>{esc(caption)}</figcaption>'
        f"</figure>"
    )
plots_section = "\n".join(plot_html)

# ---------------------------------------------------------------------------
# Commands as displayed.
# ---------------------------------------------------------------------------
cmd_sample_dry = (
    "bayescycle sample model.py --data data.json -o run/ \\\n"
    "  --backend jaxstanv5 --seed 123 --chains 4 \\\n"
    "  --warmup 300 --draws 500 --max-treedepth 8 \\\n"
    "  --target-accept 0.85 --dry-run"
)
cmd_sample = (
    "bayescycle sample model.py --data data.json -o run/ \\\n"
    "  --backend jaxstanv5 --seed 123 --chains 4 \\\n"
    "  --warmup 300 --draws 500 --max-treedepth 8 \\\n"
    "  --target-accept 0.85"
)
cmd_diagnose = "bayescycle diagnose run/ --engine /path/to/bayesite"
cmd_ppc = "bayescycle posterior-predictive run/ --seed 456 --engine /path/to/bayesite"
cmd_export = (
    "bayesite-idata run/ -o run/fit.nc --validate require\n"
    "bayesite-viz trace     run/fit.nc -o viz/trace.png\n"
    "bayesite-viz rank      run/fit.nc -o viz/rank.png\n"
    "bayesite-viz energies  run/fit.nc -o viz/energies.png\n"
    "bayesite-viz posterior run/fit.nc --kind hist -o viz/posterior.png\n"
    "bayesite-viz forest    run/fit.nc -o viz/forest.png\n"
    "bayesite-viz ess-rhat  run/fit.nc -o viz/ess-rhat.png\n"
    "bayesite-viz ppc       run/fit.nc --kind dist -o viz/ppc.png"
)

dry_run_doc = {
    "backend": "jaxstanv5",
    "model": "LinearRegression",
    "ir": "run/model.ir.json",
    "data": "run/data.json",
    "draws": "run/posterior.ndjson",
    "dims": "run/dims.json",
    "output": "run/",
    "sampler": {
        "seed": 123, "chains": 4, "warmup": 300, "draws": 500,
        "max_tree_depth": 8, "target_accept": 0.85,
    },
}


def io_table_html() -> str:
    rows = [
        (
            "bayescycle sample --backend jaxstanv5 --dry-run",
            ["model.py", "data.json"],
            ["run/model.ir.json", "run/data.json", "run/dims.json"],
            "plan only; no draws",
        ),
        (
            "bayescycle sample --backend jaxstanv5",
            ["model.py", "data.json"],
            [
                "run/model.ir.json",
                "run/data.json",
                "run/dims.json",
                "run/posterior.ndjson",
            ],
            "in-process NUTS",
        ),
        (
            "bayescycle diagnose",
            ["run/posterior.ndjson"],
            ["run/diagnostics.json"],
            "",
        ),
        (
            "bayescycle posterior-predictive",
            ["run/model.ir.json", "run/data.json", "run/posterior.ndjson"],
            ["run/posterior_predictive.ndjson"],
            "",
        ),
        (
            "bayesite-idata",
            [
                "run/model.ir.json",
                "run/data.json",
                "run/posterior.ndjson",
                "run/posterior_predictive.ndjson",
                "run/dims.json",
            ],
            ["run/fit.nc"],
            "ArviZ NetCDF",
        ),
        (
            "bayesite-viz <verb>",
            ["run/fit.nc"],
            ["viz/<verb>.png"],
            "matplotlib/ArviZ",
        ),
    ]

    def files(items: list[str], cls: str) -> str:
        return "".join(f'<span class="f {cls}">{esc(i)}</span>' for i in items)

    body = ""
    for cmd, reads, writes, note in rows:
        note_html = f'<div class="io-note">{esc(note)}</div>' if note else ""
        body += (
            "<tr>"
            f'<td class="mono io-cmd">{esc(cmd)}{note_html}</td>'
            f'<td>{files(reads, "f-read")}</td>'
            f'<td>{files(writes, "f-write")}</td>'
            "</tr>"
        )
    return (
        '<table class="io-table"><thead><tr>'
        "<th>command</th><th>reads</th><th>writes</th>"
        "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def tree_html() -> str:
    entries = [
        ("run/", "", "dir"),
        ("model.ir.json", "bayescycle sample &mdash; canonical IR compiled from model.py", "f"),
        ("data.json", "bayescycle sample &mdash; byte-for-byte copy of the input data", "f"),
        ("dims.json", "bayescycle sample &mdash; jaxstanv5 dimension/coord metadata", "f"),
        ("posterior.ndjson", "bayescycle sample &mdash; jaxstanv5/BlackJAX draws + stats", "f"),
        ("diagnostics.json", "bayescycle diagnose &mdash; recomputed R-hat/ESS/sample-stats", "f"),
        ("posterior_predictive.ndjson", "bayescycle posterior-predictive &mdash; replicated y", "f"),
        ("fit.nc", "bayesite-idata &mdash; ArviZ InferenceData (NetCDF)", "f"),
        ("viz/", "", "dir"),
        ("*.png", "bayesite-viz &mdash; ArviZ diagnostic plots", "f"),
    ]
    lines = []
    for name, origin, kind in entries:
        if kind == "dir":
            lines.append(f'<div class="tree-dir mono">{esc(name)}</div>')
        else:
            origin_html = f'<span class="tree-origin">&larr; {origin}</span>' if origin else ""
            lines.append(
                f'<div class="tree-row">'
                f'<span class="tree-file mono">{esc(name)}</span>{origin_html}</div>'
            )
    return f'<div class="tree">{"".join(lines)}</div>'


def param_table() -> str:
    rows = []
    truth = {"alpha": 1.25, "beta": 2.40, "sigma": 0.75}
    for name in ("alpha", "beta", "sigma"):
        s = posterior_summary[name]
        rows.append(
            "<tr>"
            f"<td class=\"mono\">{name}</td>"
            f"<td>{truth[name]:.3f}</td>"
            f"<td>{s['mean']:.4f}</td>"
            f"<td>{s['sd']:.4f}</td>"
            f"<td>{s['q025']:.4f}</td>"
            f"<td>{s['median']:.4f}</td>"
            f"<td>{s['q975']:.4f}</td>"
            f"<td>{diagnostics['rhat'][name]:.4f}</td>"
            f"<td>{diagnostics['ess'][name]:.0f}</td>"
            "</tr>"
        )
    return (
        '<table class="data">'
        "<thead><tr>"
        "<th>param</th><th>true</th><th>mean</th><th>sd</th>"
        "<th>2.5%</th><th>median</th><th>97.5%</th><th>R-hat</th><th>ESS</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Assemble HTML.
# ---------------------------------------------------------------------------
def section(num: str, title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="lead">{subtitle}</p>' if subtitle else ""
    return (
        f'<section class="step">'
        f'<h2><span class="num">{num}</span>{esc(title)}</h2>'
        f"{sub}{body}</section>"
    )


parts: list[str] = []

parts.append(
    section(
        "",
        "Overview",
        """
<p>This walkthrough runs a simple Bayesian linear regression end-to-end through
<span class="mono">bayescycle</span> using the in-process
<span class="mono">jaxstanv5</span>/BlackJAX backend, then exports the run
directory to an ArviZ <span class="mono">InferenceData</span> NetCDF file and
renders diagnostic plots with <span class="mono">bayesite-viz</span>.</p>
<p>The key architectural point: regardless of backend, every engine emits the
same <strong>bayescycle run-directory v0 contract</strong>. The model is authored
in <span class="mono">jaxstanv5</span>; <span class="mono">bayescycle</span> owns
artifact serialization and backend dispatch; Bayesite and bayesite-viz consume
the artifacts.</p>
<div class="pipeline">
  <span>model.py</span><span class="arr">&rarr;</span>
  <span>bayescycle sample<br><small>--backend jaxstanv5</small></span><span class="arr">&rarr;</span>
  <span>posterior.ndjson</span><span class="arr">&rarr;</span>
  <span>diagnose / ppc</span><span class="arr">&rarr;</span>
  <span>bayesite-idata</span><span class="arr">&rarr;</span>
  <span>bayesite-viz<br><small>ArviZ plots</small></span>
</div>

<h3 class="io-title">Filesystem effects &mdash; what each command reads and writes</h3>
<p class="lead">Inputs live outside the run directory; every artifact below is created
inside <span class="mono">run/</span> (plots land in <span class="mono">viz/</span>).
Each JSON block later in this page is tagged with the file it belongs to.</p>
"""
        + io_table_html()
        + tree_html(),
    )
)

parts.append(
    section(
        "1",
        "Model declaration",
        file_chip("input", "model.py", "hand-authored jaxstanv5 model")
        + code_block(model_src, "python"),
        "A flat linear regression with scalar intercept, slope, and a positive scale.",
    )
)

parts.append(
    section(
        "2",
        "Synthetic data (n = 1000)",
        file_chip("input", "data.json", "summary shown; full vectors on disk")
        + json_block(data_summary),
        "Generated from known parameters with Gaussian noise so we can check recovery.",
    )
)

parts.append(
    section(
        "3",
        "Dry-run: prepare the run directory",
        code_block(cmd_sample_dry, "bash")
        + "<p class=\"lead\">The dry run writes inputs and reports the plan without sampling:</p>"
        + file_chip("stdout", "(dry-run plan)", "printed JSON, not a file")
        + json_block(dry_run_doc)
        + "<p class=\"lead\">Resulting <span class=\"mono\">model.ir.json</span> (pretty-printed, truncated):</p>"
        + file_chip("write", "run/model.ir.json", "compiled from model.py")
        + code_block(ir_pretty, "json")
        + "<p class=\"lead\"><span class=\"mono\">dims.json</span> sidecar:</p>"
        + file_chip("write", "run/dims.json", "jaxstanv5 metadata")
        + json_block(dims_obj),
        "bayescycle compiles the jaxstanv5 model to canonical IR and copies the data.",
    )
)

parts.append(
    section(
        "4",
        "Sample in-process (jaxstanv5 / BlackJAX)",
        code_block(cmd_sample, "bash")
        + file_chip("write", "run/posterior.ndjson", "header + 2000 draws + trailer")
        + "<p class=\"lead\">Posterior stream header (per_draw_v2, with neutral model/data fingerprint):</p>"
        + file_chip("write", "run/posterior.ndjson", "line 1 &mdash; header")
        + json_block(post_header_display)
        + "<p class=\"lead\">First draw record (parameter values + per-draw sampler stats):</p>"
        + file_chip("write", "run/posterior.ndjson", "line 2 &mdash; first draw")
        + json_block(post_first_draw_display)
        + "<p class=\"lead\">Trailer (per-chain step sizes, tree-depth histograms, R-hat, ESS):</p>"
        + file_chip("write", "run/posterior.ndjson", "last line &mdash; trailer")
        + json_block(post_trailer),
        "4 chains &times; 500 draws &rarr; 2000 posterior draws written to "
        "posterior.ndjson (header + 2000 draws + trailer).",
    )
)

parts.append(
    section(
        "5",
        "Posterior summary &amp; recovery",
        file_chip("derived", "run/posterior.ndjson", "summary computed from draws")
        + param_table()
        + f'<p class="lead">Divergences across all chains: '
        f'<strong>{divergences}</strong>. Energy range: '
        f'<strong>{min(energies):.1f}</strong> &ndash; <strong>{max(energies):.1f}</strong>.</p>',
        "All three parameters are recovered tightly around their true values.",
    )
)

parts.append(
    section(
        "6",
        "Diagnose",
        code_block(cmd_diagnose, "bash")
        + file_chip("read", "run/posterior.ndjson", "fit input")
        + file_chip("write", "run/diagnostics.json", "recomputed diagnostics")
        + json_block(diag_summary),
        "Bayesite recomputes diagnostics from the posterior artifact.",
    )
)

parts.append(
    section(
        "7",
        "Posterior predictive",
        code_block(cmd_ppc, "bash")
        + file_chip("read", "run/model.ir.json, run/data.json, run/posterior.ndjson", "inputs")
        + file_chip("write", "run/posterior_predictive.ndjson", "header + 2000 draws + trailer")
        + "<p class=\"lead\">Header (truncated):</p>"
        + file_chip("write", "run/posterior_predictive.ndjson", "line 1 &mdash; header")
        + json_block(ppc_header_display)
        + "<p class=\"lead\">Summary:</p>"
        + file_chip("derived", "run/posterior_predictive.ndjson", "computed from draws")
        + json_block(ppc_summary),
        "2000 replicated datasets of 1000 y-values each, conditioned on the fit.",
    )
)

parts.append(
    section(
        "8",
        "ArviZ visualizations",
        code_block(cmd_export, "bash")
        + file_chip("write", "run/fit.nc", "ArviZ InferenceData (NetCDF)")
        + file_chip("write", "viz/*.png", "one PNG per bayesite-viz verb")
        + f'<div class="plots">{plots_section}</div>',
        "Export the run directory to NetCDF and render ArviZ plots via bayesite-viz.",
    )
)

parts.append(
    section(
        "9",
        "Bug found &amp; fixed during this run",
        """
<p>The end-to-end run surfaced a real artifact-boundary bug: BlackJAX occasionally
returns acceptance values slightly above 1.0 (for example
<span class="mono">1.000000119</span>) due to float32 roundoff, and Bayesite
correctly rejected those as outside <span class="mono">[0, 1]</span>. The
in-process writer now clamps tiny probability roundoff at the artifact boundary
and still rejects materially invalid probabilities.</p>
<p class="mono small">commit ad46adc &mdash; Clamp in-process acceptance probabilities</p>
""",
    )
)

body = "\n".join(parts)

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>bayescycle &times; jaxstanv5 &mdash; Linear regression walkthrough</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #171a21; --panel2: #1e222b; --ink: #e6e8ee;
    --muted: #9aa3b2; --accent: #6ea8fe; --accent2: #7ee787; --border: #2a2f3a;
    --code: #11141a;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  header.hero {{
    padding: 56px 24px 28px; border-bottom: 1px solid var(--border);
    background: radial-gradient(1200px 400px at 50% -120px, #20304d 0%, transparent 70%);
    text-align: center;
  }}
  header.hero h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: .2px; }}
  header.hero p {{ margin: 0; color: var(--muted); }}
  .badges {{ margin-top: 16px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }}
  .badge {{
    font-size: 12px; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--border); background: var(--panel2); color: var(--ink);
  }}
  main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
  section.step {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 22px 24px; margin: 20px 0;
  }}
  section.step h2 {{
    margin: 0 0 4px; font-size: 21px; display: flex; align-items: center; gap: 12px;
  }}
  .num {{
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 30px; height: 30px; padding: 0 8px; border-radius: 8px;
    background: var(--accent); color: #0b1020; font-weight: 700; font-size: 15px;
  }}
  .lead {{ color: var(--muted); margin: 8px 0 12px; }}
  pre.code {{
    background: var(--code); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.5;
    margin: 10px 0;
  }}
  pre.code code {{ font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; color: #d7dce6; white-space: pre; }}
  pre.code[data-lang]::before {{
    content: attr(data-lang); display: block; color: var(--muted);
    font-size: 11px; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 8px;
  }}
  .mono {{ font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; }}
  .small {{ font-size: 12px; color: var(--muted); }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 14px; }}
  table.data th, table.data td {{ border: 1px solid var(--border); padding: 7px 10px; text-align: right; }}
  table.data th:first-child, table.data td:first-child {{ text-align: left; }}
  table.data thead th {{ background: var(--panel2); color: var(--ink); }}
  table.data td {{ color: #cfd6e4; }}
  .pipeline {{
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: center;
    margin: 18px 0 4px;
  }}
  .pipeline span {{
    background: var(--panel2); border: 1px solid var(--border); border-radius: 9px;
    padding: 8px 12px; font-size: 13px; text-align: center;
  }}
  .pipeline .arr {{ background: none; border: none; color: var(--accent); font-size: 18px; padding: 0 2px; }}
  .pipeline small {{ color: var(--muted); }}
  .plots {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin-top: 12px; }}
  figure.plot {{
    margin: 0; background: #fff; border: 1px solid var(--border); border-radius: 12px;
    padding: 14px; overflow: hidden;
  }}
  figure.plot h3 {{ margin: 0 0 8px; font-size: 15px; color: #1a1d24; }}
  figure.plot img {{ width: 100%; height: auto; display: block; }}
  figure.plot figcaption {{ margin-top: 8px; font-size: 13px; color: #51596b; }}
  footer {{ text-align: center; color: var(--muted); padding: 28px; font-size: 13px; }}
  a {{ color: var(--accent); }}
  /* IO annotations */
  .io-title {{ margin: 22px 0 4px; font-size: 16px; }}
  .filechip {{
    display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin: 12px 0 -2px; padding: 5px 10px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--panel2); font-size: 12px;
  }}
  .chip-dir {{
    font-weight: 700; font-size: 10.5px; letter-spacing: .1em; padding: 2px 7px;
    border-radius: 6px; color: #0b1020;
  }}
  .chip-path {{ color: var(--ink); font-size: 12.5px; }}
  .chip-note {{ color: var(--muted); font-size: 11.5px; }}
  .chip-write .chip-dir {{ background: var(--accent2); }}
  .chip-read .chip-dir {{ background: var(--accent); }}
  .chip-input .chip-dir {{ background: #f0b562; }}
  .chip-stdout .chip-dir {{ background: #c9a0ff; }}
  .chip-derived .chip-dir {{ background: #6fd6c9; }}
  .chip-write {{ border-color: #2f5d3f; }}
  .chip-read {{ border-color: #2f4a6d; }}
  .chip-input {{ border-color: #6b5326; }}
  .chip-stdout {{ border-color: #4d3f6b; }}
  .chip-derived {{ border-color: #2c5d57; }}
  table.io-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  table.io-table th, table.io-table td {{
    border: 1px solid var(--border); padding: 9px 11px; text-align: left; vertical-align: top;
  }}
  table.io-table thead th {{ background: var(--panel2); }}
  table.io-table .io-cmd {{ font-size: 12.5px; max-width: 320px; }}
  table.io-table .io-note {{ color: var(--muted); font-size: 11px; margin-top: 3px; font-family: inherit; }}
  .f {{
    display: inline-block; margin: 2px 4px 2px 0; padding: 3px 7px; border-radius: 6px;
    font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; font-size: 11.5px;
    border: 1px solid var(--border);
  }}
  .f-read {{ background: #16233a; color: #bcd3f5; }}
  .f-write {{ background: #16321f; color: #b8e8c6; }}
  .tree {{
    background: var(--code); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; margin: 12px 0; font-size: 13px;
  }}
  .tree-dir {{ color: var(--accent); font-weight: 700; margin: 6px 0 2px; }}
  .tree-row {{ display: flex; gap: 10px; flex-wrap: wrap; padding: 2px 0 2px 18px; }}
  .tree-file {{ color: #b8e8c6; min-width: 230px; }}
  .tree-origin {{ color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<header class="hero">
  <h1>Bayesian linear regression &mdash; in-process <span class="mono">jaxstanv5</span> backend</h1>
  <p>From model declaration to ArviZ diagnostics through the bayescycle run-directory v0 contract</p>
  <div class="badges">
    <span class="badge">bayescycle sample --backend jaxstanv5</span>
    <span class="badge">BlackJAX NUTS</span>
    <span class="badge">n = 1000</span>
    <span class="badge">4 chains &times; 500 draws</span>
    <span class="badge">ArviZ via bayesite-viz</span>
  </div>
</header>
<main>
{body}
</main>
<footer>
  Generated walkthrough &mdash; all images embedded as base64; fully self-contained.
</footer>
</body>
</html>
"""

OUT.write_text(doc, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
