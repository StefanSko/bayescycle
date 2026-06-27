# ruff: noqa: E501
"""Build a self-contained HTML walkthrough of the full bayescycle workflow.

This regenerates ``workflow-walkthrough.html`` from a demo directory that was
produced by running the commands documented in the walkthrough itself. It covers
the complete McElreath-style loop: the *simulation gate* (prior predictive,
simulate, recovery, single-scenario recover, SBC) followed by the *real fit*
(sample, diagnose, posterior predictive, posterior check) and ArviZ
visualization.

Expected demo layout (see the commands embedded in the page)::

    DEMO/
      model.py
      data.json                inputs.json  truth.json  targets.json
      recover_scenario.json    sbc_scenario.json
      run-prior/               prior_predictive.ndjson (+ ir/data/dims)
      run-sim/                 simulated_data.json (+ truth/ir/data/dims)
      run-recover-fit/         posterior.ndjson  recovery_check.json (+ ir/data/dims)
      run-recover/             recovery.json (+ ir/scenario/dims)
      run-sbc/                 sbc.json (+ ir/scenario/dims)
      run/                     model.ir.json data.json dims.json posterior.ndjson
                               diagnostics.json posterior_predictive.ndjson
                               posterior_check.json fit.nc
      viz/                     prior_predictive.png trace.png rank.png energies.png
                               posterior.png forest.png ess-rhat.png ppc.png

Pass the demo directory as ``argv[1]`` or via ``BAYESCYCLE_WALKTHROUGH_DIR``.
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
        "/tmp/bayescycle-complete-workflow",
    )
)
VIZ = DEMO / "viz"
OUT = Path(__file__).with_name("workflow-walkthrough.html")

TRUTH = {"alpha": 1.25, "beta": 2.40, "sigma": 0.75}


# ---------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------
def esc(text: str) -> str:
    return html.escape(text, quote=False)


def code_block(text: str, lang: str = "") -> str:
    cls = f' data-lang="{esc(lang)}"' if lang else ""
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


def truncate(value: object, keep: int = 6) -> object:
    """Recursively trim long lists for display."""
    if isinstance(value, list):
        head = [truncate(v, keep) for v in value[:keep]]
        if len(value) > keep:
            head.append(f"... (+{len(value) - keep} more)")
        return head
    if isinstance(value, dict):
        return {k: truncate(v, keep) for k, v in value.items()}
    return value


def read_ndjson(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    sd = (sum((v - m) ** 2 for v in xs) / (len(xs) - 1)) ** 0.5
    return m, sd


# ---------------------------------------------------------------------------
# Gather artifacts.
# ---------------------------------------------------------------------------
model_src = (DEMO / "model.py").read_text()

data = json.loads((DEMO / "data.json").read_text())
x_all, y_all = data["x"], data["y"]
data_summary = {
    "n": len(x_all),
    "alpha_true": TRUTH["alpha"],
    "beta_true": TRUTH["beta"],
    "sigma_true": TRUTH["sigma"],
    "x_range": [round(min(x_all), 4), round(max(x_all), 4)],
    "y_range": [round(min(y_all), 4), round(max(y_all), 4)],
    "x_head": [round(v, 4) for v in x_all[:5]],
    "y_head": [round(v, 4) for v in y_all[:5]],
    "data_size_bytes": (DEMO / "data.json").stat().st_size,
}

ir_obj = json.loads((DEMO / "run" / "model.ir.json").read_text())
ir_pretty = json.dumps(ir_obj, indent=2)
ir_lines = ir_pretty.splitlines()
if len(ir_lines) > 48:
    ir_pretty = "\n".join(ir_lines[:48] + [f"... (+{len(ir_lines) - 48} more lines)"])
dims_obj = json.loads((DEMO / "run" / "dims.json").read_text())

# Prior predictive (in-process) -------------------------------------------------
pp_lines = read_ndjson(DEMO / "run-prior" / "prior_predictive.ndjson")
pp_header = pp_lines[0]
pp_draws = pp_lines[1:-1]
pp_param_summary = {}
for name in ("alpha", "beta", "sigma"):
    xs = [d["values"][name] for d in pp_draws]
    m, sd = mean_sd(xs)
    pp_param_summary[name] = {"prior_mean": round(m, 3), "prior_sd": round(sd, 3)}
pp_y_flat = [v for d in pp_draws for v in d["values"]["y"]]
pp_summary = {
    "prior_predictive_format": pp_header["prior_predictive_format"],
    "artifact_scope": pp_header["artifact_scope"],
    "draw_count": pp_header["draw_count"],
    "site_order": pp_header["site_order"],
    "site_roles": {s["name"]: s["role"] for s in pp_header["sites"]},
    "prior_parameter_summary": pp_param_summary,
    "prior_predictive_y_range": [round(min(pp_y_flat), 2), round(max(pp_y_flat), 2)],
    "file_size_bytes": (DEMO / "run-prior" / "prior_predictive.ndjson").stat().st_size,
}
pp_first_display = truncate(pp_draws[0], keep=4)

# Simulate ---------------------------------------------------------------------
sim_doc = json.loads((DEMO / "run-sim" / "simulated_data.json").read_text())
sim_y = sim_doc["y"]["values"] if isinstance(sim_doc["y"], dict) else sim_doc["y"]
sim_summary = {
    "truth_used": TRUTH,
    "declared_inputs": [k for k in sim_doc if k == "x"],
    "generated_observed": [k for k in sim_doc if k == "y"],
    "n": len(sim_y),
    "y_head": [round(v, 4) for v in sim_y[:6]],
    "note": "plain typed data document; carries no simulation marker so `sample` is provenance-agnostic",
}

# Recovery fit + recover-check -------------------------------------------------
rec_fit_header = read_ndjson(DEMO / "run-recover-fit" / "posterior.ndjson")[0]
recovery_check = json.loads((DEMO / "run-recover-fit" / "recovery_check.json").read_text())

# Single-scenario recover ------------------------------------------------------
recover = json.loads((DEMO / "run-recover" / "recovery.json").read_text())

# SBC --------------------------------------------------------------------------
sbc = json.loads((DEMO / "run-sbc" / "sbc.json").read_text())

# Real fit ---------------------------------------------------------------------
posterior_lines = read_ndjson(DEMO / "run" / "posterior.ndjson")
post_header = posterior_lines[0]
post_first = posterior_lines[1]
post_trailer = posterior_lines[-1]["trailer"]
param_values: dict[str, list[float]] = {"alpha": [], "beta": [], "sigma": []}
divergences = 0
energies: list[float] = []
for doc in posterior_lines:
    if "values" not in doc:
        continue
    for name in param_values:
        param_values[name].append(doc["values"][name])
    divergences += int(doc["diverging"])
    energies.append(doc["energy"])
posterior_summary = {}
for name, xs in param_values.items():
    m, sd = mean_sd(xs)
    posterior_summary[name] = {
        "mean": m,
        "sd": sd,
        "q025": quantile(xs, 0.025),
        "median": quantile(xs, 0.5),
        "q975": quantile(xs, 0.975),
    }

diagnostics = json.loads((DEMO / "run" / "diagnostics.json").read_text())
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
        }
        for c in diagnostics["chains"]
    ],
}

ppc_lines = read_ndjson(DEMO / "run" / "posterior_predictive.ndjson")
ppc_header = ppc_lines[0]
ppc_summary = {
    "posterior_predictive_format": ppc_header["posterior_predictive_format"],
    "seed": ppc_header["seed"],
    "source_fit_seed": ppc_header["source_fit_seed"],
    "draw_count": ppc_header["draw_count"],
    "site_order": ppc_header["site_order"],
    "file_size_bytes": (DEMO / "run" / "posterior_predictive.ndjson").stat().st_size,
    "line_count": len(ppc_lines),
}

posterior_check = json.loads((DEMO / "run" / "posterior_check.json").read_text())


# ---------------------------------------------------------------------------
# Visualizations.
# ---------------------------------------------------------------------------
prior_plot = VIZ / "prior_predictive.png"
arviz_plots = [
    ("trace.png", "Trace", "Per-parameter chains overlaid; well-mixed and stationary."),
    ("rank.png", "Rank", "Rank plots close to uniform across chains (good mixing)."),
    (
        "energies.png",
        "Energy (BFMI)",
        "Marginal vs transition energy overlap indicates healthy NUTS.",
    ),
    ("posterior.png", "Posterior marginals", "Histogram marginals for alpha, beta, sigma."),
    ("forest.png", "Forest", "Credible intervals per parameter across chains."),
    ("ess-rhat.png", "ESS / R-hat", "Effective sample size and convergence summary."),
    (
        "ppc.png",
        "Posterior predictive",
        "Replicated y (light) bracket the observed density (dark).",
    ),
]


def plot_figure(path: Path, title: str, caption: str) -> str:
    return (
        f'<figure class="plot"><h3>{esc(title)}</h3>'
        f'<img alt="{esc(title)}" src="{img_data_uri(path)}"/>'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


arviz_section = "\n".join(
    plot_figure(VIZ / fname, title, caption)
    for fname, title, caption in arviz_plots
    if (VIZ / fname).is_file()
)


# ---------------------------------------------------------------------------
# Commands as displayed.
# ---------------------------------------------------------------------------
cmd_prior = (
    "bayescycle prior-predictive model.py --data inputs.json \\\n"
    "  -o run-prior/ --backend jaxstanv5 --seed 123 --draws 400"
)
cmd_prior_bayesite = (
    "bayescycle prior-predictive model.py --data inputs.json \\\n"
    "  -o run-prior/ --seed 123 --draws 400   # --backend bayesite (default)"
)
cmd_sim = (
    "bayescycle simulate model.py --data inputs.json --truth truth.json \\\n  -o run-sim/ --seed 1"
)
cmd_recfit = (
    "bayescycle sample model.py --data run-sim/simulated_data.json \\\n"
    "  -o run-recover-fit/ --backend bayesite --seed 2 \\\n"
    "  --chains 4 --warmup 400 --draws 500"
)
cmd_reccheck = (
    "bayescycle recover-check run-recover-fit/ --truth truth.json \\\n"
    "  --targets targets.json --interval 0.8"
)
cmd_recover = "bayescycle recover model.py --scenario recover_scenario.json -o run-recover/"
cmd_sbc = "bayescycle sbc model.py --scenario sbc_scenario.json -o run-sbc/ --replicates 64"
cmd_sample = (
    "bayescycle sample model.py --data data.json -o run/ \\\n"
    "  --backend jaxstanv5 --seed 123 --chains 4 \\\n"
    "  --warmup 300 --draws 500 --max-treedepth 8 --target-accept 0.85"
)
cmd_diagnose = "bayescycle diagnose run/"
cmd_ppc = "bayescycle posterior-predictive run/ --seed 456"
cmd_pcheck = "bayescycle posterior-check run/ --seed 789"
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


def io_table_html() -> str:
    rows = [
        (
            "bayescycle prior-predictive --backend jaxstanv5",
            ["model.py", "inputs.json"],
            ["run-prior/prior_predictive.ndjson"],
            "simulation gate",
        ),
        (
            "bayescycle simulate",
            ["model.py", "inputs.json", "truth.json"],
            ["run-sim/simulated_data.json"],
            "fake data from truth",
        ),
        (
            "bayescycle sample --backend bayesite",
            ["model.py", "run-sim/simulated_data.json"],
            ["run-recover-fit/posterior.ndjson"],
            "fit the fake data",
        ),
        (
            "bayescycle recover-check",
            ["run-recover-fit/posterior.ndjson", "truth.json", "targets.json"],
            ["run-recover-fit/recovery_check.json"],
            "fit vs truth facts",
        ),
        (
            "bayescycle recover",
            ["model.py", "recover_scenario.json"],
            ["run-recover/recovery.json"],
            "single scenario",
        ),
        (
            "bayescycle sbc",
            ["model.py", "sbc_scenario.json"],
            ["run-sbc/sbc.json"],
            "calibration ranks",
        ),
        (
            "bayescycle sample --backend jaxstanv5",
            ["model.py", "data.json"],
            ["run/posterior.ndjson"],
            "real fit, in-process",
        ),
        ("bayescycle diagnose", ["run/posterior.ndjson"], ["run/diagnostics.json"], ""),
        (
            "bayescycle posterior-predictive",
            ["run/model.ir.json", "run/data.json", "run/posterior.ndjson"],
            ["run/posterior_predictive.ndjson"],
            "",
        ),
        (
            "bayescycle posterior-check",
            ["run/model.ir.json", "run/data.json", "run/posterior.ndjson"],
            ["run/posterior_check.json"],
            "",
        ),
        ("bayesite-idata", ["run/*"], ["run/fit.nc"], "ArviZ NetCDF"),
        ("bayesite-viz <verb>", ["run/fit.nc"], ["viz/<verb>.png"], "matplotlib/ArviZ"),
    ]

    def files(items: list[str], cls: str) -> str:
        return "".join(f'<span class="f {cls}">{esc(i)}</span>' for i in items)

    body = ""
    for cmd, reads, writes, note in rows:
        note_html = f'<div class="io-note">{esc(note)}</div>' if note else ""
        body += (
            "<tr>"
            f'<td class="mono io-cmd">{esc(cmd)}{note_html}</td>'
            f"<td>{files(reads, 'f-read')}</td>"
            f"<td>{files(writes, 'f-write')}</td>"
            "</tr>"
        )
    return (
        '<table class="io-table"><thead><tr>'
        "<th>command</th><th>reads</th><th>writes</th>"
        "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def tree_html() -> str:
    groups = [
        (
            "run-prior/",
            [
                (
                    "prior_predictive.ndjson",
                    "bayescycle prior-predictive &mdash; declared-data-conditioned site draws",
                )
            ],
        ),
        (
            "run-sim/",
            [
                (
                    "simulated_data.json",
                    "bayescycle simulate &mdash; declared inputs + generated observed y",
                )
            ],
        ),
        (
            "run-recover-fit/",
            [
                ("posterior.ndjson", "bayescycle sample &mdash; fit of the simulated data"),
                (
                    "recovery_check.json",
                    "bayescycle recover-check &mdash; fit-vs-truth rank/interval facts",
                ),
            ],
        ),
        (
            "run-recover/",
            [("recovery.json", "bayescycle recover &mdash; single-scenario recovery facts")],
        ),
        (
            "run-sbc/",
            [("sbc.json", "bayescycle sbc &mdash; calibration rank facts over replicates")],
        ),
        (
            "run/",
            [
                ("model.ir.json", "bayescycle sample &mdash; canonical IR compiled from model.py"),
                ("data.json", "bayescycle sample &mdash; byte-for-byte copy of the input data"),
                ("dims.json", "bayescycle sample &mdash; jaxstanv5 dimension/coord metadata"),
                ("posterior.ndjson", "bayescycle sample &mdash; jaxstanv5/BlackJAX draws + stats"),
                (
                    "diagnostics.json",
                    "bayescycle diagnose &mdash; recomputed R-hat/ESS/sample-stats",
                ),
                (
                    "posterior_predictive.ndjson",
                    "bayescycle posterior-predictive &mdash; replicated y",
                ),
                (
                    "posterior_check.json",
                    "bayescycle posterior-check &mdash; observed-vs-replicated facts",
                ),
                ("fit.nc", "bayesite-idata &mdash; ArviZ InferenceData (NetCDF)"),
            ],
        ),
        ("viz/", [("*.png", "bayesite-viz &mdash; ArviZ diagnostic + prior-predictive plots")]),
    ]
    lines = []
    for directory, files in groups:
        lines.append(f'<div class="tree-dir mono">{esc(directory)}</div>')
        for name, origin in files:
            lines.append(
                '<div class="tree-row">'
                f'<span class="tree-file mono">{esc(name)}</span>'
                f'<span class="tree-origin">&larr; {origin}</span></div>'
            )
    return f'<div class="tree">{"".join(lines)}</div>'


def rank_bars(histogram: list[int], bins: int = 24) -> str:
    """Coarsen a rank histogram into a compact sparkline of bars."""
    if not histogram:
        return ""
    size = math.ceil(len(histogram) / bins)
    coarse = [sum(histogram[i : i + size]) for i in range(0, len(histogram), size)]
    peak = max(coarse) or 1
    cells = "".join(
        f'<span class="bar" style="height:{max(2, round(36 * c / peak))}px" title="{c}"></span>'
        for c in coarse
    )
    return f'<div class="bars">{cells}</div>'


def recovery_check_table() -> str:
    rows = ""
    for name in recovery_check["target_order"]:
        t = recovery_check["targets"][name]
        contains = recovery_check["interval_contains_truth_by_target"][name]
        mark = '<span class="ok">in</span>' if contains else '<span class="bad">out</span>'
        rows += (
            "<tr>"
            f'<td class="mono">{name}</td>'
            f"<td>{t['truth']:.3f}</td>"
            f"<td>{t['mean']:.4f}</td>"
            f"<td>{t['lower']:.4f}</td>"
            f"<td>{t['upper']:.4f}</td>"
            f"<td>{mark}</td>"
            f"<td>{t['rank']}/{t['rank_bounds']['max']}</td>"
            f"<td>{t['rhat']:.4f}</td>"
            f"<td>{t['ess']:.0f}</td>"
            "</tr>"
        )
    return (
        '<table class="data"><thead><tr>'
        f"<th>target</th><th>truth</th><th>post. mean</th>"
        f"<th>{int(recovery_check['interval'] * 100)}% lo</th>"
        f"<th>{int(recovery_check['interval'] * 100)}% hi</th>"
        "<th>truth?</th><th>rank</th><th>R-hat</th><th>ESS</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def recover_table() -> str:
    rows = ""
    for name in recover["parameter_order"]:
        p = recover["parameters"][name]
        contains = recover["interval_contains_truth_by_parameter"][name]
        mark = '<span class="ok">in</span>' if contains else '<span class="bad">out</span>'
        rows += (
            "<tr>"
            f'<td class="mono">{name}</td>'
            f"<td>{p['truth']:.4f}</td>"
            f"<td>{p['mean']:.4f}</td>"
            f"<td>{p['lower']:.4f}</td>"
            f"<td>{p['upper']:.4f}</td>"
            f"<td>{mark}</td>"
            f"<td>{p['rank']}/{p['rank_bounds']['max']}</td>"
            f"<td>{p['rhat']:.4f}</td>"
            f"<td>{p['ess']:.0f}</td>"
            "</tr>"
        )
    return (
        '<table class="data"><thead><tr>'
        "<th>param</th><th>prior truth</th><th>post. mean</th>"
        f"<th>{int(recover['interval'] * 100)}% lo</th><th>{int(recover['interval'] * 100)}% hi</th>"
        "<th>truth?</th><th>rank</th><th>R-hat</th><th>ESS</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def sbc_section_html() -> str:
    rows = ""
    for name in sbc["parameter_order"]:
        p = sbc["parameters"][name]
        rows += (
            '<div class="sbc-row">'
            f'<span class="sbc-name mono">{esc(name)}</span>'
            f"{rank_bars(p['rank_histogram'])}"
            f'<span class="sbc-meta">rank in 0..{p["rank_bounds"]["max"]}, '
            f"{sbc['replicate_count']} replicates</span>"
            "</div>"
        )
    return f'<div class="sbc">{rows}</div>'


def posterior_check_table() -> str:
    order = {"mean": 0, "sd": 1, "min": 2, "max": 3}
    checks = sorted(posterior_check["checks"], key=lambda c: order.get(c["statistic"], 9))
    rows = ""
    for c in checks:
        s = c["summary"]
        total = s["replicated_draw_count"]
        le = s["count_replicated_less_equal_observed"]
        rows += (
            "<tr>"
            f'<td class="mono">{esc(c["statistic"])}</td>'
            f"<td>{s['observed']:.4f}</td>"
            f"<td>{s['replicated_mean']:.4f}</td>"
            f"<td>{s['replicated_min']:.4f}</td>"
            f"<td>{s['replicated_max']:.4f}</td>"
            f"<td>{le}/{total}</td>"
            "</tr>"
        )
    return (
        '<table class="data"><thead><tr>'
        "<th>statistic</th><th>observed</th><th>replicated mean</th>"
        "<th>repl. min</th><th>repl. max</th><th>#rep&le;obs</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


def param_table() -> str:
    rows = ""
    for name in ("alpha", "beta", "sigma"):
        s = posterior_summary[name]
        rows += (
            "<tr>"
            f'<td class="mono">{name}</td>'
            f"<td>{TRUTH[name]:.3f}</td>"
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
        '<table class="data"><thead><tr>'
        "<th>param</th><th>true</th><th>mean</th><th>sd</th>"
        "<th>2.5%</th><th>median</th><th>97.5%</th><th>R-hat</th><th>ESS</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Assemble HTML.
# ---------------------------------------------------------------------------
def section(num: str, title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="lead">{subtitle}</p>' if subtitle else ""
    return (
        f'<section class="step"><h2><span class="num">{num}</span>{esc(title)}</h2>'
        f"{sub}{body}</section>"
    )


def divider(label: str) -> str:
    return f'<div class="divider"><span>{esc(label)}</span></div>'


parts: list[str] = []

parts.append(
    section(
        "",
        "Overview",
        """
<p>This page runs the <strong>complete bayescycle workflow</strong> end-to-end on a
Bayesian linear regression (<span class="mono">n = 1000</span>): first the
<strong>simulation gate</strong> &mdash; prior predictive, fake-data simulation,
recovery, single-scenario recover, and SBC &mdash; then the <strong>real fit</strong>
&mdash; sample, diagnose, posterior predictive, and posterior check &mdash;
finishing with ArviZ visualization.</p>
<p>The architectural invariant: every command is a typed state transition that
writes the same <strong>bayescycle run-directory v0 contract</strong>. The model is
authored in <span class="mono">jaxstanv5</span>; <span class="mono">bayescycle</span>
owns run-directory paths and artifact serialization and dispatches to a backend;
Bayesite and bayesite-viz consume the artifacts. Backends implement narrow
operation capabilities &mdash; the in-process <span class="mono">jaxstanv5</span>
backend serves <span class="mono">sample</span> and
<span class="mono">prior-predictive</span>; the remaining workflow verbs are
served by Bayesite.</p>
<div class="pipeline">
  <span>model.py</span><span class="arr">&rarr;</span>
  <span>prior&nbsp;predictive<br><small>simulate &middot; recover &middot; sbc</small></span><span class="arr">&rarr;</span>
  <span>sample<br><small>--backend jaxstanv5</small></span><span class="arr">&rarr;</span>
  <span>diagnose<br>posterior&nbsp;check</span><span class="arr">&rarr;</span>
  <span>bayesite-idata</span><span class="arr">&rarr;</span>
  <span>bayesite-viz<br><small>ArviZ plots</small></span>
</div>

<h3 class="io-title">Filesystem effects &mdash; what each command reads and writes</h3>
<p class="lead">Each model-level command owns a fresh run directory; follow-up
commands operate on an existing one. The engine <span class="mono">--out</span>
flag is reserved so artifacts always land at standard paths.</p>
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
        + code_block(model_src, "python")
        + '<p class="lead">Compiled to canonical IR on every run (truncated):</p>'
        + file_chip("write", "run/model.ir.json", "compiled from model.py")
        + code_block(ir_pretty, "json")
        + file_chip("write", "run/dims.json", "jaxstanv5 dimension metadata")
        + json_block(dims_obj),
        "A flat linear regression with scalar intercept, slope, and a positive scale.",
    )
)

parts.append(
    section(
        "2",
        "Synthetic data (n = 1000)",
        file_chip("input", "data.json", "summary shown; full vectors on disk")
        + json_block(data_summary),
        "Generated from known parameters with Gaussian noise so recovery is checkable.",
    )
)

parts.append(divider("Simulation gate"))

parts.append(
    section(
        "3",
        "Prior predictive",
        code_block(cmd_prior, "bash")
        + file_chip("input", "inputs.json", "declared inputs only (observed y omitted)")
        + file_chip("write", "run-prior/prior_predictive.ndjson", "header + 400 draws + trailer")
        + '<p class="lead">Artifact summary (parameter + observed sites, with prior-implied ranges):</p>'
        + file_chip("derived", "run-prior/prior_predictive.ndjson", "computed from draws")
        + json_block(pp_summary)
        + '<p class="lead">First draw record (truncated):</p>'
        + json_block(pp_first_display)
        + (
            '<p class="lead">Prior predictive draws of <span class="mono">y</span> imply this data range before seeing observations:</p>'
            + plot_figure(
                prior_plot,
                "Prior predictive",
                "Left: 40 prior predictive draws of y vs x. Right: the prior predictive marginal of y.",
            )
            if prior_plot.is_file()
            else ""
        )
        + '<p class="lead">The same command runs on the Bayesite backend; the in-process path serves it via <span class="mono">jaxstanv5.simulation.simulate_prior_predictive</span>:</p>'
        + code_block(cmd_prior_bayesite, "bash"),
        "Draw parameters and observed values from the prior to check prior-implied data.",
    )
)

parts.append(
    section(
        "4",
        "Simulate fake data from truth",
        code_block(cmd_sim, "bash")
        + file_chip("input", "truth.json", "constrained free-value truth")
        + file_chip(
            "write", "run-sim/simulated_data.json", "declared inputs + generated observed y"
        )
        + json_block(sim_summary),
        "Fix a known truth and simulate observed data the estimator must recover.",
    )
)

parts.append(
    section(
        "5",
        "Recovery fit + recover-check",
        code_block(cmd_recfit, "bash")
        + file_chip(
            "read",
            "run-sim/simulated_data.json",
            "typed data document, consumed natively by Bayesite",
        )
        + file_chip(
            "write",
            "run-recover-fit/posterior.ndjson",
            f"{rec_fit_header['chain_count']} chains, fit of fake data",
        )
        + code_block(cmd_reccheck, "bash")
        + file_chip(
            "write", "run-recover-fit/recovery_check.json", "fit-vs-truth rank/interval facts"
        )
        + recovery_check_table()
        + '<p class="lead small">Factual report: equal-tailed interval containment and ranks, no pass/fail verdict. Here the tight n = 1000 posterior places two of three truths just outside the 80% interval &mdash; exactly the kind of fact the human gate inspects.</p>',
        "Fit the simulated data, then compare the posterior back to the known truth.",
    )
)

parts.append(
    section(
        "6",
        "Single-scenario recover",
        code_block(cmd_recover, "bash")
        + file_chip("input", "recover_scenario.json", "declared data + sampler settings + seed")
        + file_chip(
            "write",
            "run-recover/recovery.json",
            "prior &rarr; simulate &rarr; fit &rarr; rank, in one report",
        )
        + recover_table()
        + f'<p class="lead small">One self-contained simulation: the engine draws a truth from the prior, simulates data, fits it, and ranks the truth. '
        f"Divergences across {recover['sampler_summary']['chain_count']} chains: "
        f"<strong>{recover['sampler_summary']['total_divergences']}</strong>.</p>",
        "A self-contained recovery scenario the engine runs in one shot.",
    )
)

parts.append(
    section(
        "7",
        "Simulation-based calibration (SBC)",
        code_block(cmd_sbc, "bash")
        + file_chip(
            "input", "sbc_scenario.json", "declared data + sampler settings + replicate count"
        )
        + file_chip(
            "write", "run-sbc/sbc.json", f"rank facts over {sbc['replicate_count']} replicates"
        )
        + sbc_section_html()
        + '<p class="lead small">Each bar is a coarsened rank histogram across replicates. SBC reports ranks and histograms only &mdash; uniformity is for the consumer to judge, not an engine verdict.</p>',
        "Repeat prior &rarr; simulate &rarr; fit &rarr; rank to calibrate the estimator.",
    )
)

parts.append(divider("Real fit"))

parts.append(
    section(
        "8",
        "Sample in-process (jaxstanv5 / BlackJAX)",
        code_block(cmd_sample, "bash")
        + file_chip("write", "run/posterior.ndjson", "header + 2000 draws + trailer")
        + '<p class="lead">Posterior stream header (per_draw_v2, neutral model/data fingerprint):</p>'
        + json_block(post_header)
        + '<p class="lead">First draw record (parameter values + per-draw sampler stats):</p>'
        + json_block(post_first)
        + '<p class="lead">Trailer (per-chain step sizes, tree-depth histograms, R-hat, ESS):</p>'
        + json_block(post_trailer),
        "4 chains &times; 500 draws &rarr; 2000 posterior draws on the real data.",
    )
)

parts.append(
    section(
        "9",
        "Posterior summary & recovery",
        file_chip("derived", "run/posterior.ndjson", "summary computed from draws")
        + param_table()
        + f'<p class="lead">Divergences across all chains: <strong>{divergences}</strong>. '
        f"Energy range: <strong>{min(energies):.1f}</strong> &ndash; <strong>{max(energies):.1f}</strong>.</p>",
        "All three parameters are recovered tightly around their true values.",
    )
)

parts.append(
    section(
        "10",
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
        "11",
        "Posterior predictive",
        code_block(cmd_ppc, "bash")
        + file_chip("read", "run/model.ir.json, run/data.json, run/posterior.ndjson", "inputs")
        + file_chip("write", "run/posterior_predictive.ndjson", "header + 2000 draws + trailer")
        + json_block(ppc_summary),
        "2000 replicated datasets of 1000 y-values each, conditioned on the fit.",
    )
)

parts.append(
    section(
        "12",
        "Posterior check",
        code_block(cmd_pcheck, "bash")
        + file_chip("write", "run/posterior_check.json", "observed-vs-replicated discrepancy facts")
        + posterior_check_table()
        + '<p class="lead small">Built-in discrepancy summaries with tail counts, not a model-fit verdict. Observed statistics sit near the centre of the replicated distribution.</p>',
        "Compare observed summaries to posterior-predictive replicates.",
    )
)

parts.append(
    section(
        "13",
        "ArviZ visualizations",
        code_block(cmd_export, "bash")
        + file_chip("write", "run/fit.nc", "ArviZ InferenceData (NetCDF)")
        + file_chip("write", "viz/*.png", "one PNG per bayesite-viz verb")
        + f'<div class="plots">{arviz_section}</div>',
        "Export the run directory to NetCDF and render ArviZ plots via bayesite-viz.",
    )
)

parts.append(
    section(
        "",
        "Backend capability matrix",
        """
<p>Backends implement narrow per-operation capabilities rather than one fat
interface. Unsupported combinations fail explicitly during preparation instead
of falling back silently.</p>
<table class="data caps"><thead><tr>
<th>command</th><th>Bayesite</th><th>jaxstanv5 in-process</th></tr></thead><tbody>
<tr><td class="mono">sample</td><td class="ok">yes</td><td class="ok">yes</td></tr>
<tr><td class="mono">prior-predictive</td><td class="ok">yes</td><td class="ok">yes</td></tr>
<tr><td class="mono">simulate</td><td class="ok">yes</td><td class="bad">unsupported</td></tr>
<tr><td class="mono">recover</td><td class="ok">yes</td><td class="bad">unsupported</td></tr>
<tr><td class="mono">sbc</td><td class="ok">yes</td><td class="bad">unsupported</td></tr>
<tr><td class="mono">diagnose</td><td class="ok">yes</td><td class="bad">unsupported</td></tr>
<tr><td class="mono">posterior-predictive</td><td class="ok">yes</td><td class="bad">unsupported</td></tr>
<tr><td class="mono">posterior-check</td><td class="ok">yes</td><td class="bad">unsupported</td></tr>
<tr><td class="mono">recover-check</td><td class="ok">yes</td><td class="bad">unsupported</td></tr>
</tbody></table>
<p class="small">All workflow reports are v0-provisional: machine-readable and
tested, but consumers must check the format marker before depending on field
stability.</p>
""",
    )
)

body = "\n".join(parts)

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>bayescycle &mdash; complete workflow walkthrough</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #171a21; --panel2: #1e222b; --ink: #e6e8ee;
    --muted: #9aa3b2; --accent: #6ea8fe; --accent2: #7ee787; --bad: #ff7b72;
    --border: #2a2f3a; --code: #11141a;
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
  section.step h2 {{ margin: 0 0 4px; font-size: 21px; display: flex; align-items: center; gap: 12px; }}
  .num {{
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 30px; height: 30px; padding: 0 8px; border-radius: 8px;
    background: var(--accent); color: #0b1020; font-weight: 700; font-size: 15px;
  }}
  .lead {{ color: var(--muted); margin: 8px 0 12px; }}
  .small {{ font-size: 12.5px; color: var(--muted); }}
  pre.code {{
    background: var(--code); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.5; margin: 10px 0;
  }}
  pre.code code {{ font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; color: #d7dce6; white-space: pre; }}
  pre.code[data-lang]::before {{
    content: attr(data-lang); display: block; color: var(--muted);
    font-size: 11px; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 8px;
  }}
  .mono {{ font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 14px; }}
  table.data th, table.data td {{ border: 1px solid var(--border); padding: 7px 10px; text-align: right; }}
  table.data th:first-child, table.data td:first-child {{ text-align: left; }}
  table.data thead th {{ background: var(--panel2); color: var(--ink); }}
  table.data td {{ color: #cfd6e4; }}
  table.caps td {{ text-align: left; }}
  .ok {{ color: var(--accent2); font-weight: 600; }}
  .bad {{ color: var(--bad); font-weight: 600; }}
  .divider {{
    display: flex; align-items: center; gap: 14px; margin: 30px 4px 6px; color: var(--accent);
    font-size: 13px; letter-spacing: .16em; text-transform: uppercase; font-weight: 700;
  }}
  .divider::before, .divider::after {{ content: ""; height: 1px; background: var(--border); flex: 1; }}
  .pipeline {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: center; margin: 18px 0 4px; }}
  .pipeline span {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 9px; padding: 8px 12px; font-size: 13px; text-align: center; }}
  .pipeline .arr {{ background: none; border: none; color: var(--accent); font-size: 18px; padding: 0 2px; }}
  .pipeline small {{ color: var(--muted); }}
  .plots {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin-top: 12px; }}
  figure.plot {{ margin: 0; background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 14px; overflow: hidden; }}
  figure.plot h3 {{ margin: 0 0 8px; font-size: 15px; color: #1a1d24; }}
  figure.plot img {{ width: 100%; height: auto; display: block; }}
  figure.plot figcaption {{ margin-top: 8px; font-size: 13px; color: #51596b; }}
  footer {{ text-align: center; color: var(--muted); padding: 28px; font-size: 13px; }}
  .io-title {{ margin: 22px 0 4px; font-size: 16px; }}
  .filechip {{ display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 12px 0 -2px; padding: 5px 10px; border-radius: 8px; border: 1px solid var(--border); background: var(--panel2); font-size: 12px; }}
  .chip-dir {{ font-weight: 700; font-size: 10.5px; letter-spacing: .1em; padding: 2px 7px; border-radius: 6px; color: #0b1020; }}
  .chip-path {{ color: var(--ink); font-size: 12.5px; }}
  .chip-note {{ color: var(--muted); font-size: 11.5px; }}
  .chip-write .chip-dir {{ background: var(--accent2); }}
  .chip-read .chip-dir {{ background: var(--accent); }}
  .chip-input .chip-dir {{ background: #f0b562; }}
  .chip-stdout .chip-dir {{ background: #c9a0ff; }}
  .chip-derived .chip-dir {{ background: #6fd6c9; }}
  .chip-write {{ border-color: #2f5d3f; }} .chip-read {{ border-color: #2f4a6d; }}
  .chip-input {{ border-color: #6b5326; }} .chip-stdout {{ border-color: #4d3f6b; }}
  .chip-derived {{ border-color: #2c5d57; }}
  table.io-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  table.io-table th, table.io-table td {{ border: 1px solid var(--border); padding: 9px 11px; text-align: left; vertical-align: top; }}
  table.io-table thead th {{ background: var(--panel2); }}
  table.io-table .io-cmd {{ font-size: 12.5px; max-width: 320px; }}
  table.io-table .io-note {{ color: var(--muted); font-size: 11px; margin-top: 3px; }}
  .f {{ display: inline-block; margin: 2px 4px 2px 0; padding: 3px 7px; border-radius: 6px; font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; font-size: 11.5px; border: 1px solid var(--border); }}
  .f-read {{ background: #16233a; color: #bcd3f5; }} .f-write {{ background: #16321f; color: #b8e8c6; }}
  .tree {{ background: var(--code); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin: 12px 0; font-size: 13px; }}
  .tree-dir {{ color: var(--accent); font-weight: 700; margin: 8px 0 2px; }}
  .tree-row {{ display: flex; gap: 10px; flex-wrap: wrap; padding: 2px 0 2px 18px; }}
  .tree-file {{ color: #b8e8c6; min-width: 220px; }}
  .tree-origin {{ color: var(--muted); font-size: 12px; }}
  .sbc {{ display: flex; flex-direction: column; gap: 10px; margin: 8px 0; }}
  .sbc-row {{ display: flex; align-items: flex-end; gap: 12px; }}
  .sbc-name {{ min-width: 56px; color: var(--ink); }}
  .sbc-meta {{ color: var(--muted); font-size: 12px; }}
  .bars {{ display: flex; align-items: flex-end; gap: 2px; height: 40px; padding: 2px 6px; background: var(--code); border: 1px solid var(--border); border-radius: 8px; }}
  .bars .bar {{ width: 7px; background: linear-gradient(180deg, var(--accent), #2f4a6d); border-radius: 2px 2px 0 0; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<header class="hero">
  <h1>bayescycle &mdash; complete workflow walkthrough</h1>
  <p>Simulation gate &rarr; real fit &rarr; ArviZ diagnostics, through the run-directory v0 contract</p>
  <div class="badges">
    <span class="badge">prior-predictive &middot; simulate &middot; recover &middot; sbc</span>
    <span class="badge">sample --backend jaxstanv5</span>
    <span class="badge">diagnose &middot; posterior-check</span>
    <span class="badge">n = 1000</span>
    <span class="badge">BlackJAX NUTS</span>
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
