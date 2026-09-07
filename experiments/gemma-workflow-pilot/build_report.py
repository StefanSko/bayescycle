"""Small offline evidence viewer; intentionally excludes raw model thinking."""

# Embedded HTML/CSS retains its own source layout.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def esc(value: object) -> str:
    return html.escape(str(value))


def main() -> None:
    assessment = json.loads((ROOT / "evaluator/assessment.json").read_text())
    rows = []
    for key, label in (
        ("a-numpyro", "NumPyro + ArviZ"),
        ("b-bayesjax", "Direct Bayesjax"),
        ("c-bayescycle", "Bayescycle CLI"),
    ):
        session = assessment["arms"][key]["sessions"]["initial"]
        seconds = round(session["wall_seconds"])
        rows.append(
            f"<tr><td>{label}</td><td>{seconds // 60}m {seconds % 60:02d}s</td>"
            f"<td>{session['tool_calls']}</td>"
            f"<td>{session['max_input_tokens_reported']:,}</td>"
            "<td>Length stop; no model or fit</td></tr>"
        )
    evidence = []
    for title, path in (
        ("Full findings and limitations", "README.md"),
        ("Original replication protocol", "PROTOCOL.md"),
        ("Exploratory authoring protocol", "AUTHORING-PROTOCOL.md"),
        ("Shared compact task", "AUTHORING-BRIEF.md"),
        ("NumPyro API card", "a-model/API.md"),
        ("Bayeswire API card", "b-model/API.md"),
        ("Gemma's actual NumPyro model", "a-model/model.py"),
        ("Evaluator's numerical validation", "evaluator/a-model/validation/result.json"),
        (
            "Exact retained-array comparison",
            "evaluator/a-model/validation/exact-reference-comparison.json",
        ),
        ("Session outcomes and errors (no thinking)", "evaluator/assessment.json"),
    ):
        evidence.append(
            f"<details><summary>{esc(title)}</summary><p class='path'>{esc(path)}</p>"
            f"<pre>{esc((ROOT / path).read_text())}</pre></details>"
        )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Gemma: Bayesian workflow pilot</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f5f0;color:#242a29;font:17px/1.6 system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:48px 24px}h1{font-size:clamp(32px,5vw,52px);line-height:1.1}
h2{margin-top:36px;font-size:25px}.eyebrow,.path{font-size:13px;color:#53605b}.lede{font-size:21px}
.box{padding:20px 24px;background:white;border:1px solid #d8ddd8;border-radius:12px;margin:22px 0}
.warning{border-left:5px solid #a75020}.success{border-left:5px solid #24785b}.table{overflow:auto}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}
summary{cursor:pointer;padding:16px 0;font-weight:650}details{border-top:1px solid #ced5ce}
pre{font:13px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:20px}
small{color:#53605b}code{overflow-wrap:anywhere}footer{margin:40px 0;color:#53605b;font-size:14px}
</style><main>
<p class="eyebrow">LOCAL MODEL FOLLOW-UP · 4 SEPTEMBER 2026 · EXPLORATORY, ONE ATTEMPT PER ARM</p>
<h1>Where the local agent got stuck.</h1>
<p class="lede">Does a narrower modeling interface help a smaller local agent do reliable Bayesian work?</p>
<div class="box warning"><strong>Not demonstrated in this test.</strong> All three full-study attempts
stopped before producing code. In a smaller follow-up, NumPyro produced a correct model;
Bayeswire did not deliver one within the fixed budget. These are agent-execution failures,
not evidence of incorrect Bayesian inference code.</div>
<p><code>ollama/gemma4-pi:12b</code> · existing Q4_K_M model · 32K context · 4096 response tokens.
No downloads or configuration tuning. Actual read-tool integration check passed.</p>
<h2>1. Full study: 0 of 3 completed</h2>
<p>Same study fixture as the hosted pilot; 15 minutes per arm, sequential A → C → B.
Raw numeric JSON consumed large contexts. No scientific API was actually used, and no
revision or handoff was attempted.</p>
<div class="box table"><table><thead><tr><th>Setup</th><th>Time</th><th>Calls</th>
<th>Max input tokens</th><th>Outcome</th></tr></thead><tbody>ROWS</tbody></table></div>
<p><small>Failed-session duration is not a speed ranking. No compaction-start event was observed.
Near-context-limit length stops do not identify every internal Ollama/Pi cause.</small></p>
<h2>2. Compact authoring probe: 1 of 2 delivered</h2>
<p>A separate, openly post-hoc probe: one model file, compact data shapes, short API cards,
read/write/edit tools, four minutes each. No raw arrays, plotting or workflow glue.</p>
<div class="box success"><strong>NumPyro: correct model, 3m 19s.</strong>
<p>The <em>evaluator</em> then fitted Gemma's unchanged model. Every retained parameter array
exactly matched the hosted NumPyro reference.</p>
<p>β = <strong>0.588040</strong>; 95% interval [0.472937, 0.698431].<br>
Max rank Rhat 1.0031 · min bulk/tail ESS 707/731 · zero retained divergences.</p>
<small>Gemma authored the model. It did not perform this fitting or diagnostic validation.</small></div>
<div class="box warning"><strong>Bayeswire: no model after four minutes.</strong>
<p>After reading the brief and API card, it reached a response-token limit while attempting
<code>read(path=".")</code>. Pi refused the potentially truncated call. The deadline then ended
the session. Its max reported input was only 2,596 tokens—not the large-context pattern above.</p>
<small>No model existed to compile. The evaluator did not supply or repair one.</small></div>
<h2>What to keep from this</h2>
<p>Focus next on <strong>small agent tasks, compact evidence and executable checks</strong>,
not merely swapping the LLM provider. This is consistent with valuing explicit phases;
it does not establish a need for five distributions or multiple process boundaries.</p>
<p>One attempt per interface does not establish an error rate. Familiarity, API cards,
stochastic decoding and budgets matter. The compact probe changed several things at once.
Human scientific control remains untested.</p>
<h2>Inspect the evidence</h2><p>Expand any item. Everything below is embedded in this offline file.</p>
EVIDENCE
<footer>Analysis agents: local Ollama. Preparation/evaluation/reporting: current hosted assistant.
No hosted repairs to agent implementations. Full raw event/session logs remain local and are
not embedded here. The original hosted pilot report is unchanged.</footer>
</main></html>
"""
    document = document.replace("ROWS", "\n".join(rows)).replace("EVIDENCE", "\n".join(evidence))
    (ROOT / "report.html").write_text(document)
    print(f"Wrote {ROOT / 'report.html'} ({len(document.encode()):,} bytes)")


if __name__ == "__main__":
    main()
