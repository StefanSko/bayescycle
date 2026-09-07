# Scripted prior revision — fresh session

Read your arm's STUDY.md and saved initial artifacts, plus ../PROTOCOL.md.
Do not read other arms, ../evaluator/, ../prepare.py, or prior session logs.
Write only within this arm. No dependency/source/shared-input changes.
This is a fixed evaluator request, not human scientific approval.

Change ONLY beta's prior from Normal(0,1) to Normal(0,0.25). Preserve all completed
initial artifacts and the exact initial executable model source; do not overwrite
initial posterior draws or rewrite their result summaries. Keep provenance clear.
Use the same float64, chains/warmup/draws, target acceptance and seeds. Do not
retune in response to diagnostics. Identify which model-dependent results are
superseded and rerun prior prediction, recovery, and supplied-data fit under the
new prior as needed. Do not silently invoke a different backend.

Write the revised supplied-data samples to results/revised/posterior.npz and
summary to results/revised/result.json, using the same schema as initial.
Use distinct revised prior/recovery output directories. Record limitations,
actual command failures, diagnostics and plots. Update STUDY.md so a fresh agent
can tell which model/results are current proposed work, which are superseded,
and what human decisions remain. Never label a model or result human-approved.

Commands: `uv run --project .. python ...` or, for C, the Bayescycle CLI through
`uv run --project .. bayescycle ...`. No installations except the existing C
pinned visualization integration if required. You have 10 minutes. If initial
work is incomplete, do not pretend it succeeded; explain the blocker and retain
evidence rather than silently repair the baseline as part of revision scoring.
Finish with a concise factual summary.
