# File-only handoff — fresh agent

You have no previous conversation. Work from this arm's study files only.
Do NOT read other arms, ../evaluator/, ../prepare.py, or earlier agent logs.
Do not refit, regenerate simulations, modify existing artifacts, install packages,
or change the study/model. You may write HANDOFF.md and handoff-result.json only.
You have five minutes. Use `uv run --project .. python ...` for calculations.

1. Identify the scientific question, current proposed model and beta prior, and
   distinguish superseded from current results. Say whether anything is actually
   human-approved. Do not infer approval from a completed run or passing diagnostics.
2. Locate the current saved posterior draws and independently recompute beta's
   mean, SD, 2.5% and 97.5% quantiles, rank Rhat, bulk/tail ESS and MCSE(mean).
   Compare to the saved summary; do not merely copy it.
3. Check recorded input/model hashes against the exact saved files where possible.
   Be precise about what is and is not verifiable; a source hash is not a proof
   that arbitrary Python execution was deterministic.
4. Explain the relevant diagnostic caveats and the next human decision.

Write HANDOFF.md with the evidence paths and reproducible calculation command.
Write handoff-result.json with `current_beta_prior_sd`, `posterior_path`,
`human_approved` (boolean), `beta` numerical metrics, `summary_matches` (boolean),
and `provenance_status` (string). Stop without advancing the study.

LOCAL REPLICATION: Do not inspect experiments/workflow-value-pilot or any previous pilot's solutions, results, figures, reports or logs. Only its scientific environment is shared automatically through uv. Do not change the supplied UV_PROJECT_ENVIRONMENT or UV_NO_SYNC. The local PROTOCOL.md defines this run.
