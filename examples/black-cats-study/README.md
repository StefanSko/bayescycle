# Black cats study seed

This is a minimal study directory for dogfooding the `bayescycle-study` skill.
It starts at the `estimand` phase with two blocking human questions:

1. What counts as a black cat?
2. How should transfer outcomes be interpreted?

Use from the repository root with a fresh agent invocation such as:

```text
/skill:bayescycle-study examples/black-cats-study phase=estimand
```

The `runs/` directory is intentionally not committed. Raw Bayesite NDJSON and
run outputs should live there and be summarized into registered artifacts before
approval.

Visualization is part of the gate: use `bayesite-idata` and `bayesite-viz` to
turn approved run directories into visual artifacts for prior predictive,
recovery, fit diagnostics, and posterior predictive critique.
