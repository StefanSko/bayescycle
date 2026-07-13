# Playground safe boundary red-team report

- Timestamp:
- Reviewer model/session:
- Commit:
- Worktree baseline/final:
- Playground origin:
- External sink origin:
- Synthetic canary:
- Evidence root:

## Verdict

`CLEAN` / `IN-SCOPE ESCAPE(S)` / `INCOMPLETE`

## Probe ledger

| ID | Classification | Invariant clause | Evidence | Notes |
|---|---|---|---|---|
| C1 | UNTESTED | Disposable compiler 3 | | |
| N1 | UNTESTED | Disposable compiler 5 | | |
| M1 | UNTESTED | Disposable compiler 10; runtime 7 | | |
| P1 | UNTESTED | Disposable compiler 1, 4, 8 | | |
| L1 | UNTESTED | Disposable compiler 2, 6 | | |
| L2 | UNTESTED | Disposable compiler 2, 10 | | |
| L3 | UNTESTED | Disposable compiler 2, 6 | | |
| L4 | UNTESTED | Disposable compiler 2 | | |
| H1 | UNTESTED | Disposable compiler 9 | | |
| O1 | UNTESTED | Disposable compiler 6, 10 | | |
| E1 | UNTESTED | Disposable compiler 11; runtime 3 | | |
| E2 | UNTESTED | Runtime 2, 3 | | |
| S1 | UNTESTED | Trust model; browser constraints | | |
| R1 | UNTESTED | Runtime 7, 8 | | |

Allowed classifications are exactly `RESISTED`, `ESCAPED`,
`EXPECTED CAPABILITY`, and `UNTESTED`.

## Alleged escapes

For each alleged escape include:

- invariant clause;
- exact bounded reproduction;
- expected and actual result;
- synthetic payload/canary;
- evidence paths;
- confidence and alternative explanations.

## Explicit residual risks / expected capabilities

Record current-compile mutation, public same-origin static fetches, excluded
persistence behavior, and other explicit non-guarantees without calling them
escapes unless another in-scope clause is violated.

## UI/error observations

Visible, bounded, actionable error evidence only. Do not fix findings.

## Cleanup proof

- Rodney stopped:
- Browser/profile removed:
- Playground server stopped:
- External sink stopped:
- Temporary child processes checked:
- Final `git status --porcelain`:
- Final commit:
