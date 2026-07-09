# Implementation notes — VectorBounds / censored PartiallyObserved (#45/#46)

Working log of non-obvious facts discovered while implementing the feature.
Newest entries at the bottom of each section. Written for whoever touches
this code next; safe to drop before merge if unwanted.

## Design provenance (pre-implementation discussion, 2026-07-09)

- **Both issues assumed the bound rides as a field on `VectorScatterOp`.**
  That placement would have been a *breaking* wire change: the codec always
  emits every registered field and the decoder strict-matches field sets
  (`ir.py:272-279`), so "optional field" still changes canonical bytes of
  every existing `VectorScatterOp` document. Issue #45's "byte-identical"
  acceptance criterion was unsatisfiable under its own proposal.
- **Stan settled the seam question.** The censored-imputation idiom the
  issues cite (`vector<lower=c>[N] t;`) declares bounds on the *parameter*,
  not the sampling statement, and Stan constraints reference data as a
  matter of course. Hence `VectorBounds` as a constraint node on the free
  value — a purely *additive* tag change, keeping `bayeswire_ir` at 1 and
  every existing corpus document byte-identical.
- **Prior-predictive insight:** prior predictive of a PO site is "PO with no
  observables" — draw the full vector from the base distribution; the
  observed/missing split doesn't exist before conditioning. The blanket
  `TypeError` in `simulation/core.py` is a missing dispatch rule, not
  missing math. Scoped in: scalar iid bases + unbounded MVN. Scoped out:
  bounded MVN (per-coordinate truncation of a correlated joint has no
  closed form).
- **Exponential memorylessness bonus:** an exponential truncated to
  `t > c` is exactly `c + Exponential(rate)` — bounded prior draws for the
  black-cat model are a sample-and-shift, no inverse-CDF machinery needed.
- Decision log with diagrams:
  https://claude.ai/code/artifact/862c9a2a-06e2-4ac9-9184-fb4261b77c9a

## Codebase facts that shaped the plan

- `_field_kind` (`_ir_registry.py:130-136`) classifies any non-dict/tuple
  hint as `VALUE`, so `DataRef | None` on a constraint node serializes via
  the existing nullable-value rule (spec rule 6, the `ResolvedFreeValue.size`
  pattern) with zero codec changes. The feared "constraints can only hold
  floats" limitation was a convention, not a mechanism.
- **Produce-conformance forces the stage order.** Every case in
  `reference_models.py` must ship a JAX-oracle evaluation fixture
  (`test_produce.py:91-100`), and only bayesjax can generate those — so new
  reference models/corpus entries must land *after* the backend transform
  work, not with the bayeswire surface. Stage order: surface → transforms →
  prior-pred/tests → fixtures+corpus.
- **Scalar-base PO already worked.** Sites sum an elementwise logpdf
  (`compiler/core.py:213`), so `PartiallyObserved(Exponential(rate), ...)`
  had iid vector semantics all along; the corpus only ever exercised MVN.
  The feared prerequisite work item didn't exist.
- Strict-decoder blast radius is small in practice: the only runtime
  consumers of IR documents at rest are the pinned Rust engine and the
  corpus tests; `bayesite-idata` deliberately reads `model.ir.json` with
  tolerant ad-hoc helpers ("never a runtime dependency" on bayeswire).
- Stale pre-monorepo references in the issues: `src/jaxstanv5/...` paths
  are now `packages/bayesjax/src/bayesjax/...`; "bump the bayeswire pin" is
  obsolete (workspace dependency, lockstep versions).

## Stage 1 — bayeswire surface (VectorBounds node, eDSL, decorator, spec)

- Red→green evidence (Pi run): ImportError → 4 passed (construction);
  `UnserializableValue` → 1 passed (round-trip); unexpected-kwarg → 5 passed
  (decorator/validation). Full suite 220 passed.
- `regenerate_corpus.py` after registering the tag changed *only*
  `spec/ir-v1-tags.md` (+1 row); zero corpus JSON diffs — empirical
  confirmation the change is additive.
- Bound validation reuses `_validate_exact_vector_data` and additionally
  requires the bound's length *dimension object* to be the same as
  `missing_idx`'s (`_exact_vector_dim` comparison) — stronger than "same
  rank", it pins the shared `n_mis` symbol at declaration time.
- Unbounded PO still resolves `constraint=None` — byte-identity for
  existing models is preserved at the resolution level, not just the codec
  level.
- Inline-expression PO references reuse the declaration branch for free
  values, so the constraint is attached exactly once regardless of how the
  PO is referenced.
- Pre-existing, unrelated: `ty check` reports 2 `unsupported-base` warnings
  in `tests/unit/model/test_dimensions.py` and `tests/unit/test_public_hooks.py`
  (present on main; not introduced here).

## Stage 2 — bayesjax transforms, bind resolution, validation

- **A dispatch site the plan missed:** `_constrain_sample_values` in
  `inference/core.py` maps posterior draws back to constrained space.
  Without handling VectorBounds there, sampling would have *worked* but
  reported the bounded coordinates in unconstrained space — a silent
  wrong-answer bug. Found by the "audit every isinstance-on-constraint
  site" discipline, not by a test failing.
- Phase boundary implemented as planned: `ResolvedVectorBounds` (backend-
  local frozen dataclass holding concrete arrays) is constructed at bind
  time; `BoundModel.vector_bounds: Mapping[str, ResolvedVectorBounds]`
  (default empty — existing bind paths unaffected). The wire-level
  `VectorBounds` (DataRefs) never reaches jitted code.
- Interval branch log|J| uses the softplus identity
  `log σ(u) = −softplus(−u)`, mirroring the existing Interval constraint.
- Support-compatibility validation needs to know which distribution
  consumes a free value; there is no back-pointer, so binding walks the
  stochastic-site expression trees (`_distribution_for_free_value`).
  Support checks cover Exponential/HalfNormal (lower ≥ 0), Beta ([0,1]),
  Uniform (within low/high); Normal/MVN unrestricted by design.
- Bound-referenced data vectors are excluded from the generic finite-data
  check and re-validated in bound resolution so errors name the free value
  ("VectorBounds for free value 'y' ... must contain only finite values").
- The interval-censored path (both bounds, sigmoid transform) is exercised
  end-to-end at the logp level with a Normal base, not just in unit tests.
- Full suite: 433 passed (~90 s wall, JAX compile dominated).

## Stage 3 — prior-predictive, statistical validation, CmdStan reference

- Prior-predictive PO rule implemented as "full-vector draw, then overwrite
  missing coordinates with truncated draws" (`.at[missing_idx].set(...)`) —
  observed slots draw unrestricted (prior-predictively they *are* the data
  being predicted), missing slots respect their bounds.
- `ScalarIntervalDomain` already had `DistributionValue | None` bounds, and
  `jax.random.uniform(minval=cdf(lo), maxval=cdf(hi))` broadcasts arrays —
  so per-coordinate truncated inverse-CDF sampling needed no new machinery.
  The Exponential lower-bound case short-circuits via memorylessness
  (`c + Exp(rate)` draw), skipping cdf/icdf entirely.
- **Statistical validation numbers** (seeded, deterministic):
  - Conjugate check: censored exponential with Exponential(1) prior ⇒
    analytic posterior Gamma(28.0, 13.636), mean 2.0534. NUTS: mean 2.0611
    (z = 1.20 vs MCSE), variance z = 1.19, R-hat 1.0005, ESS 3789,
    0 divergences. The censored contribution enters the Gamma rate as
    Σc_i — direct confirmation of the CCDF marginalization semantics.
  - Bias recovery: true rate 2.0 → censored fit 2.056 (|err| 0.056);
    complete-case fit 5.016 (|err| 3.02, the McElreath ~2.5× upward bias).
  - Prior-predictive memorylessness: bounded coordinate means match
    c_i + 1/rate within MC tolerance; unbounded match 1/rate.
- The CmdStan reference model uses `vector<lower=missing_lower>[N_mis]` —
  Stan's data-dependent parameter bound, i.e. the exact construct
  VectorBounds was modeled on. Script is manual (requires cmdstan), not in
  pytest; only py_compile-checked in CI-able runs.
- Full suite: 438 passed.

## Stage 4 — golden models, oracle fixtures, corpus

- Two new corpus models: `censored_exponential` (lower bounds, the wire
  path the black cat needs) and `interval_censored_normal` (both bounds,
  pinning the upper-bound encoding and the lub transform). Deliberate
  simplification vs issue #45's "two rates" sketch: golden models pin
  bytes, the statistical story lives in the integration suite.
- **Byte-identity gate held:** `git diff` on the corpus shows only new
  files plus strictly-appended entries in `hashes.json` /
  `fingerprints.json` — every pre-existing corpus document byte-identical,
  exactly what the additive-tag design promised.
- `VectorBounds` on the wire is the nullable-value pattern:
  `{"node":"VectorBounds","lower":{"node":"DataRef",...},"upper":null}`.
- Conformance both directions green: produce (bayeswire, 228) and consume
  (bayesjax evaluates the new fixtures' float64 oracle logp/gradient within
  spec tolerance, 9 fixture tests). Root workspace guards 13 passed.
- Oracle fixture values (spot record): censored_exponential eval[0]
  logp = -9.25; interval_censored_normal eval[0] logp = -8.9765.

## Codex review round 1 (PR #55)

- Both findings were real edge-case bugs in the bind-time support
  validation, and both reproduced red before fixing:
  1. Vectorized `Uniform(low, high)` bases (length-n support arrays) were
     compared directly against length-n_mis bound arrays — broadcast crash,
     or wrong-coordinate comparison if n happened to equal n_mis. Fixed by
     threading the site's evaluated `missing_idx` into support validation
     and gathering non-scalar supports (`support[missing_idx]`) first.
  2. Degenerate one-sided bounds passed validation (e.g. Beta base with
     `missing_lower=[1.0]`): the transform then maps every draw outside the
     support and logp is -inf everywhere — a zero-mass model accepted
     silently. Fixed with strict opposite-edge checks: lower bounds must be
     `< support_upper`, upper bounds `> support_lower`.
- Lesson recorded: the original support checks validated each bound against
  its *own* edge but not the *opposite* edge, and assumed scalar supports.
  Both are instances of "validation written from the happy-path example."
- Post-fix suite: 444 passed.
