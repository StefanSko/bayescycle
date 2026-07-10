# Implementation notes — VectorBounds / censored PartiallyObserved (#45/#46)

Working log of non-obvious facts discovered while implementing features.
Newest entries are appended for whoever touches these paths next.

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

## Codex review round 2 (PR #55) — P1: fold base support into one-sided bounds

- The deeper version of round 1's finding: a one-sided bound on a base with
  a finite *opposite* support edge (Beta + lower-only, Exponential +
  upper-only) left that edge unenforced — the one-sided transform maps a
  whole region of unconstrained space to -inf density (red evidence:
  `log_density([50.]) == -inf`). Validation alone can't fix this; the
  *transform choice* has to see the true support.
- Fix: bind-time **support folding** — finite base edges are folded into
  `ResolvedVectorBounds`, so a Beta + lower=0.8 site resolves to the
  interval (0.8, 1.0) and gets the lub transform. The folded pair then
  flows through the existing lower<upper validation, which now subsumes
  round 1's opposite-edge checks (Beta lower=1.0 folds to a degenerate
  interval and is rejected by the standard path).
- Numerics rider: at saturated sigmoid (|u| ≳ 20 in float32) the two-sided
  inverse transform hits the boundary *exactly* in floating point, and
  strict-support bases (Beta) are -inf at their endpoints. Standard guard
  applied: clip to one ULP inside via `nextafter` — a no-op at interior
  values (consume-conformance oracle values unchanged, verified).
- Corpus unaffected by folding: censored_exponential is lower-only on a
  base with an infinite upper edge (no fold); interval_censored_normal has
  a full-support base (no fold).
- Post-fix suite: 447 passed; consume-conformance 9 passed.

## Follow-up — deterministic VectorBounds owner resolution (#56, 2026-07-10)

- BayesJAX and Bayesite currently recover the distribution supplying implicit
  support edges by selecting the first stochastic-site value expression that
  references the constrained free value. That makes an unrelated earlier
  factor alter the unconstrained transform.
- The implementation invariant is narrower and structural: a VectorBounds
  free value has exactly one same-name owner site. The owner value is either a
  direct same-name `ParamRef` (generic vector parameter) or a `VectorScatterOp`
  whose `missing_values` is a direct same-name `ParamRef` (PartiallyObserved).
  Differently named factors remain free to score either the missing slot or
  the assembled vector and never participate in owner selection.
- This is a v1 semantic clarification, not an encoding change: no tag, field
  list, encoding rule, or existing canonical model bytes change. The planned
  adversarial corpus case is additive resolved metadata because the current
  declaration eDSL intentionally has no general Factor surface.
- First red-test attempt hit an unrelated validation boundary: an empty
  `jnp.asarray([])` defaults to float and is correctly rejected as index data.
  Pinning the empty observed index to integer dtype exposed the intended owner
  resolution failures instead of weakening index validation.
- Red evidence after correcting the fixture: an earlier differently named
  full-scatter Normal factor made the upper-only Exponential target evaluate to
  `-inf` at `q=1`, while missing, duplicate, and expression-valued same-name
  owners were all silently accepted. The same-name/direct-value resolver made
  all five focused owner tests green and removed the recursive first-reference
  expression walk entirely.
- A same-name direct `ParamRef` remains a valid owner. This preserves the
  backend's generic VectorBounds metadata path in addition to the
  PartiallyObserved scatter path; the invariant is not scatter-only.
- The adversarial corpus model starts from an ordinary upper-censored
  Exponential declaration, then prepends a differently named Normal factor
  over the same full scatter at the resolved-`ModelMeta` boundary. Setting the
  upper bound to 1 makes the old transform invalid already at oracle point
  `q=0.1`, while the named-owner interval transform stays finite.
- Regenerating every oracle fixture on the current toolchain caused irrelevant
  last-bit drift in four pre-existing fixtures. Those files were restored;
  only the new fixture is retained. Corpus regeneration then changed only the
  appended hash/fingerprint entries plus the three new model/data/fixture
  files, preserving all existing model and oracle bytes.
- Bayescycle-side green gate: bayeswire 232 tests, bayesjax 453 tests, and 13
  root guards passed; Ruff format/check passed for both packages. `ty check`
  retained only the known unsupported-base diagnostics in inheritance-rejection
  tests (two in bayeswire, one in bayesjax).

## Cross-workflow follow-up from Bayesite Codex review

- Bayescycle PR #57's first Codex review was clean, but Bayesite PR #29 review
  found that Rust prior-predictive treated the adversarial non-owner scatter as
  separately generative. BayesJAX happened to ignore it because
  `_partially_observed_sites` filters by same-name free sites, which avoided two
  draws but silently discarded the extra density factor.
- A product-of-experts Factor has no supported ancestral simulation semantics.
  Red BayesJAX coverage confirmed the adversarial model was silently accepted;
  both backends now reject differently named assignable factors over
  VectorBounds free values before prior-predictive drawing. The check remains
  VectorBounds-specific so pre-existing unbounded resolved metadata keeps its
  established behavior.
- Post-follow-up gates: BayesJAX Ruff/ty plus 454 tests and all 13 root guards
  passed; ty retained only its known inheritance-rejection diagnostic.
- Codex round 2 on PR #57 found the shallow predictive check still silently
  dropped wrapped factors such as `y + 0`. Red coverage reproduced it. Unlike
  owner selection, predictive safety must inspect the entire site-value tree:
  any differently named factor whose value references a VectorBounds slot has
  no supported ancestral interpretation and must fail before drawing.
- The explicit typed expression/index walk catches direct, wrapped, indexed,
  and scatter references without changing density owner selection. BayesJAX's
  full 455-test suite, Ruff/ty, and 13 root guards passed after the fix.
- Codex round 3 identified the other half of a stochastic factor: its
  distribution parameters can reference the bounded value even when its value
  expression does not. Red coverage with `Normal(y, 1)` at a constant value
  reproduced another silent drop. Predictive validation therefore checks both
  value-expression and distribution-expression references, while exempting
  actual free/observed declaration sites that simulation handles.
- The distribution walk follows explicit dataclass fields and delegates all IR
  expression/index traversal to the typed helper. Full BayesJAX validation is
  green at 456 tests plus Ruff/ty and 13 root guards.
- Codex round 4 showed the reference-based approach was the wrong abstraction:
  a factor can reuse another declaration's name and bypass the exemption. More
  expression cases would only continue the chase. The root ambiguity is that
  one `stochastic_sites` sequence contains both declaration-backed generative
  sites and arbitrary density factors.
- Holistic replacement: prior predictive now inventories declaration-backed
  sites structurally, claims exactly one site for every Param, Observed, and
  non-Param free declaration, and rejects every unclaimed site as a Factor.
  Param/Observed claims match both target and declaration distribution;
  non-Param free values use the same-name direct/scatter owner shape. This is
  independent of expression nesting, references, factor names, and site order.
  Red tests cover an unrelated factor and a factor colliding with a declaration
  name in addition to direct, wrapped, and distribution-reference forms.
- The inventory preserves legacy differently named Param-site labels when
  target and declaration distribution match, but duplicate matching sites are
  ambiguous and fail. The reference walkers are gone. Final Python gates are
  green: bayeswire 232 tests, BayesJAX 460 tests, 13 root guards, Ruff, and ty
  with only the three known inheritance-rejection diagnostics.

## Follow-up — closed namespaced `Submodel` composition (#48, 2026-07-10)

### Agreed design and stage 1 north-star

- Composition is an authoring-time operation: `Submodel(Child)` prefixes and
  flattens the child's already-resolved `ModelMeta`. There is deliberately no
  hierarchical wire node and no backend recursion; dotted names remain opaque
  strings to every consumer.
- The namespace is closed. Child `Data` declarations surface as prefixed bind
  keys; there is no parent-to-child input wiring or constructor configuration.
- A child contributes its complete model, including `Observed` likelihood
  factors. Parameters, data, derived expressions, and partially observed values
  are referenceable from the parent; observed values remain non-expression
  declarations, matching ordinary same-class behavior.
- Child dimension variable names, dimension labels, and coordinate keys are all
  prefixed. This makes repeated instances isolated by construction.
- First public red→green evidence: the north-star test failed at collection with
  `ImportError: cannot import name 'Submodel' from 'bayeswire'`, then passed
  after the declaration proxy and resolved-metadata prefix/merge path landed.
- Interesting seam: parent expressions that reference a child derived expression
  cannot point at an `ExpressionRef` because no such final IR node exists. The
  resolver embeds a prefixed copy of the child's final expression tree, exactly
  as same-class expression reuse does during Python class-body evaluation.
- A union of separate parameter/data/namespace proxy types made normal class-body
  code fail strict typing: `effects.n` statically remained the full union. One
  private typed proxy plus an explicit member-kind enum keeps the public dynamic
  attribute syntax usable while resolution still rejects category mistakes.
- Auditing every `Data` dispatch site found two non-obvious consumers beyond
  expressions: child scalar data may size parent `Data`/`Param` declarations,
  and child exact-vector data may feed all `PartiallyObserved` inputs and bounds.
  Both now normalize through qualified data-dimension symbols before final IR.
- `bayescycle` model auto-discovery needed authoring provenance that intentionally
  does not belong in `ModelMeta`. Adding a public `model_submodels(...)` hook
  would have made IR-reconstructed classes observably different through the
  public hooks, violating an existing invariant. Instead, the loader inspects
  the explicit public `Submodel` declarations still present in each authoring
  class body, removes referenced local components, and selects the sole root.

### Corpus and backend conformance

- The new `composed_measurements` case pins repeated submodels, child `Observed`
  factors, parent use of child expressions, dotted data keys, and flat packing.
  Produce-conformance went red on the six expected missing corpus artifacts,
  then green at 38 tests after fixture/corpus generation.
- Regenerating all JAX-oracle fixtures changed a few pre-existing floating-point
  values under the current JAX environment. Those unrelated rewrites were
  restored; the committed corpus diff is strictly the new case plus appended
  hash/fingerprint entries. Existing model documents remained byte-identical.
- bayesjax consume-conformance passes all 10 cases, including compilation and
  gradient evaluation of the new dotted-name document. No backend composition
  branch was added: the backend sees only ordinary opaque string keys.
- The pinned Bayesite v0.2.0 binary sampled the new corpus document unchanged;
  its posterior header and draw maps preserved `first.location` and
  `second.location`. `bayesite-idata` then exported that real run to NetCDF and
  xarray reopened both dotted posterior variable names intact. This checks the
  external engine and artifact/export seams, not only Python metadata.

## Codex review round 1 (PR #58)

- Both findings reproduced red before fixing:
  1. A direct re-export (`theta = child.theta`) was neither a declaration nor a
     deferred operator tree, so `_resolve_expressions` silently skipped it and a
     later composed wrapper could not see `theta`. Direct member proxies now
     resolve as named expressions, preserving the same embedded-expression
     semantics across another composition level.
  2. `Submodel.model_cls` and `.symbol` were implementation fields, so children
     with those valid declaration names were shadowed before `__getattr__` ran.
     Internal target/identity and member-proxy details now live in one private
     state object; the only reserved child segment is the deliberately obscure
     `_bayeswire_state`. Nested proxies are covered as well. Bayescycle reaches
     the target through the narrow `submodel_target(...)` function rather than
     introspecting storage.
