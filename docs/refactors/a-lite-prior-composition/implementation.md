# A-Lite prior composition — implementation record

This document records engineering difficulties and non-obvious decisions from
implementing authoring-time `with_prior(...)` composition. It is implementation
provenance, not a normative API or wire-format specification. The normative
contract is [`packages/bayeswire/docs/prior-composition.md`](../../../packages/bayeswire/docs/prior-composition.md).

## Scope and worktree

The work was isolated in:

```text
/Users/stefansko/Projects/bayescycle-a-prior-composition
feature/bayeswire-prior-composition-core
```

The start prompt is `START-A.md`; no `PROMPT_A.md` or `START_A.md` exists in the
worktree. Six copied planning documents remain intentionally untracked. They
were treated as design context, while checked-in instructions, invariants, and
IR specifications remained authoritative.

## Difficulties and lessons

### A narrow API still required a strict closure boundary

The public operation is only complete same-name prior replacement, but a safe
implementation could not be a map splice. The source and target first become
private immutable parameter/outcome kernels, are wired explicitly, composed,
and only then closed back into ordinary `ModelMeta`.

The difficult part was defining what "closed" means for both normal authored
models and models reconstructed from decoded metadata. Python annotations do
not enforce runtime field roles. Several malformed decoded values could look
close enough to valid declarations to survive ordinary attribute access. The
closure checks therefore grew from name/reference checks into recursive checks
for declaration roles, expression and index node inventories, owner shapes,
data schemas, sizes, sidecar containers, and final serialization.

### Declaration ownership must be structural

A stochastic site that references a parameter is not necessarily that
parameter's declaration-backed prior; it can be an additional density factor.
Owner selection therefore cannot use first-reference or site-order heuristics.
The implementation identifies Param owners by declaration name, distribution,
and direct value shape, and separately validates Observed, non-Param free-value,
and `VectorBounds` owners.

This surfaced several easy-to-miss cases:

- observed bind inputs also need ownership checks;
- non-Param free values can collide with owner names;
- `VectorBounds` permits a specialized direct/scatter owner shape;
- final merged metadata must be checked again because individually valid source
  and target declarations can become ambiguous after composition.

### Dependency order and factor order are different

Hierarchical source parameters require a DAG to prove ancestral closure and to
order parameter declarations. That topological order must not be reused for
stochastic factors. Source declaration-backed sites retain their original
factor order, followed by retained target factors in target order. Conflating
the two orders changed valid density semantics.

### Recursive IR traversal had to follow registry field kinds

Distribution extensions can contain references in nested dataclasses, tuples,
and registered map fields. Initial walkers handled common built-ins but missed
references and ancestry hidden in map-valued custom fields. Traversal was
changed to follow registered node structure while preserving duplicate
reference roles where multiplicity matters.

A related ongoing lesson is that registration and encoding are not by
themselves proof of semantic role validity or decode round-trip validity. A
registered node can still carry a runtime container of the wrong field kind,
and nested fields such as `Truncated.base` or `VectorBounds.lower` have roles
that generic dataclass traversal does not establish. These boundaries require
explicit validation.

### Python scalar equality is too permissive for IR metadata

Python considers `True == 1`, but JSON coordinates preserve distinct boolean
and integer scalar kinds. Dimension interface and merge checks therefore need
both exact runtime type and value equality. Similar care was needed for strict
integer sizes, finite numeric values, immutable tuple containers, and resolved
rank/shape schemas.

### Root discovery needed one Bayeswire-owned dependency hook

Files can define the source prior, target, and composed root together. Loaders
must not inspect private composition state, so Bayeswire exposes only
`model_dependencies(...)`. Bayescycle and the browser worker use that hook to
remove referenced local components and select a sole root.

Two less obvious cases required regressions:

- dependency roles must preserve duplicates rather than collapse by identity;
- even one locally discovered model must have its transitive graph validated,
  otherwise a cycle can bypass root selection.

The browser boundary remains source-only: a disposable worker receives source,
executes it, and returns only closed ordinary IR.

### Corpus changes had to remain strictly additive

The north-star composed model must encode identically to an equivalent
handwritten flat model. Existing corpus documents, hashes, fingerprints, and
IR-v1 tag assignments were repeatedly checked for byte stability. Only the new
`alternative_prior_regression` case was appended; no composition node or
provenance entered wire IR.

### Browser staging can become stale

The playground executes a staged Bayeswire copy, not necessarily the current
source tree. After Python changes, `playground/scripts/stage_assets.py` must run
before browser validation. Otherwise a green browser result can exercise an
older closure implementation. The visible Chromium journey did compile a file
with target, source, and composed root, generated from the replacement prior,
and completed an observed-data fit; final gates must restage exact HEAD again.

### The external engine is a separate repository

Bayesite is not a workspace package. Its conformance copy must be refreshed by
byte-reviewed vendoring from the final Bayeswire commit in a separate worktree,
then validated with the Rust repository's own ladder. Refreshing it before the
Python branch is final creates avoidable repeated vendor commits.

### Independent review exposed progressively deeper decoded-metadata cases

Repeated fresh `xhigh` reviews found defects beyond normal eDSL construction:
recursive map traversal, ownership collisions, factor ordering, strict size and
schema types, bool-versus-int coordinates, mutable sidecars, empty models,
single-root cycles, semantic field roles, and index executability. Each
confirmed issue was converted to a failing regression before its fix.

At commit `2880a81`, review round 6 still reported four unresolved boundaries:

1. encoder field-kind mismatches can produce documents the decoder rejects;
2. nested `VectorBounds` and `Truncated` fields can carry the wrong semantic
   roles;
3. `register_distribution(...)` can reclassify an existing constraint node;
4. legacy accessors can normalize malformed empty list containers before
   validation.

These are implementation defects, not reasons to broaden into Full A or change
IR v1.

### Review concurrency was external, not a repository state problem

Before launching review round 6, a process check showed an existing
`run-review.sh` invocation for:

```text
/Users/stefansko/Projects/bayesite-functional-generation
```

That was an unrelated Pi session using the same global review harness. The
implementation session waited for it to exit before launching its own
read-only review, avoiding resource/session contention. It did not touch this
worktree and no feature-branch state was lost.

## Validation state at this checkpoint

Before review round 6, the following were green on the feature worktree:

- Bayeswire: Ruff, ty (two pre-existing unsupported-base warnings), 351 tests;
- Bayesjax: full package validation;
- Bayescycle: full package validation;
- root workspace guards;
- playground format, lint, and browser tests.

That validation does not override the four confirmed review findings above.
After fixing them, exact-HEAD staging, all package/root/playground suites, a
fresh independent review, final Bayesite vendoring, and the PR review loop are
still required.
