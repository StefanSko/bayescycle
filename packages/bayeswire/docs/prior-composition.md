# Authoring-time prior composition

## Status and scope

This document defines the Bayeswire v0 contract for immutable, complete prior
replacement:

```python
ComposedModel = with_prior(TargetModel, prior=SourcePriorModel)
```

Both inputs and the result are ordinary closed Bayeswire model classes. The
operation transforms resolved authoring metadata only: it performs no data
binding, numerical evaluation, sampling, inference, I/O, or workflow
orchestration.

The only public composition surface in v0 is `with_prior(...)`. Private
factorization, wiring, composition, and closure values are not compatibility
APIs. In particular, this contract does not expose open kernels, inputs,
exports, arbitrary wiring or renaming, partial replacement, or a component wire
format.

## Inputs and immutability

`target` and `prior` must each be a Bayeswire model class accepted by
`model_meta(...)`. Composition does not mutate either class, its `ModelMeta`, or
its attached dimension metadata. It returns a distinct model class accepted by
the same execution-facing metadata and serialization hooks as a decorated model
or a class reconstructed by `bindable_from_meta(...)`.

Each input must be independently closed and internally valid under the closure,
ownership, role, and dimension rules below. Names from one input cannot repair a
dangling reference or ambiguous role in the other.

The result is closed. No private input, export, wiring, source/target node, or
other composition state is present in its `ModelMeta` or dimension sidecar.

## Structural declaration ownership

A stochastic site is the declaration-backed site for a Param when both of these
are true:

```text
site.value == ParamRef(param_name)
site.distribution == params[param_name].distribution
```

The site's label need not equal the Param name unless the Param free slot uses
`VectorBounds`. A `VectorBounds` free slot follows the specialized IR rule: its
unique structural Param owner must also be the unique same-name stochastic
site. Exactly one site must match each Param, and one site cannot own multiple
declarations. Ownership never follows from a general expression reference or
stochastic-site position.

An observed declaration is associated structurally with a site whose value is
the direct `DataRef` of the observed name and whose distribution equals the
observed distribution. A non-Param free value uses the existing canonical
same-name owner rule, including the `VectorScatterOp` form for
`PartiallyObserved`. Additional unassociated sites are density factors.

## Source-prior contract

The source is prior-only and must satisfy all of these structural rules:

- its resolved free-value keys equal its Param keys exactly;
- each resolved Param free slot has exactly the Param's constraint and size;
- every source Param has exactly one declaration-backed stochastic site;
- there are no observed declarations or observed sites;
- there are no non-Param free values, including `PartiallyObserved` values;
- there are no additional density-factor sites.

An empty legacy `free_values` map is interpreted through
`resolved_free_values(...)`; a non-empty partial or inconsistent map is not a
legacy omission and is rejected.

Source derived expressions and declared data are permitted. Source-only Params
are also permitted and remain parameters of the result, which allows
hierarchical priors. Param dependencies in source prior distributions must form
an ancestral DAG consistent with source Param order: every `ParamRef` in a
Param's resolved distribution refers to an earlier source Param. Self,
forward, and cyclic dependencies are rejected rather than reordered. Every
`DataRef` used by a source prior refers to declared source data.

## Target factorization

Every target Param declaration, its corresponding free slot, and exactly its
structurally declaration-backed prior site are replaced. Target Param keys and
Param-backed resolved free slots must agree exactly in key, constraint, and
size, with the same empty-map legacy fallback as the source. Missing, duplicate,
or malformed Param sites make the target invalid for composition; a different
site that merely references the Param is retained as an additional factor.

Composition retains, in target order:

- declared data and observed nodes;
- derived expressions;
- `PartiallyObserved` free values and their sites;
- closed `Submodel` contents already flattened into target metadata;
- other non-Param free values and their canonical owners;
- observed likelihood sites and every additional density factor.

Retained additional factors remain score factors. This operation does not give
them ancestral sampling semantics, so existing generation-capability checks may
still reject the composed model.

## Same-name parameter interface

Every target Param name must be a source Param name. Source-only Params are
allowed; missing target names are not.

For each matching name, source and target declarations must have exactly equal:

- resolved constraints;
- resolved scalar/vector size structure;
- normalized `DataRef` name for a data-dependent size;
- optional dimension metadata, as defined below;
- coordinate metadata for every named dimension.

The source distribution may differ and is the replacement prior. There is no
constraint subtyping, broadcasting, symbolic shape solving, implicit renaming,
or backend-assisted compatibility in v0.

## Data merge

Resolved source data is emitted first in source order. Target-only data follows
in target order. Same-name data is emitted once at its source position only
when both resolved schemas and optional dimension metadata are exactly equal.
Incompatible same-name data is rejected. No concrete data value is read or
bound.

Data-reference identity is by the final public data name. Consequently, equal
same-name data-dependent parameter sizes remain valid after merging.

## Expressions and role collisions

Resolved source expressions are emitted first in source order. Target
expressions follow in target order. A same-name expression is always a
collision, even when the resolved trees happen to be equal.

Within each input and across the merge, names in the value namespace may have
only these overlaps:

- one Param, its same-name free slot, and its structural owner site;
- one non-Param free slot and its canonical same-name owner site;
- one observed node and its structurally associated observed site;
- one source Param and the matching target Param that it replaces;
- one pair of reviewed compatible same-name data declarations.

All other overlaps among Params, data, expressions, observed names, and
non-Param free values are rejected, including duplicate observed declarations.
A stochastic site's `name` is a factor label, not a value definition; it does
not by itself collide with a value name or establish ownership. Site labels and
site sequence are still preserved exactly for retained factors.

Source-only Params, data, and expressions therefore cannot collide with target
retained values. Namespaces are never synthesized and equal expression trees
are never deduplicated.

## Typed closure

Closure is validated recursively on each input before factorization and again on
the final metadata. References are resolved by kind:

- `ParamRef` may name any resolved free value, including a non-Param free value
  used by its canonical owner;
- `DataRef` in an observed owner may name that observed bind input; elsewhere it
  names declared data unless the existing declaration form explicitly defines
  another bind-input role;
- a Param/free-value `size` and every `DataDimRef` in a data schema must name
  declared scalar data;
- distribution, constraint, expression, index, schema, size, and stochastic-site
  trees are traversed, including nested registered dataclass fields.

Observed owner values and non-Param free-value owner forms are validated by
their existing structural rules. Composition does not rewrite references except
that same-name target `ParamRef` values naturally refer to installed source
Params and compatible same-name data is shared.

## Normative order

The closed result has this exact order:

```text
params:
  source Params in source order

data:
  source data in source order
  target-only data in target order

observed_nodes:
  target order

expressions:
  source expressions in source order
  target expressions in target order

free_values:
  source Param free slots in source Param order
  retained target non-Param free values in target free-value order

stochastic_sites:
  source declaration-backed Param sites in source stochastic-site order
  retained target non-Param sites in target stochastic-site order
```

Source validation makes its Param sites the complete source site set, but the
source stochastic-site sequence remains authoritative for factor order. It
must also be consistent with the ancestral Param dependency order above.

## Dimension sidecar

A model may have no attached sidecar. For one variable, dimension metadata is
therefore an optional value:

```text
None | ordered tuple[dimension name]
```

`None` and an explicitly attached empty tuple are different. Matching target
and source Params and shared data require exact equality of this optional value.
For every shared dimension name, coordinate metadata is also compared as an
optional value; absent coordinates and an attached coordinate tuple are
different.

Dimension composition uses this order and ownership:

- source Param and source data variable entries first;
- equal matching target Param entries are replaced by source entries;
- equal shared data entries are emitted once at the source position;
- target retained data and observed-outcome entries follow in target order;
- source coordinate entries are followed by target-only coordinate entries;
- a coordinate name present on both sides must have exactly equal tuples.

V0 does not add dimensions to non-Param free values: `PartiallyObserved` has no
authoring dimension declaration, and the ordinary sidecar attachment boundary
permits only final Params, data, and observed nodes. Any injected sidecar entry
for another variable role is rejected.

Composition closure validates allowed variable keys, coordinate usage, known
static ranks, and coordinate lengths for known static axis sizes before class
construction. Param rank is zero for `size=None` and one otherwise; data rank
comes from its resolved schema. Checks requiring concrete bound array shapes
remain a backend bind-time concern, as required by the dimension-sidecar v1
specification.

## Dependencies and root discovery

Composition records authoring-only direct model dependencies:

```python
model_dependencies(ComposedModel) == (TargetModel, SourcePriorModel)
```

`model_dependencies(...)` is the Bayeswire-owned root-discovery hook for all
model classes. A decorated model reports its direct `Submodel` targets in
class-declaration order; a model with neither composition nor Submodels reports
`()`. A model reconstructed from standalone IR has no authoring graph and
reports `()`.

Dependency tuples are immutable, deterministic, and acyclic. Root discovery
uses the transitive closure of this one hook, excludes referenced models only
from the local candidate set, and fails on cycles. Bayescycle and the playground
do not inspect private composition state or independently walk `Submodel`
declarations.

A class returned by `with_prior(...)` belongs to the calling module for local
root selection. Thus a local composition of imported inputs is a local
candidate, while a merely imported alias of a composition remains non-local.
Its direct dependency order remains target then prior; nested composition and
Submodel graphs are traversed through their own direct dependency tuples.

Dependency state is authoring provenance for root selection only. It is absent
from `ModelMeta`, canonical model bytes, decoded standalone IR, and the
dimension sidecar.

## Wire and consumer boundary

Prior composition closes and flattens before `ModelMeta`. The result serializes
as ordinary `bayeswire_ir` version 1:

- no `ModelMeta` field changes;
- no node tag or encoding rule changes;
- no component or dependency node is serialized;
- existing corpus documents and hashes remain byte-identical;
- only reviewed new closed-model corpus cases may be added.

Bayesjax and Bayesite bind and execute the result as an ordinary closed model.
If a backend must understand source/target composition, or any unresolved
composition state must cross the IR boundary, this contract must be revisited
rather than broadened silently.
