# Rust-Style Python Principles

These principles are intended for `bayeswire`: a small library with a narrow public story and strong correctness constraints.

## 1. Make invalid states hard to represent

Prefer structures that encode what is valid by construction.

Examples:
- keep incomplete vs bound vs compiled concepts separate
- distinguish public result types from backend sampler state
- avoid optional fields that are only valid in some undocumented modes

If two states have different invariants, they probably deserve different types or at least different explicit representations.

For `bayeswire`, this includes symbolic-vs-numeric distribution arguments,
static-vs-data-dependent shape dimensions, deferred-vs-resolved declaration
state, and resolved-metadata-vs-serialized-document state.

## 2. Keep the happy path obvious

A user should be able to understand the library through one path:

```python
@model
class MyModel:
    ...

document = meta_to_dict(model_meta(MyModel))
restored = meta_from_dict(document)
```

Anything that weakens that story should be justified carefully. Binding and
sampling belong to backend packages that consume this one.

Hierarchical models are part of this happy path, not an advanced side channel.
The user should be able to write hierarchical declarations with ordinary model
expressions as distribution arguments and rely on the compiler to resolve them
later through explicit internal states.

## 3. Prefer narrow interfaces over broad convenience

Good abstractions remove accidental complexity.
Bad abstractions hide the real behavior and add more names than value.

Ask:
- does this API reduce concepts?
- or does it just spread one concept across more files and wrappers?

## 4. Separate contract from mechanism

Public API should expose stable concepts.
Internal code may change freely.

For `bayeswire`:
- public: model declarations, distribution/constraint metadata, resolved
  `ModelMeta`, the IR codec, the dimension sidecar codec
- internal: the node registry, symbol tables, deferred syntax capture

## 5. Prefer explicit state transitions

Instead of hidden mode changes, prefer explicit phases:
- declared model
- bound model
- compiled model
- sampled result

Even when not formalized in the type system, code structure should make these transitions obvious.

## 6. Avoid stringly, dictly, or duck-typed architecture where a type would clarify intent

Small dataclasses or typed containers are often better than ad hoc nested dicts.
Use dicts when the domain is naturally map-shaped; do not default to them for convenience.

Do not rely on incidental methods or attributes to identify core internal states
when an explicit private type or protocol would make the invariant clearer. Loose
Python objects are fine at public boundaries; normalize them before they spread.

## 7. Localize complexity

Some complexity is real, especially around declaration resolution and the codec's field-kind rules. Keep that complexity behind narrow internal seams rather than leaking it across the codebase.

## 8. Delete speculative architecture

Do not keep generic layers around for features explicitly out of project scope.
If the only reason an abstraction exists is “maybe we add five inference engines later,” it probably should not exist now.

## 9. Favor boring names and direct modules

Prefer:
- `model_meta`
- `meta_to_dict`
- `canonical_bytes`
- `ModelMeta`

Avoid abstract names that obscure responsibility unless they buy real precision.

## 10. Validation is part of design

A good design is easier to validate.
If a design makes it hard to golden-test wire bytes, round-trip documents, or conformance-test consumers, the design may be wrong.

## 11. Minimize cross-cutting magic

Especially in a DSL, hidden behavior can become expensive quickly.
Keep the implementation understandable:
- where declarations are discovered
- how deferred syntax resolves to final expression IR
- how node classes map to wire tags and field kinds
- how documents decode back to resolved metadata

## 12. Optimize for maintainability under scientific pressure

This project is not just software; it is inference software.
Choose designs that make correctness reviews, benchmark analysis, and regression debugging easier.
