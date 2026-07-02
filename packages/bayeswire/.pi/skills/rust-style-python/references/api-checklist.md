# API and Architecture Checklist

Use this checklist before adding or revising public APIs, internal boundaries, or major abstractions.

## Scope fit

- Does this directly help users define a model, express a hierarchical model, serialize resolved metadata, or consume the wire format safely?
- If not, does it belong in `bayeswire` core at all?
- Does it conflict with `AGENTS.md` or `docs/invariants.md` non-goals?

## Public vs internal

- Is this concept truly user-facing?
- Can this remain internal?
- Does this expose backend details, benchmark details, or implementation-specific state?
- Would changing the backend later break this API?

## State and invariants

- What invariant does this code protect?
- Is that invariant explicit in the structure or only implied by comments?
- Are there hidden mode combinations that should be separated?
- Can invalid combinations be made unrepresentable or at least harder to construct?

## Surface area

- Is this the smallest API that solves the problem?
- Does it add a second way to do the same thing?
- Does it require extra wrappers, helper layers, or flags to feel usable?
- Can the design be simplified by deleting an option or special case?

## Naming

- Are names direct and responsibility-aligned?
- Would a new contributor understand what the module/type/function does from the name alone?
- Is a generic name hiding avoidable complexity?

## Data flow

- Is data flow explicit from declaration -> resolved metadata -> serialized document?
- Are transformations easy to trace?
- Are side effects and mutation minimized?

## Typing

- Would a small typed container be clearer than a dict or tuple here?
- Is this relying on duck typing or incidental attributes where an explicit private type/protocol would clarify intent?
- Does loose Python input stop at a public boundary, or does it leak into core internals as `Any`?
- Are hierarchical-model states such as symbolic distribution args and data-dependent shapes represented explicitly?
- Are return values structured enough to support safe downstream use?
- Are there optional fields that signal muddled state?

## Dependency discipline

- Does this add a runtime dependency?
- Is that dependency core to the product, or only convenience?
- Can the same goal be achieved with existing runtime dependencies and better code organization?

## Validation impact

- How will this be unit tested?
- Does this change canonical bytes, tags, or field lists? If so, it needs a
  spec changelog entry, a regenerated corpus, and a consumer-pin plan.
- Does produce-conformance against `corpus/` still pass byte-for-byte?

## Astral validation loop

At minimum before push, and preferably after each meaningful commit, run:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

If the change touches the codec, the registry, or any registered dataclass, review the corpus diff before committing.

## Final test

Can you explain the change in a few lines without mentioning internal incidental machinery?
If not, the design may still be too complicated.
