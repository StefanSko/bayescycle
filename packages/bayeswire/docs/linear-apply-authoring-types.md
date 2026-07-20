# Linear-map application interface spike

This experiment makes a typed authoring scaffold importable without changing
model resolution, IR, serialization, or numerical consumers.

```python
from bayeswire.math import linear

@model
class MvnNonCentered:
    n = Data.scalar()
    mean = Data.vector(n)
    latent_chol = Data.matrix(n, n)
    z = Param(Normal(0.0, 1.0), size=n)

    theta = mean + linear(latent_chol).apply(z)
```

`linear(matrix)` is implemented only far enough to return an immutable builder.
Calling `.apply(vector)` deliberately raises `NotImplementedError` in this
spike.

## Public typing

The interface lives in
[`src/bayeswire/math.py`](../src/bayeswire/math.py):

```python
class LinearMap(Protocol):
    def apply(self, vector: object, /) -> DeferredMatVecOp: ...


def linear(matrix: object, /) -> LinearMap: ...
```

The protocol exposes one algebraic affordance: application to a vector. It does
not expose expression arithmetic, indexing, composition, or general matrix
multiplication on the incomplete builder.

`object` is intentional at the authoring boundary. Before `@model` resolves
names, either operand may be a declaration, submodel member, scalar-composed
matrix expression, or another deferred expression. Exact ranks and dimensions
remain a backend binding concern.

## Intended lowering

A completed `.apply(...)` implementation would be only:

```python
@dataclass(frozen=True)
class _DeferredLinearMap:
    matrix: object

    def apply(self, vector: object, /) -> DeferredMatVecOp:
        return DeferredMatVecOp(matrix=self.matrix, vector=vector)
```

The existing declaration-resolution path would then continue unchanged:

```text
linear(latent_chol)
    -> authoring-only _DeferredLinearMap(matrix=latent_chol)

.apply(z)
    -> DeferredMatVecOp(matrix=latent_chol, vector=z)

@model resolution
    -> MatVecOp(
           matrix=DataRef("latent_chol"),
           vector=ParamRef("z"),
       )
```

The temporary `_DeferredLinearMap` never enters `ModelMeta` or serialized IR.
Only the already-specified `MatVecOp` crosses the declaration boundary.

## Relationship to current `@` authoring

This branch still contains the existing `matrix @ vector` implementation so the
new interface can be reviewed independently. If adopted, `linear(...).apply(...)`
would replace it rather than become a second public spelling.

That follow-up would remove `__matmul__` from declarations, deferred expression
classes, final expression classes, and the public expression protocol. Direct
construction and serialization of `MatVecOp` would remain unchanged.

Both authoring forms lower to identical IR, so switching syntax should leave:

- canonical model bytes and hashes unchanged;
- the Bayeswire corpus unchanged;
- Bayesjax binding and evaluation unchanged;
- Bayesite native and Wasm consumers unchanged;
- exact `[m, n]` applied to `[n]` producing `[m]` semantics unchanged.

## Deliberately unresolved

1. Whether the public name should be `linear`, `linear_map`, or
   `as_linear_map`.
2. Whether an incomplete `linear(matrix)` builder escaping a class-body
   expression needs a dedicated declaration error.
3. Whether obvious rank mismatches should fail during declaration resolution or
   remain exclusively backend binding errors.
4. Whether `LinearMap` should remain a public protocol or only appear as an
   inferred return type.

No new IR or consumer capability is implied by this interface spike.
