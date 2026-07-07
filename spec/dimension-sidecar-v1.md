# bayeswire dimension sidecar, version 1

This document specifies the dimension-metadata sidecar: the JSON document
that carries `Dim(...)` labels and coordinates alongside — never inside — a
`bayeswire_ir` model document. The reference implementation is
`dimension_metadata_to_dict` / `dimension_metadata_from_dict` in
`bayeswire.model`.

Dimension labels are authoring-side semantic metadata only. They do not
change log density, transforms, sampling, or distribution shapes, which is
why they are not part of the model IR or its canonical bytes: two models
that differ only in dimension labels have identical model hashes.

## Document shape

```json
{
  "dims":   {"<variable>": ["<dim-name>", ...], ...},
  "coords": {"<dim-name>": [<scalar>, ...], ...}
}
```

- The document is a JSON object with exactly the keys `dims` and `coords`;
  anything else is malformed.
- `dims` maps each labeled model variable to its ordered dimension names.
  Variable names and dimension names are non-empty strings. A variable with
  no axes maps to `[]`.
- `coords` optionally maps dimension names to coordinate value arrays.
  Coordinate values are JSON scalars only: string, integer, float, boolean,
  or null. Float coordinates must be finite.
- Every dimension named in `coords` must appear in some variable's `dims`
  entry; coordinates for undeclared dimensions are malformed.

## Semantics

- Order is semantic: a variable's dimension names are positional axis
  labels, and coordinate arrays are ordered per-axis labels.
- Producers write the sidecar from resolved dimension metadata
  (`dimension_metadata_to_dict`). Consumers decode it with
  `dimension_metadata_from_dict` and reattach it at model reconstruction:
  `bindable_from_meta(meta, dimensions=...)`. Without the sidecar, a
  reconstructed model carries no dimension metadata; that is a valid state,
  not an error.
- Decoding executes no user code.
- Validation against concrete shapes (coordinate lengths versus bound axis
  sizes, dimension rank versus bound array rank) is a bind-time concern for
  backends, not part of this document format.

## Relationship to run artifacts

Workflow harnesses conventionally store the sidecar as `dims.json` next to
`model.ir.json` in a run directory. That filename is a run-directory
convention, not part of this format.
