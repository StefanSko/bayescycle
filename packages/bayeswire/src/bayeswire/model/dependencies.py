"""Public authoring-time model dependency discovery."""

from __future__ import annotations

from bayeswire.model.core import Submodel, submodel_target
from bayeswire.model.decorator import is_model_class


def _require_model_class(value: object, *, label: str) -> type[object]:
    if not isinstance(value, type) or not is_model_class(value):
        raise TypeError(
            f"{label} must be a bayeswire model class decorated with @model or produced "
            "by bindable_from_meta(...)"
        )
    return value


def _direct_model_dependencies(model_cls: type[object]) -> tuple[type[object], ...]:
    """Read direct composition and Submodel edges from their owning declarations."""
    explicit = model_cls.__dict__.get("_model_dependencies", ())
    if not isinstance(explicit, tuple):
        raise TypeError("private model dependency state must be an immutable tuple")

    dependencies: list[type[object]] = []
    seen: set[type[object]] = set()
    for value in explicit:
        dependency = _require_model_class(value, label="model dependency")
        if dependency not in seen:
            dependencies.append(dependency)
            seen.add(dependency)

    for value in model_cls.__dict__.values():
        if not isinstance(value, Submodel):
            continue
        dependency = submodel_target(value)
        if dependency not in seen:
            dependencies.append(dependency)
            seen.add(dependency)
    return tuple(dependencies)


def _validate_dependency_graph(root: type[object]) -> None:
    """Reject malformed or cyclic authoring dependency graphs."""
    visited: set[type[object]] = set()
    path: list[type[object]] = []

    def visit(model_cls: type[object]) -> None:
        if model_cls in path:
            start = path.index(model_cls)
            cycle = path[start:] + [model_cls]
            rendered = " -> ".join(model.__name__ for model in cycle)
            raise ValueError(f"model dependency cycle detected: {rendered}")
        if model_cls in visited:
            return
        path.append(model_cls)
        for dependency in _direct_model_dependencies(model_cls):
            visit(dependency)
        path.pop()
        visited.add(model_cls)

    visit(root)


def model_dependencies(model_cls: object) -> tuple[type[object], ...]:
    """Return validated direct authoring dependencies in declaration order.

    Composition contributes target then prior edges. Decorated classes
    contribute direct ``Submodel`` targets. Standalone decoded IR has no
    authoring graph and reports an empty tuple.
    """
    model = _require_model_class(model_cls, label="model")
    _validate_dependency_graph(model)
    return _direct_model_dependencies(model)
