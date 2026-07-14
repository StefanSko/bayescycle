"""Load a bayeswire model declaration from a Python file."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

from bayeswire.model import ModelMeta, is_model_class, model_dependencies, model_meta


class ModelLoadError(RuntimeError):
    """Raised when a Python model file cannot produce one model metadata object."""


@dataclass(frozen=True)
class LoadedModel:
    """A resolved model selected from a Python module."""

    name: str
    model_cls: type[object]
    meta: ModelMeta


def load_model(path: Path, model_name: str | None) -> LoadedModel:
    """Execute ``path`` and return the selected bayeswire model metadata."""
    module = _execute_module(path)
    if model_name is not None:
        return _load_named_model(module, model_name)
    return _load_only_model(module)


def _execute_module(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ModelLoadError(f"model file does not exist: {resolved}")

    module_name = _module_name_for_path(resolved)
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ModelLoadError(f"cannot load model file: {resolved}")

    module = importlib.util.module_from_spec(spec)
    parent = str(resolved.parent)
    inserted_path = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted_path = True
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"executing model file failed: {resolved}: {exc}") from exc
    finally:
        if inserted_path:
            sys.path.remove(parent)
    return module


def _module_name_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return f"_bayescycle_model_{path.stem}_{digest}"


def _load_named_model(module: ModuleType, name: str) -> LoadedModel:
    namespace = cast("Mapping[str, object]", module.__dict__)
    if name not in namespace:
        raise ModelLoadError(f"model {name!r} was not found in {module.__file__}")
    value = namespace[name]
    if not is_model_class(value):
        raise ModelLoadError(f"object {name!r} is not a bayeswire @model declaration")
    model_cls = cast(type[object], value)
    try:
        model_dependencies(model_cls)
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(f"invalid bayeswire model dependency graph: {exc}") from exc
    return LoadedModel(
        name=name,
        model_cls=model_cls,
        meta=model_meta(value),
    )


def _load_only_model(module: ModuleType) -> LoadedModel:
    matches: list[LoadedModel] = []
    namespace = cast("Mapping[str, object]", module.__dict__)
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if _declared_in_module(value, module) and is_model_class(value):
            matches.append(
                LoadedModel(
                    name=name,
                    model_cls=cast(type[object], value),
                    meta=model_meta(value),
                )
            )

    if not matches:
        raise ModelLoadError(
            f"no bayeswire @model declaration was found in {module.__file__}; "
            "pass --model NAME if the model is imported"
        )
    roots = _unreferenced_model_roots(matches)
    if len(roots) == 1:
        return roots[0]

    names = ", ".join(match.name for match in matches)
    raise ModelLoadError(
        f"multiple bayeswire models were found in {module.__file__}: {names}; "
        "choose one with --model NAME"
    )


def _unreferenced_model_roots(models: list[LoadedModel]) -> list[LoadedModel]:
    """Return local model classes not reachable from another local model."""
    referenced: set[type[object]] = set()
    pending = [loaded.model_cls for loaded in models]
    traversed: set[type[object]] = set()
    try:
        while pending:
            model_cls = pending.pop()
            if model_cls in traversed:
                continue
            traversed.add(model_cls)
            dependencies = model_dependencies(model_cls)
            referenced.update(dependencies)
            pending.extend(dependencies)
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(f"invalid bayeswire model dependency graph: {exc}") from exc
    return [loaded for loaded in models if loaded.model_cls not in referenced]


def _declared_in_module(value: object, module: ModuleType) -> bool:
    declared_module = getattr(value, "__module__", None)
    return isinstance(declared_module, str) and declared_module == module.__name__
