"""Load a jaxstanv5 model declaration from a Python file."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from jaxstanv5.model import ModelMeta


class ModelLoadError(RuntimeError):
    """Raised when a Python model file cannot produce one model metadata object."""


class ModelObject(Protocol):
    """Object carrying jaxstanv5 model metadata."""

    _model_meta: ModelMeta


@dataclass(frozen=True)
class LoadedModel:
    """A resolved model selected from a Python module."""

    name: str
    model_cls: type[object]
    meta: ModelMeta


def load_model(path: Path, model_name: str | None) -> LoadedModel:
    """Execute ``path`` and return the selected jaxstanv5 model metadata."""
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
    if not _is_model_object(value):
        raise ModelLoadError(f"object {name!r} is not a jaxstanv5 @model declaration")
    return LoadedModel(
        name=name,
        model_cls=cast(type[object], value),
        meta=cast(ModelObject, value)._model_meta,
    )


def _load_only_model(module: ModuleType) -> LoadedModel:
    matches: list[LoadedModel] = []
    namespace = cast("Mapping[str, object]", module.__dict__)
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if _declared_in_module(value, module) and _is_model_object(value):
            matches.append(
                LoadedModel(
                    name=name,
                    model_cls=cast(type[object], value),
                    meta=cast(ModelObject, value)._model_meta,
                )
            )

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ModelLoadError(
            f"no jaxstanv5 @model declaration was found in {module.__file__}; "
            "pass --model NAME if the model is imported"
        )
    names = ", ".join(match.name for match in matches)
    raise ModelLoadError(
        f"multiple jaxstanv5 models were found in {module.__file__}: {names}; "
        "choose one with --model NAME"
    )


def _is_model_object(value: object) -> bool:
    return isinstance(value, type) and isinstance(getattr(value, "_model_meta", None), ModelMeta)


def _declared_in_module(value: object, module: ModuleType) -> bool:
    declared_module = getattr(value, "__module__", None)
    return isinstance(declared_module, str) and declared_module == module.__name__
