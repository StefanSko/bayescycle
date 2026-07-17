const PYODIDE_BASE = new URL("../../vendor/pyodide/", import.meta.url);
const BAYESWIRE_BASE = new URL("../../vendor/bayeswire/", import.meta.url);
const BAYESWIRE_ROOT = "/vendor";
let runtime;

try {
  runtime = await initializeRuntime();
  lockDownNetwork();
  self.postMessage({ type: "ready", protocol: 1 });
} catch (error) {
  setTimeout(() => { throw error; });
}

self.addEventListener("message", (event) => {
  void handleMessage(event.data);
});

async function handleMessage(message) {
  if (!validRequest(message)) return;
  runtime.globals.set("__playground_source", message.source);
  runtime.globals.set("__playground_mode", message.mode ?? "model");
  if (message.mode === "with-prior") {
    runtime.globals.set("__playground_prior_source", message.priorSource);
  }
  try {
    const serialized = runtime.runPython(PYTHON_COMPILE);
    const result = JSON.parse(serialized);
    if (result.ok === true) {
      const bytes = decodeBase64(result.ir_base64);
      self.postMessage(
        {
          type: "compiled",
          id: message.id,
          irBytes: bytes.buffer,
          modelSchema: result.model_schema,
        },
        [bytes.buffer],
      );
    } else {
      self.postMessage({
        type: "compile-error",
        id: message.id,
        exceptionType: result.exception_type,
        message: result.message,
        traceback: result.traceback,
      });
    }
  } finally {
    runtime.globals.delete("__playground_source");
    runtime.globals.delete("__playground_mode");
    runtime.globals.delete("__playground_prior_source");
  }
}

function lockDownNetwork() {
  const blockedFetch = () => Promise.reject(new TypeError("Network access is disabled while compiling model source"));
  const blockedConstructor = function blockedNetworkApi() {
    throw new TypeError("Network access is disabled while compiling model source");
  };
  for (const [name, value] of [
    ["fetch", blockedFetch],
    ["WebSocket", blockedConstructor],
    ["EventSource", blockedConstructor],
    ["XMLHttpRequest", blockedConstructor],
    ["Worker", blockedConstructor],
    ["SharedWorker", blockedConstructor],
    ["importScripts", blockedConstructor],
  ]) {
    Object.defineProperty(globalThis, name, {
      value,
      configurable: false,
      enumerable: false,
      writable: false,
    });
  }
}

function validRequest(value) {
  if (value === null || typeof value !== "object" || value.type !== "compile" ||
      value.protocol !== 1 || typeof value.id !== "string" ||
      typeof value.source !== "string") return false;
  if (value.mode === undefined) return value.priorSource === undefined;
  return value.mode === "with-prior" && typeof value.priorSource === "string";
}

async function initializeRuntime() {
  const module = await import(new URL("pyodide.mjs", PYODIDE_BASE).href);
  const pyodide = await module.loadPyodide({ indexURL: PYODIDE_BASE.href });
  pyodide.FS.mkdirTree(BAYESWIRE_ROOT);
  const manifestResponse = await fetch(new URL("MANIFEST.json", BAYESWIRE_BASE));
  if (!manifestResponse.ok) throw new Error("Could not load vendored bayeswire manifest");
  const manifest = await manifestResponse.json();
  if (!Array.isArray(manifest) || !manifest.every((path) => typeof path === "string")) {
    throw new Error("Invalid vendored bayeswire manifest");
  }
  for (const path of manifest) {
    const response = await fetch(new URL(path, BAYESWIRE_BASE));
    if (!response.ok) throw new Error(`Could not load vendored bayeswire file ${path}`);
    const destination = `${BAYESWIRE_ROOT}/${path}`;
    pyodide.FS.mkdirTree(destination.slice(0, destination.lastIndexOf("/")));
    pyodide.FS.writeFile(destination, new Uint8Array(await response.arrayBuffer()));
  }
  pyodide.runPython(`import sys\nsys.path.insert(0, ${JSON.stringify(BAYESWIRE_ROOT)})`);
  return pyodide;
}

function decodeBase64(encoded) {
  const binary = atob(encoded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

const PYTHON_COMPILE = String.raw`
import base64
from dataclasses import fields, is_dataclass
import json
import math
import traceback

import bayeswire.ir
from bayeswire import with_prior
from bayeswire.constraints import Interval, Ordered, Positive, UnitInterval, VectorBounds
from bayeswire.distributions import Normal, Truncated
from bayeswire.model import (
    DataDimRef,
    ModelMeta,
    ResolvedDataRankSchema,
    ResolvedDataShapeSchema,
    attached_model_dimensions,
    is_model_class,
    model_dependencies,
    model_meta,
)
from bayeswire.model.dimensions import ResolvedModelDimensions
from bayeswire.model.expr import (
    ConstNode,
    DataRef,
    ParamRef,
    ScalarIndex,
    VectorScatterOp,
)


SCHEMA_FORMAT = "bayescycle.playground.model-schema.v0"


def _friendly_value(value):
    if isinstance(value, ConstNode):
        return _friendly_value(value.value)
    if isinstance(value, DataRef | ParamRef):
        return value.name
    if value is None or isinstance(value, bool | int | float | str):
        return repr(value) if not isinstance(value, str) else value
    if isinstance(value, tuple):
        return "[" + ", ".join(_friendly_value(item) for item in value) + "]"
    if is_dataclass(value) and not isinstance(value, type):
        arguments = ", ".join(
            _friendly_value(getattr(value, field.name)) for field in fields(value)
        )
        return f"{type(value).__name__}({arguments})"
    return type(value).__name__


def _constraint_label(constraint):
    if constraint is None:
        return None
    if isinstance(constraint, Positive):
        return "> 0"
    if isinstance(constraint, UnitInterval):
        return "0 < value < 1"
    if isinstance(constraint, Interval):
        return f"{constraint.lower} < value < {constraint.upper}"
    if isinstance(constraint, Ordered):
        return "strictly increasing"
    if isinstance(constraint, VectorBounds):
        sides = []
        if constraint.lower is not None:
            sides.append(f"> {_friendly_value(constraint.lower)}")
        if constraint.upper is not None:
            sides.append(f"< {_friendly_value(constraint.upper)}")
        return "per-coordinate " + " and ".join(sides)
    return type(constraint).__name__


def _constant_number(value):
    if isinstance(value, ConstNode):
        value = value.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parameter_default(parameter):
    if parameter.size is not None:
        return None
    distribution = parameter.distribution
    if isinstance(distribution, Truncated):
        distribution = distribution.base
    if not isinstance(distribution, Normal):
        return None
    value = _constant_number(distribution.loc)
    if value is None:
        return None
    constraint = parameter.constraint
    if isinstance(constraint, Positive) and value <= 0:
        return None
    if isinstance(constraint, UnitInterval) and not 0 < value < 1:
        return None
    if isinstance(constraint, Interval) and not constraint.lower < value < constraint.upper:
        return None
    return value


def _parameter_shape(name, parameter, dimensions):
    variable_dims = dimensions.variables.get(name) if dimensions is not None else None
    if variable_dims is not None and variable_dims.names:
        return list(variable_dims.names)
    if parameter.size is None:
        return []
    if isinstance(parameter.size, DataRef):
        return [parameter.size.name]
    return [str(parameter.size)]


def _data_kind(schema):
    rank = schema.rank if isinstance(schema, ResolvedDataRankSchema) else len(schema.dims)
    return {0: "scalar", 1: "vector", 2: "matrix"}.get(rank, "array")


def _walk_children(value):
    if is_dataclass(value) and not isinstance(value, type):
        return [getattr(value, field.name) for field in fields(value)]
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list | tuple | set | frozenset):
        return list(value)
    return []


def _data_ref_names(root, names):
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, DataRef):
            names.add(value.name)
        stack.extend(_walk_children(value))


def _integer_position_data_names(root, names):
    seen = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        if isinstance(value, ScalarIndex):
            _data_ref_names(value.expr, names)
            continue
        if isinstance(value, VectorScatterOp):
            for indexish in (value.length, value.observed_idx, value.missing_idx):
                _data_ref_names(indexish, names)
            stack.extend([value.observed_values, value.missing_values])
            continue
        if is_dataclass(value) and not isinstance(value, type):
            seen.add(id(value))
            for field in fields(value):
                child = getattr(value, field.name)
                if field.name == "total_count":
                    _data_ref_names(child, names)
                else:
                    stack.append(child)
            continue
        children = _walk_children(value)
        if children:
            seen.add(id(value))
            stack.extend(children)


def _integer_data_names(meta):
    names = set()
    for data in meta.data.values():
        if isinstance(data.schema, ResolvedDataShapeSchema):
            names.update(dim.name for dim in data.schema.dims if isinstance(dim, DataDimRef))
    for parameter in meta.params.values():
        if isinstance(parameter.size, DataRef):
            names.add(parameter.size.name)
    _integer_position_data_names(meta, names)
    return names


def _data_length(schema):
    if (
        isinstance(schema, ResolvedDataShapeSchema)
        and len(schema.dims) == 1
        and isinstance(schema.dims[0], int)
    ):
        return schema.dims[0]
    return None


def _model_schema(model):
    meta = model_meta(model)
    dimensions = attached_model_dimensions(model)
    integer_data = _integer_data_names(meta)
    return {
        "schema_format": SCHEMA_FORMAT,
        "parameters": [
            {
                "name": name,
                "prior": _friendly_value(parameter.distribution),
                "constraint": _constraint_label(parameter.constraint),
                "shape": _parameter_shape(name, parameter, dimensions),
                "default": _parameter_default(parameter),
            }
            for name, parameter in meta.params.items()
        ],
        "data": [
            {
                "name": name,
                "dtype": "int64" if name in integer_data else "float64",
                "kind": _data_kind(data.schema),
                "length": _data_length(data.schema),
            }
            for name, data in meta.data.items()
        ],
        "observed": [{"name": observed.name} for observed in meta.observed_nodes],
    }


def _local_models(namespace, module_name):
    models = []
    seen = set()
    for value in namespace.values():
        if (
            is_model_class(value)
            and getattr(value, "__module__", None) == module_name
            and id(value) not in seen
        ):
            models.append(value)
            seen.add(id(value))
    return models


def _select_editor_model(namespace):
    models = _local_models(namespace, "__playground_editor__")
    referenced = set()
    traversed = set()
    pending = list(models)
    while pending:
        model = pending.pop()
        if id(model) in traversed:
            continue
        traversed.add(id(model))
        dependencies = model_dependencies(model)
        referenced.update(dependencies)
        pending.extend(dependencies)
    if len(models) > 1:
        roots = [model for model in models if model not in referenced]
        if len(roots) == 1:
            models = roots
    if len(models) != 1:
        raise ValueError(f"Expected exactly one @model class, found {len(models)}")
    return models[0]


def _compile_model(source):
    namespace = {"__name__": "__playground_editor__"}
    exec(compile(source, "<playground-editor>", "exec"), namespace)
    return _select_editor_model(namespace), namespace


def _param_ref_names(root):
    names = set()
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, ParamRef):
            names.add(value.name)
        else:
            stack.extend(_walk_children(value))
    return names


def _completed_prior(target, authored_prior):
    """Complete an authored partial prior with untouched target declarations."""
    target_meta = model_meta(target)
    source_meta = model_meta(authored_prior)
    target_names = set(target_meta.params)
    source_names = set(source_meta.params)
    replacements = target_names & source_names

    supporting_names = set()
    pending = list(replacements)
    while pending:
        name = pending.pop()
        parameter = source_meta.params[name]
        for dependency in _param_ref_names(parameter.distribution):
            if dependency in source_names and dependency not in supporting_names:
                supporting_names.add(dependency)
                pending.append(dependency)
    unused = source_names - target_names - supporting_names
    if unused:
        name = next(name for name in source_meta.params if name in unused)
        raise ValueError(
            f"prior parameter {name!r} does not exist in the target or support a "
            "hierarchical replacement"
        )

    selected_params = {}
    for name, parameter in target_meta.params.items():
        selected_params[name] = source_meta.params.get(name, parameter)
    for name, parameter in source_meta.params.items():
        if name not in selected_params:
            selected_params[name] = parameter

    params = {}
    pending_params = dict(selected_params)
    while pending_params:
        ready = [
            name for name, parameter in pending_params.items()
            if not (_param_ref_names(parameter.distribution) & pending_params.keys())
        ]
        if not ready:
            raise ValueError("prior parameter dependencies could not be ordered")
        for name in ready:
            params[name] = pending_params.pop(name)

    def owner_sites(meta):
        owners = {}
        for site in meta.stochastic_sites:
            if isinstance(site.value, ParamRef) and site.value.name in meta.params:
                owners[site.value.name] = site
        return owners

    target_owners = owner_sites(target_meta)
    source_owners = owner_sites(source_meta)
    sites = tuple(
        source_owners.get(name, target_owners.get(name))
        for name in params
    )
    if any(site is None for site in sites):
        raise ValueError("prior composition could not resolve every parameter owner site")

    data = dict(target_meta.data)
    data.update(source_meta.data)
    free_values = {
        name: source_meta.free_values.get(name, target_meta.free_values.get(name))
        for name in params
    }
    if any(value is None for value in free_values.values()):
        raise ValueError("prior composition could not resolve every parameter free slot")

    target_dimensions = attached_model_dimensions(target)
    source_dimensions = attached_model_dimensions(authored_prior)
    all_variables = {}
    all_coords = {}
    for dimensions in (target_dimensions, source_dimensions):
        if dimensions is None:
            continue
        all_variables.update(dimensions.variables)
        all_coords.update(dimensions.coords)
    declared_names = (
        set(params)
        | set(data)
        | set(source_meta.expressions)
        | {observed.name for observed in source_meta.observed_nodes}
    )
    variables = {
        name: variable_dims
        for name, variable_dims in all_variables.items()
        if name in declared_names
    }
    used_dimension_names = {
        dimension_name
        for variable_dims in variables.values()
        for dimension_name in variable_dims.names
    }
    coords = {
        name: values for name, values in all_coords.items()
        if name in used_dimension_names
    }
    dimensions = ResolvedModelDimensions(variables=variables, coords=coords)

    completed_meta = ModelMeta(
        params=params,
        data=data,
        observed_nodes=source_meta.observed_nodes,
        expressions=dict(source_meta.expressions),
        free_values=free_values,
        stochastic_sites=sites,
    )
    return bayeswire.ir.bindable_from_meta(completed_meta, dimensions=dimensions)


def _compile_scenario(source, prior_source):
    target, target_namespace = _compile_model(source)
    prior_namespace = dict(target_namespace)
    prior_namespace["__name__"] = "__playground_prior__"
    exec(compile(prior_source, "<playground-prior>", "exec"), prior_namespace)
    models = _local_models(prior_namespace, "__playground_prior__")
    if len(models) != 1:
        raise ValueError(
            f"Expected exactly one @model class in prior-only source, found {len(models)}"
        )
    try:
        # Validate the authored source before completing its intentionally
        # omitted target Params. Only Bayeswire's missing-target error is the
        # expected signal that partial completion is needed; every structural
        # source error remains verbatim.
        return with_prior(target, prior=models[0])
    except ValueError as error:
        if not str(error).startswith("prior is missing target parameter "):
            raise
    completed_prior = _completed_prior(target, models[0])
    return with_prior(target, prior=completed_prior)


def compile_editor_source(source, mode="model", prior_source=None):
    try:
        if mode == "model":
            selected_model, _namespace = _compile_model(source)
        elif mode == "with-prior" and isinstance(prior_source, str):
            selected_model = _compile_scenario(source, prior_source)
        else:
            raise ValueError("Unsupported playground compiler mode")
        ir_bytes = bayeswire.ir.canonical_bytes(model_meta(selected_model))
        result = {
            "ok": True,
            "ir_base64": base64.b64encode(ir_bytes).decode("ascii"),
            "model_schema": _model_schema(selected_model),
        }
    except BaseException as error:
        frames = traceback.extract_tb(error.__traceback__)
        frames = [
            frame for frame in frames
            if frame.filename in ("<playground-editor>", "<playground-prior>")
            or "/bayeswire/" in frame.filename
        ]
        result = {
            "ok": False,
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": "Traceback (most recent call last):\n"
                + "".join(traceback.format_list(frames))
                + "".join(traceback.format_exception_only(type(error), error)),
        }
    return json.dumps(result, separators=(",", ":"))


compile_editor_source(
    __playground_source,
    __playground_mode,
    globals().get("__playground_prior_source"),
)
`;
