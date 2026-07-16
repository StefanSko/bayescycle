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
  return value !== null && typeof value === "object" && value.type === "compile" &&
    value.protocol === 1 && typeof value.id === "string" && typeof value.source === "string";
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
from bayeswire.constraints import Interval, Ordered, Positive, UnitInterval, VectorBounds
from bayeswire.distributions import Normal, Truncated
from bayeswire.model import (
    DataDimRef,
    ResolvedDataRankSchema,
    ResolvedDataShapeSchema,
    attached_model_dimensions,
    is_model_class,
    model_dependencies,
    model_meta,
)
from bayeswire.model.expr import ConstNode, DataRef, ParamRef


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


def _integer_data_names(meta):
    names = set()
    for data in meta.data.values():
        if isinstance(data.schema, ResolvedDataShapeSchema):
            names.update(dim.name for dim in data.schema.dims if isinstance(dim, DataDimRef))
    for parameter in meta.params.values():
        if isinstance(parameter.size, DataRef):
            names.add(parameter.size.name)
    return names


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
            }
            for name, data in meta.data.items()
        ],
        "observed": [{"name": observed.name} for observed in meta.observed_nodes],
    }


def compile_editor_source(source):
    try:
        namespace = {"__name__": "__playground_editor__"}
        exec(compile(source, "<playground-editor>", "exec"), namespace)
        models = []
        seen = set()
        for value in namespace.values():
            if (
                is_model_class(value)
                and getattr(value, "__module__", None) == "__playground_editor__"
                and id(value) not in seen
            ):
                models.append(value)
                seen.add(id(value))
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
        selected_model = models[0]
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
            if frame.filename == "<playground-editor>" or "/bayeswire/" in frame.filename
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


compile_editor_source(__playground_source)
`;
