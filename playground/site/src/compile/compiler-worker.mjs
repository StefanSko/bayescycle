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
        { type: "compiled", id: message.id, irBytes: bytes.buffer, irHash: result.ir_hash },
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
import builtins
import hashlib
import json
import math
import traceback
import types

import bayeswire.ir
from bayeswire.model.decorator import ModelMeta


def _invoke_trusted_compile(
    source,
    _ir_globals=dict(vars(bayeswire.ir)),
    _ir_codes={
        name: getattr(bayeswire.ir, name).__code__
        for name in ("meta_to_dict", "_encode_value", "_encode_map", "_encode_node")
    },
    _ir_defaults={
        name: getattr(bayeswire.ir, name).__defaults__
        for name in ("meta_to_dict", "_encode_value", "_encode_map", "_encode_node")
    },
    _node_specs=dict(bayeswire.ir.NODE_SPECS_BY_CLASS),
    _encode_string=json.encoder.encode_basestring,
    _math_isfinite=math.isfinite,
    _function_type=types.FunctionType,
    _namespace_type=types.SimpleNamespace,
    _builtins=dict(vars(builtins)),
    _model_meta_type=ModelMeta,
    _isinstance=isinstance,
    _bool_type=bool,
    _dict_type=dict,
    _float_type=float,
    _int_type=int,
    _list_type=list,
    _str_type=str,
    _repr=repr,
    _stringify=str,
    _type=type,
    _base64_b64encode=base64.b64encode,
    _sha256=hashlib.sha256,
    _extract_tb=traceback.extract_tb,
    _format_list=traceback.format_list,
    _format_exception_only=traceback.format_exception_only,
):
    # Remove the only global reference before user source executes. Build an
    # isolated serializer function graph whose globals and mutable registry
    # were snapshotted before user code can mutate bayeswire.ir.
    globals().pop("_invoke_trusted_compile", None)
    trusted_globals = dict(_ir_globals)
    trusted_globals["__builtins__"] = _builtins
    trusted_globals["NODE_SPECS_BY_CLASS"] = _node_specs
    trusted_globals["math"] = _namespace_type(isfinite=_math_isfinite)
    for name, code in _ir_codes.items():
        trusted_globals[name] = _function_type(
            code,
            trusted_globals,
            name,
            _ir_defaults[name],
        )
    trusted_meta_to_dict = trusted_globals["meta_to_dict"]

    def encode_json(value):
        if value is None:
            return "null"
        if _isinstance(value, _bool_type):
            return "true" if value else "false"
        if _isinstance(value, _str_type):
            return _encode_string(value)
        if _isinstance(value, _int_type):
            return _stringify(value)
        if _isinstance(value, _float_type):
            if not _math_isfinite(value):
                raise ValueError("strict JSON cannot encode a non-finite float")
            return _repr(value)
        if _isinstance(value, _list_type):
            return "[" + ",".join(encode_json(item) for item in value) + "]"
        if _isinstance(value, _dict_type):
            return "{" + ",".join(
                _encode_string(key) + ":" + encode_json(item)
                for key, item in value.items()
            ) + "}"
        raise TypeError(f"strict JSON cannot encode {_type(value).__name__}")

    try:
        namespace = {"__name__": "__playground_editor__"}
        exec(compile(source, "<playground-editor>", "exec"), namespace)
        models = []
        seen = set()
        for value in namespace.values():
            if (
                _isinstance(value, _type)
                and _isinstance(value.__dict__.get("_model_meta"), _model_meta_type)
                and getattr(value, "__module__", None) == "__playground_editor__"
                and id(value) not in seen
            ):
                models.append(value)
                seen.add(id(value))
        if len(models) != 1:
            raise ValueError(f"Expected exactly one @model class, found {len(models)}")
        ir_bytes = encode_json(trusted_meta_to_dict(models[0]._model_meta)).encode("utf-8")
        result = {
            "ok": True,
            "ir_base64": _base64_b64encode(ir_bytes).decode("ascii"),
            "ir_hash": _sha256(ir_bytes).hexdigest(),
        }
    except BaseException as error:
        frames = _extract_tb(error.__traceback__)
        frames = [
            frame for frame in frames
            if frame.filename == "<playground-editor>" or "/bayeswire/" in frame.filename
        ]
        result = {
            "ok": False,
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": "Traceback (most recent call last):\n"
                + "".join(_format_list(frames))
                + "".join(_format_exception_only(_type(error), error)),
        }
    return encode_json(result)


_invoke_trusted_compile(__playground_source)
`;
