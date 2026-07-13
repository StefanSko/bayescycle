const PYODIDE_BASE = new URL("../../vendor/pyodide/", import.meta.url);
const BAYESWIRE_BASE = new URL("../../vendor/bayeswire/", import.meta.url);
const BAYESWIRE_ROOT = "/vendor";
let runtime;

try {
  runtime = await initializeRuntime();
  self.postMessage({ type: "ready" });
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

function validRequest(value) {
  return value !== null && typeof value === "object" && value.type === "compile" &&
    typeof value.id === "string" && typeof value.source === "string";
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
import hashlib
import json
import traceback

import bayeswire.ir
from bayeswire.model.decorator import is_model_class

try:
    # Capture trusted functions before user source can mutate their modules.
    # The JS client discards this worker after every compilation, so module
    # mutations cannot affect a later project.
    _trusted_canonical_bytes = bayeswire.ir.canonical_bytes
    _namespace = {"__name__": "__playground_editor__"}
    exec(compile(__playground_source, "<playground-editor>", "exec"), _namespace)
    _models = []
    _seen = set()
    for _value in _namespace.values():
        if (
            is_model_class(_value)
            and getattr(_value, "__module__", None) == "__playground_editor__"
            and id(_value) not in _seen
        ):
            _models.append(_value)
            _seen.add(id(_value))
    if len(_models) != 1:
        raise ValueError(f"Expected exactly one @model class, found {len(_models)}")
    _ir_bytes = _trusted_canonical_bytes(_models[0]._model_meta)
    _result = {
        "ok": True,
        "ir_base64": base64.b64encode(_ir_bytes).decode("ascii"),
        "ir_hash": hashlib.sha256(_ir_bytes).hexdigest(),
    }
except BaseException as _error:
    _frames = traceback.extract_tb(_error.__traceback__)
    _frames = [
        _frame for _frame in _frames
        if _frame.filename == "<playground-editor>" or "/bayeswire/" in _frame.filename
    ]
    _result = {
        "ok": False,
        "exception_type": f"{type(_error).__module__}.{type(_error).__qualname__}",
        "message": str(_error),
        "traceback": "Traceback (most recent call last):\n"
            + "".join(traceback.format_list(_frames))
            + "".join(traceback.format_exception_only(type(_error), _error)),
    }
json.dumps(_result, separators=(",", ":"))
`;
