const PYODIDE_BASE = new URL("../../vendor/pyodide/", import.meta.url);
const BAYESWIRE_BASE = new URL("../../vendor/bayeswire/", import.meta.url);
const BAYESWIRE_ROOT = "/vendor";
let runtime;

try {
  runtime = await initializeRuntime();
  lockDownNetwork();
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


def _invoke_trusted_compile(
    source,
    _canonical_bytes=bayeswire.ir.canonical_bytes,
    _is_model_class=is_model_class,
    _base64=base64,
    _hashlib=hashlib,
    _json=json,
    _traceback=traceback,
):
    # Remove the only global reference before user source executes. Trusted
    # collaborators remain function locals/defaults and the worker is thrown
    # away after this call, so __main__ cannot replace or poison them.
    globals().pop("_invoke_trusted_compile", None)
    try:
        namespace = {"__name__": "__playground_editor__"}
        exec(compile(source, "<playground-editor>", "exec"), namespace)
        models = []
        seen = set()
        for value in namespace.values():
            if (
                _is_model_class(value)
                and getattr(value, "__module__", None) == "__playground_editor__"
                and id(value) not in seen
            ):
                models.append(value)
                seen.add(id(value))
        if len(models) != 1:
            raise ValueError(f"Expected exactly one @model class, found {len(models)}")
        ir_bytes = _canonical_bytes(models[0]._model_meta)
        result = {
            "ok": True,
            "ir_base64": _base64.b64encode(ir_bytes).decode("ascii"),
            "ir_hash": _hashlib.sha256(ir_bytes).hexdigest(),
        }
    except BaseException as error:
        frames = _traceback.extract_tb(error.__traceback__)
        frames = [
            frame for frame in frames
            if frame.filename == "<playground-editor>" or "/bayeswire/" in frame.filename
        ]
        result = {
            "ok": False,
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": "Traceback (most recent call last):\n"
                + "".join(_traceback.format_list(frames))
                + "".join(_traceback.format_exception_only(type(error), error)),
        }
    return _json.dumps(result, separators=(",", ":"))


_invoke_trusted_compile(__playground_source)
`;
