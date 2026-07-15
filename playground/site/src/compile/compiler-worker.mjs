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
        { type: "compiled", id: message.id, irBytes: bytes.buffer },
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
import json
import traceback

import bayeswire.ir
from bayeswire.model import is_model_class, model_dependencies, model_meta


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
        ir_bytes = bayeswire.ir.canonical_bytes(model_meta(models[0]))
        result = {
            "ok": True,
            "ir_base64": base64.b64encode(ir_bytes).decode("ascii"),
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
