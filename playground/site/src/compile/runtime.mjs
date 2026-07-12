// Source: src/compile/runtime.ts, bayesledger @ 7346d71.

const PYODIDE_BROWSER_BASE = new URL("../../vendor/pyodide/", import.meta.url);
const BAYESWIRE_BROWSER_BASE = new URL("../../vendor/bayeswire/", import.meta.url);
const BAYESWIRE_ROOT = "/vendor";

/**
 * @typedef {{ok: true, irBytes: Uint8Array, irHash: string}} PythonCompileSuccess
 * @typedef {{ok: false, exceptionType: string, message: string, traceback: string}} PythonCompileFailure
 * @typedef {PythonCompileSuccess | PythonCompileFailure} CompileResult
 */

let runtimePromise;

/** @param {string} source @returns {Promise<CompileResult>} */
export async function compileInPyodide(source) {
  const runtime = await getRuntime();
  runtime.globals.set("__bayesledger_source", source);
  try {
    const serialized = runtime.runPython(PYTHON_COMPILE);
    if (typeof serialized !== "string") {
      throw new Error("Pyodide compiler returned a non-string result");
    }
    const result = parseSerializedResult(serialized);
    if (!result.ok) {
      return {
        ok: false,
        exceptionType: result.exception_type,
        message: result.message,
        traceback: result.traceback,
      };
    }
    return {
      ok: true,
      irBytes: decodeBase64(result.ir_base64),
      irHash: result.ir_hash,
    };
  } finally {
    runtime.globals.delete("__bayesledger_source");
  }
}

async function getRuntime() {
  runtimePromise ??= initializeRuntime();
  return runtimePromise;
}

async function initializeRuntime() {
  const loaderUrl = new URL("pyodide.mjs", PYODIDE_BROWSER_BASE);
  const module = await import(loaderUrl.href);
  const runtime = await module.loadPyodide({ indexURL: PYODIDE_BROWSER_BASE.href });

  runtime.FS.mkdirTree(BAYESWIRE_ROOT);
  const manifest = await readManifest();
  const files = await Promise.all(
    manifest.map(async (path) => [path, await readVendoredFile(path)]),
  );
  for (const [path, bytes] of files) {
    const destination = `/vendor/${path}`;
    runtime.FS.mkdirTree(destination.slice(0, destination.lastIndexOf("/")));
    runtime.FS.writeFile(destination, bytes);
  }
  runtime.runPython(`import sys\nsys.path.insert(0, ${JSON.stringify(BAYESWIRE_ROOT)})`);
  return runtime;
}

async function readManifest() {
  const response = await globalThis.fetch(new URL("MANIFEST.json", BAYESWIRE_BROWSER_BASE));
  if (!response.ok) throw new Error("Could not load vendored bayeswire manifest");
  const value = await response.json();
  if (!Array.isArray(value) || !value.every((path) => typeof path === "string")) {
    throw new Error("Invalid vendored bayeswire manifest");
  }
  return value;
}

async function readVendoredFile(path) {
  const response = await globalThis.fetch(new URL(path, BAYESWIRE_BROWSER_BASE));
  if (!response.ok) throw new Error(`Could not load vendored bayeswire file ${path}`);
  return new Uint8Array(await response.arrayBuffer());
}

function parseSerializedResult(text) {
  const value = JSON.parse(text);
  if (value === null || typeof value !== "object") {
    throw new Error("Invalid Pyodide compiler result");
  }
  if (value.ok === true && typeof value.ir_base64 === "string" && typeof value.ir_hash === "string") {
    return { ok: true, ir_base64: value.ir_base64, ir_hash: value.ir_hash };
  }
  if (
    value.ok === false &&
    typeof value.exception_type === "string" &&
    typeof value.message === "string" &&
    typeof value.traceback === "string"
  ) {
    return {
      ok: false,
      exception_type: value.exception_type,
      message: value.message,
      traceback: value.traceback,
    };
  }
  throw new Error("Invalid Pyodide compiler result");
}

function decodeBase64(encoded) {
  const binary = globalThis.atob(encoded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

const PYTHON_COMPILE = String.raw`
import base64
import builtins
import hashlib
import json
import sys
import traceback

import bayeswire.ir
from bayeswire.model.decorator import is_model_class

try:
    _original_import = builtins.__import__
    _stdlib = sys.stdlib_module_names

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            root = name.partition(".")[0]
            if root != "bayeswire" and root not in _stdlib:
                raise ImportError(
                    f"Import of {name!r} is not allowed; editor models may import only "
                    "the Python standard library and bayeswire"
                )
        return _original_import(name, globals, locals, fromlist, level)

    _editor_builtins = vars(builtins).copy()
    _editor_builtins["__import__"] = _restricted_import
    _namespace = {
        "__name__": "__bayesledger_editor__",
        "__builtins__": _editor_builtins,
    }
    exec(compile(__bayesledger_source, "<bayesledger-editor>", "exec"), _namespace)
    _models = []
    _seen = set()
    for _value in _namespace.values():
        if (
            is_model_class(_value)
            and getattr(_value, "__module__", None) == "__bayesledger_editor__"
            and id(_value) not in _seen
        ):
            _models.append(_value)
            _seen.add(id(_value))
    if len(_models) != 1:
        raise ValueError(f"Expected exactly one @model class, found {len(_models)}")

    _ir_bytes = bayeswire.ir.canonical_bytes(_models[0]._model_meta)
    _result = {
        "ok": True,
        "ir_base64": base64.b64encode(_ir_bytes).decode("ascii"),
        "ir_hash": hashlib.sha256(_ir_bytes).hexdigest(),
    }
except BaseException as _error:
    _frames = traceback.extract_tb(_error.__traceback__)
    _frames = [
        _frame for _frame in _frames
        if _frame.filename == "<bayesledger-editor>" or "/bayeswire/" in _frame.filename
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
