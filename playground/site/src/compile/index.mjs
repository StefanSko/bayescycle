/**
 * Source-only compiler client. The worker never receives project documents or run artifacts.
 * @typedef {{ok: true, irBytes: Uint8Array, irHash: string, executionContext: "worker"} |
 *   {ok: false, exceptionType: string, message: string, traceback: string,
 *    executionContext: "worker"}} CompileResult
 */
let worker;
let ready;
const pending = new Map();

function compilerWorker() {
  if (worker !== undefined) return worker;
  worker = new Worker(new URL("./compiler-worker.mjs", import.meta.url), { type: "module" });
  ready = new Promise((resolve, reject) => {
    const startup = setTimeout(() => reject(new Error("Compiler worker startup timed out")), 120_000);
    worker.addEventListener("message", function onReady(event) {
      if (event.data?.type !== "ready") return;
      clearTimeout(startup);
      worker.removeEventListener("message", onReady);
      resolve();
    });
    worker.addEventListener("error", (event) => reject(new Error(event.message)), { once: true });
  });
  worker.addEventListener("message", (event) => {
    const message = event.data;
    if (!validResponse(message)) return;
    const request = pending.get(message.id);
    if (request === undefined) return;
    pending.delete(message.id);
    clearTimeout(request.timeout);
    if (message.type === "compiled") {
      request.resolve({
        ok: true,
        irBytes: new Uint8Array(message.irBytes),
        irHash: message.irHash,
        executionContext: "worker",
      });
    } else {
      request.resolve({
        ok: false,
        exceptionType: message.exceptionType,
        message: message.message,
        traceback: message.traceback,
        executionContext: "worker",
      });
    }
    disposeCompilerWorker();
  });
  return worker;
}

/** @param {string} source @param {{timeoutMs?: number}} [options] @returns {Promise<CompileResult>} */
export async function compile(source, options = {}) {
  if (typeof source !== "string") throw new TypeError("model source must be a string");
  const activeWorker = compilerWorker();
  await ready;
  if (pending.size !== 0) throw new Error("Compiler worker is busy");
  const id = crypto.randomUUID();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pending.delete(id);
      reject(new Error("Model compilation timed out"));
      resetCompiler();
    }, options.timeoutMs ?? 30_000);
    pending.set(id, { resolve, reject, timeout });
    activeWorker.postMessage({ type: "compile", id, source });
  });
}

export function resetCompiler() {
  disposeCompilerWorker();
  for (const request of pending.values()) {
    clearTimeout(request.timeout);
    request.reject(new Error("Compiler worker was reset"));
  }
  pending.clear();
}

function disposeCompilerWorker() {
  worker?.terminate();
  worker = undefined;
  ready = undefined;
}

function validResponse(value) {
  if (value === null || typeof value !== "object" || typeof value.id !== "string") return false;
  if (value.type === "compiled") {
    return value.irBytes instanceof ArrayBuffer && typeof value.irHash === "string";
  }
  return value.type === "compile-error" &&
    typeof value.exceptionType === "string" &&
    typeof value.message === "string" &&
    typeof value.traceback === "string";
}
