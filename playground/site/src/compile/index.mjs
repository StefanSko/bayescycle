const PROTOCOL_VERSION = 1;
const DEFAULT_STARTUP_TIMEOUT_MS = 120_000;
const DEFAULT_COMPILE_TIMEOUT_MS = 30_000;
export const MAX_IR_BYTES = 8 * 1024 * 1024;

/**
 * Source-only disposable compiler client. Every call owns one worker and the
 * worker is terminated before the returned promise settles.
 */
export class CompilerClient {
  constructor({
    workerFactory = () => new Worker(new URL("./compiler-worker.mjs", import.meta.url), { type: "module" }),
    startupTimeoutMs = DEFAULT_STARTUP_TIMEOUT_MS,
    timeoutMs = DEFAULT_COMPILE_TIMEOUT_MS,
    maxOutputBytes = MAX_IR_BYTES,
  } = {}) {
    this.workerFactory = workerFactory;
    this.startupTimeoutMs = startupTimeoutMs;
    this.timeoutMs = timeoutMs;
    this.maxOutputBytes = maxOutputBytes;
  }

  /** @param {string} source @param {{timeoutMs?: number, signal?: AbortSignal}} [options] */
  compile(source, options = {}) {
    if (typeof source !== "string") return Promise.reject(new TypeError("model source must be a string"));

    let worker;
    try {
      worker = this.workerFactory();
    } catch (error) {
      return Promise.reject(error);
    }

    const id = crypto.randomUUID();
    const signal = options.signal;
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;

    return new Promise((resolve, reject) => {
      let settled = false;
      let disposed = false;
      let startupTimer;
      let compileTimer;

      const dispose = () => {
        if (disposed) return;
        disposed = true;
        clearTimeout(startupTimer);
        clearTimeout(compileTimer);
        signal?.removeEventListener("abort", onAbort);
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
        worker.terminate();
      };
      const fail = (error) => {
        if (settled) return;
        settled = true;
        dispose();
        reject(error);
      };
      const succeed = (value) => {
        if (settled) return;
        settled = true;
        dispose();
        resolve(value);
      };
      const onAbort = () => fail(new Error("Model compilation was cancelled"));
      const onError = (event) => fail(new Error(boundedMessage(event.message, "Compiler worker failed")));
      const onMessage = (event) => {
        const message = event.data;
        if (isReady(message)) {
          clearTimeout(startupTimer);
          compileTimer = setTimeout(
            () => fail(new Error("Model compilation timed out")),
            timeoutMs,
          );
          try {
            worker.postMessage({ type: "compile", protocol: PROTOCOL_VERSION, id, source });
          } catch (error) {
            fail(error);
          }
          return;
        }
        if (message === null || typeof message !== "object" || message.id !== id) return;
        if (validSuccess(message)) {
          const irBytes = new Uint8Array(message.irBytes);
          if (irBytes.byteLength > this.maxOutputBytes) {
            fail(new Error(`Compiler output exceeds ${this.maxOutputBytes} bytes`));
            return;
          }
          // Dispose the mutable interpreter before performing trusted hashing.
          dispose();
          void sha256(irBytes).then(
            (irHash) => succeed({ ok: true, irBytes, irHash, executionContext: "worker" }),
            fail,
          );
          return;
        }
        if (validFailure(message)) {
          succeed({
            ok: false,
            exceptionType: message.exceptionType,
            message: boundedMessage(message.message, "Compilation failed"),
            traceback: boundedMessage(message.traceback, "Compilation failed"),
            executionContext: "worker",
          });
          return;
        }
        fail(new Error("Compiler worker returned a malformed response"));
      };

      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
      signal?.addEventListener("abort", onAbort, { once: true });
      if (signal?.aborted === true) {
        onAbort();
        return;
      }
      startupTimer = setTimeout(
        () => fail(new Error("Compiler worker startup timed out")),
        this.startupTimeoutMs,
      );
    });
  }
}

/** @param {string} source @param {{timeoutMs?: number, signal?: AbortSignal}} [options] */
export function compile(source, options) {
  return new CompilerClient().compile(source, options);
}

// Kept temporarily for the pre-consolidation corpus harness; there is no
// shared production worker to reset.
export function resetCompiler() {}

function isReady(value) {
  return value !== null && typeof value === "object" && value.type === "ready" &&
    value.protocol === PROTOCOL_VERSION;
}

function validSuccess(value) {
  const keys = Object.keys(value);
  return value.type === "compiled" && value.irBytes instanceof ArrayBuffer &&
    keys.every((key) => ["type", "id", "irBytes", "irHash"].includes(key));
}

function validFailure(value) {
  const keys = Object.keys(value);
  return value.type === "compile-error" &&
    typeof value.exceptionType === "string" &&
    typeof value.message === "string" &&
    typeof value.traceback === "string" &&
    keys.every((key) => ["type", "id", "exceptionType", "message", "traceback"].includes(key));
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function boundedMessage(value, fallback) {
  const text = typeof value === "string" && value !== "" ? value : fallback;
  return text.length <= 16_384 ? text : `${text.slice(0, 16_384)}…`;
}
