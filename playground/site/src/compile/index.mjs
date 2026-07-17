const PROTOCOL_VERSION = 1;
const DEFAULT_STARTUP_TIMEOUT_MS = 120_000;
const DEFAULT_COMPILE_TIMEOUT_MS = 30_000;
export const MAX_IR_BYTES = 8 * 1024 * 1024;
export const MAX_MODEL_SOURCE_BYTES = 1024 * 1024;
export const MAX_MODEL_SCHEMA_ENTRIES = 1024;
export const MAX_MODEL_SCHEMA_TEXT_CHARACTERS = 512;

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
    if (exceedsUtf8Bytes(source, MAX_MODEL_SOURCE_BYTES)) {
      return Promise.reject(new Error(
        `Model source exceeds maximum UTF-8 size of ${MAX_MODEL_SOURCE_BYTES} bytes`,
      ));
    }
    return this.#request({ source }, options, "Model compilation");
  }

  /**
   * @param {string} source
   * @param {string} priorSource
   * @param {{timeoutMs?: number, signal?: AbortSignal}} [options]
   */
  compileScenario(source, priorSource, options = {}) {
    if (typeof source !== "string") return Promise.reject(new TypeError("model source must be a string"));
    if (typeof priorSource !== "string") {
      return Promise.reject(new TypeError("prior-only source must be a string"));
    }
    if (exceedsUtf8Bytes(source, MAX_MODEL_SOURCE_BYTES)) {
      return Promise.reject(new Error(
        `Model source exceeds maximum UTF-8 size of ${MAX_MODEL_SOURCE_BYTES} bytes`,
      ));
    }
    if (exceedsUtf8Bytes(priorSource, MAX_MODEL_SOURCE_BYTES)) {
      return Promise.reject(new Error(
        `Prior-only source exceeds maximum UTF-8 size of ${MAX_MODEL_SOURCE_BYTES} bytes`,
      ));
    }
    if (exceedsCombinedUtf8Bytes(source, priorSource, MAX_MODEL_SOURCE_BYTES)) {
      return Promise.reject(new Error(
        `Model source and prior-only source together exceed maximum UTF-8 size of ${MAX_MODEL_SOURCE_BYTES} bytes`,
      ));
    }
    return this.#request(
      { source, mode: "with-prior", priorSource },
      options,
      "Prior composition",
    );
  }

  #request(payload, options, operationLabel) {
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
      const cancellationError = () => new Error(`${operationLabel} was cancelled`);
      const onAbort = () => fail(cancellationError());
      const onError = (event) => fail(new Error(boundedMessage(event.message, "Compiler worker failed")));
      const onMessage = (event) => {
        const message = event.data;
        if (isReady(message)) {
          clearTimeout(startupTimer);
          compileTimer = setTimeout(
            () => fail(new Error(`${operationLabel} timed out`)),
            timeoutMs,
          );
          try {
            worker.postMessage({
              type: "compile", protocol: PROTOCOL_VERSION, id, ...payload,
            });
          } catch (error) {
            fail(error);
          }
          return;
        }
        if (message === null || typeof message !== "object" || message.id !== id) return;
        const expectsTarget = payload.mode === "with-prior";
        if (validSuccess(message, expectsTarget)) {
          const irBytes = new Uint8Array(message.irBytes);
          const targetIrBytes = expectsTarget
            ? new Uint8Array(message.targetIrBytes)
            : null;
          let modelSchema;
          let schemaBytes;
          try {
            modelSchema = validateModelSchema(message.modelSchema);
            schemaBytes = new TextEncoder().encode(JSON.stringify(modelSchema));
          } catch (error) {
            fail(error);
            return;
          }
          if (irBytes.byteLength > this.maxOutputBytes ||
              schemaBytes.byteLength > this.maxOutputBytes ||
              (targetIrBytes?.byteLength ?? 0) > this.maxOutputBytes) {
            fail(new Error(`Compiler output exceeds ${this.maxOutputBytes} bytes`));
            return;
          }
          // Dispose the mutable interpreter before performing trusted hashing.
          dispose();
          const hashes = targetIrBytes === null
            ? Promise.all([sha256(irBytes)])
            : Promise.all([sha256(irBytes), sha256(targetIrBytes)]);
          void hashes.then(
            ([irHash, targetIrHash]) => {
              if (signal?.aborted === true) {
                fail(cancellationError());
                return;
              }
              succeed({
                ok: true, irBytes, irHash, modelSchema, executionContext: "worker",
                ...(targetIrBytes === null ? {} : { targetIrBytes, targetIrHash }),
              });
            },
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

/**
 * @param {string} source
 * @param {string} priorSource
 * @param {{timeoutMs?: number, signal?: AbortSignal}} [options]
 */
export function compileScenario(source, priorSource, options) {
  return new CompilerClient().compileScenario(source, priorSource, options);
}

function isReady(value) {
  return value !== null && typeof value === "object" && value.type === "ready" &&
    value.protocol === PROTOCOL_VERSION;
}

function validSuccess(value, expectsTarget) {
  const keys = Object.keys(value);
  const required = ["type", "id", "irBytes", "modelSchema"];
  const allowed = [...required, "irHash"];
  if (expectsTarget) {
    required.push("targetIrBytes");
    allowed.push("targetIrBytes");
  }
  return value.type === "compiled" && value.irBytes instanceof ArrayBuffer &&
    (!expectsTarget || value.targetIrBytes instanceof ArrayBuffer) &&
    value.modelSchema !== null && typeof value.modelSchema === "object" &&
    required.every((key) => keys.includes(key)) &&
    keys.every((key) => allowed.includes(key));
}

export function validateModelSchema(value, maxSchemaBytes = MAX_IR_BYTES) {
  requireObjectKeys(value, ["schema_format", "parameters", "data", "observed"], "model schema");
  if (value.schema_format !== "bayescycle.playground.model-schema.v0") {
    throw malformedSchema("unsupported schema_format");
  }
  if (!Array.isArray(value.parameters) || !Array.isArray(value.data) ||
      !Array.isArray(value.observed)) {
    throw malformedSchema("parameters, data, and observed must be arrays");
  }
  const entryCount = value.parameters.length + value.data.length + value.observed.length;
  if (entryCount > MAX_MODEL_SCHEMA_ENTRIES) {
    throw malformedSchema(
      `parameters, data, and observed exceed ${MAX_MODEL_SCHEMA_ENTRIES} total entries`,
    );
  }
  let shapeByteBudget = maxSchemaBytes;
  const parameterNames = new Set();
  const parameters = value.parameters.map((parameter, index) => {
    requireObjectKeys(
      parameter,
      ["name", "prior", "constraint", "shape", "default"],
      `parameter ${index}`,
    );
    requireName(parameter.name, parameterNames, `parameter ${index}`);
    if (typeof parameter.prior !== "string" || parameter.prior === "") {
      throw malformedSchema(`parameter ${index} prior must be non-empty text`);
    }
    requireBoundedText(parameter.prior, `parameter ${index} prior`);
    if (parameter.constraint !== null && typeof parameter.constraint !== "string") {
      throw malformedSchema(`parameter ${index} constraint must be text or null`);
    }
    if (parameter.constraint !== null) {
      requireBoundedText(parameter.constraint, `parameter ${index} constraint`);
    }
    if (!Array.isArray(parameter.shape)) {
      throw malformedSchema(`parameter ${index} shape must contain non-empty strings`);
    }
    if (parameter.shape.length * 3 > shapeByteBudget) {
      throw malformedSchema("parameter shapes exceed compiler output byte limit");
    }
    for (const dimension of parameter.shape) {
      if (typeof dimension !== "string" || dimension === "") {
        throw malformedSchema(`parameter ${index} shape must contain non-empty strings`);
      }
      requireBoundedText(dimension, `parameter ${index} shape dimension`);
      shapeByteBudget -= utf8BytesUpTo(dimension, shapeByteBudget) + 3;
      if (shapeByteBudget < 0) {
        throw malformedSchema("parameter shapes exceed compiler output byte limit");
      }
    }
    if (parameter.default !== null &&
        (typeof parameter.default !== "number" || !Number.isFinite(parameter.default))) {
      throw malformedSchema(`parameter ${index} default must be a finite number or null`);
    }
    return Object.freeze({ ...parameter, shape: Object.freeze([...parameter.shape]) });
  });
  const dataNames = new Set();
  const data = value.data.map((entry, index) => {
    requireObjectKeys(entry, ["name", "dtype", "kind", "length"], `data ${index}`);
    requireName(entry.name, dataNames, `data ${index}`);
    if (!["bool", "int32", "int64", "float32", "float64"].includes(entry.dtype)) {
      throw malformedSchema(`data ${index} dtype is unsupported`);
    }
    if (!["scalar", "vector", "matrix", "array"].includes(entry.kind)) {
      throw malformedSchema(`data ${index} kind is unsupported`);
    }
    if (entry.length !== null && (!Number.isSafeInteger(entry.length) || entry.length < 0)) {
      throw malformedSchema(`data ${index} length must be a non-negative integer or null`);
    }
    return Object.freeze({ ...entry });
  });
  const observedNames = new Set();
  const observed = value.observed.map((entry, index) => {
    requireObjectKeys(entry, ["name"], `observed ${index}`);
    requireName(entry.name, observedNames, `observed ${index}`);
    return Object.freeze({ ...entry });
  });
  return Object.freeze({
    schema_format: value.schema_format,
    parameters: Object.freeze(parameters),
    data: Object.freeze(data),
    observed: Object.freeze(observed),
  });
}

function requireObjectKeys(value, expected, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw malformedSchema(`${label} must be an object`);
  }
  const wanted = new Set(expected);
  let keyCount = 0;
  for (const key in value) {
    if (!Object.hasOwn(value, key)) continue;
    keyCount += 1;
    if (keyCount > expected.length || !wanted.has(key)) {
      throw malformedSchema(`${label} requires exactly ${[...wanted].sort().join(", ")}`);
    }
  }
  if (keyCount !== expected.length) {
    throw malformedSchema(`${label} requires exactly ${[...wanted].sort().join(", ")}`);
  }
}

function requireName(name, names, label) {
  if (typeof name !== "string" || name === "") {
    throw malformedSchema(`${label} name must be non-empty text`);
  }
  requireBoundedText(name, `${label} name`);
  if (names.has(name)) throw malformedSchema(`duplicate name ${name}`);
  names.add(name);
}

function requireBoundedText(value, label) {
  let characters = 0;
  for (const _character of value) {
    characters += 1;
    if (characters > MAX_MODEL_SCHEMA_TEXT_CHARACTERS) {
      throw malformedSchema(
        `${label} exceeds ${MAX_MODEL_SCHEMA_TEXT_CHARACTERS} characters`,
      );
    }
  }
}

function malformedSchema(detail) {
  return new Error(`Compiler worker returned a malformed model schema: ${detail}`);
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

function utf8BytesUpTo(value, maximumBytes) {
  let byteLength = 0;
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    byteLength += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 :
      codePoint <= 0xffff ? 3 : 4;
    if (byteLength > maximumBytes) return maximumBytes + 1;
  }
  return byteLength;
}

function exceedsUtf8Bytes(value, maximumBytes) {
  return utf8BytesUpTo(value, maximumBytes) > maximumBytes;
}

function exceedsCombinedUtf8Bytes(left, right, maximumBytes) {
  const leftBytes = utf8BytesUpTo(left, maximumBytes);
  if (leftBytes > maximumBytes) return true;
  return utf8BytesUpTo(right, maximumBytes - leftBytes) > maximumBytes - leftBytes;
}

function boundedMessage(value, fallback) {
  const text = typeof value === "string" && value !== "" ? value : fallback;
  return text.length <= 16_384 ? text : `${text.slice(0, 16_384)}…`;
}
