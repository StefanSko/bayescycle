// Source: src/engine/abi.ts, bayesledger @ 7346d71.

import { EngineError } from "./types.mjs";

const UTF8 = new TextEncoder();
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });
const COMMIT_PATTERN = /^[0-9a-f]{40}$/u;
const SHA256_PATTERN = /^[0-9a-f]{64}$/u;

export class BayesiteAbi {
  constructor(wasm, engineVersion) {
    this.wasm = wasm;
    this.engineVersion = engineVersion;
  }

  /** @param {BufferSource} bytes @param {import("./types.mjs").EngineMetadata} metadata */
  static async instantiate(bytes, metadata) {
    validateMetadata(metadata);
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const wasm = readExports(instance.exports);
    const abi = new BayesiteAbi(wasm, metadata.engine_version);
    abi.assertCapabilitiesVersion(metadata.engine_version);
    return abi;
  }

  /** @param {Record<string, unknown>} request */
  run(request) {
    const requestBytes = encodeEngineRequest(request);
    const requestPointer = this.wasm.bayesite_alloc(requestBytes.byteLength);
    const lengthPointer = this.wasm.bayesite_alloc(4);
    let responsePointer;
    let responseLength = 0;
    try {
      new Uint8Array(this.wasm.memory.buffer, requestPointer, requestBytes.byteLength).set(
        requestBytes,
      );
      responsePointer = this.wasm.bayesite_run(
        requestPointer,
        requestBytes.byteLength,
        lengthPointer,
      );
      // bayesite_run may grow memory and detach old views.
      responseLength = new DataView(this.wasm.memory.buffer).getUint32(lengthPointer, true);
      return new Uint8Array(this.wasm.memory.buffer, responsePointer, responseLength).slice();
    } finally {
      this.wasm.bayesite_dealloc(requestPointer, requestBytes.byteLength);
      this.wasm.bayesite_dealloc(lengthPointer, 4);
      if (responsePointer !== undefined) {
        this.wasm.bayesite_dealloc(responsePointer, responseLength);
      }
    }
  }

  assertCapabilitiesVersion(expected) {
    const bytes = this.run({ command: "capabilities" });
    const text = UTF8_DECODER.decode(bytes);
    const value = parseObject(text, "capabilities response");
    const error = engineErrorFromObject(value);
    if (error !== undefined) {
      // Bayesite 0.2.1 exposes capabilities in the native CLI but its wasm
      // dispatch table predates that command. Keep the version pinned by the
      // checked ENGINE.json metadata until the wasm command is available.
      if (
        error.error === "InvalidSettings" &&
        error.message.includes('unknown command "capabilities"')
      ) {
        return;
      }
      throw error;
    }
    if (value.version !== expected) {
      throw new EngineError(
        "EngineVersionMismatch",
        `ENGINE.json expects Bayesite ${expected}, wasm reports ${String(value.version)}`,
      );
    }
  }
}

/** Preserve exact model IR bytes while framing the outer engine request. */
export function encodeEngineRequest(request) {
  const fields = Object.entries(request).map(([name, value]) => {
    const encoded = name === "model" && value instanceof Uint8Array
      ? decodeUtf8(value)
      : JSON.stringify(value);
    if (encoded === undefined) throw new EngineError("InvalidRequest", `request field ${name} is not serializable`);
    return `${JSON.stringify(name)}:${encoded}`;
  });
  return UTF8.encode(`{${fields.join(",")}}`);
}

/** @param {Uint8Array} bytes */
export function decodeUtf8(bytes) {
  return UTF8_DECODER.decode(bytes);
}

/** @param {Uint8Array} bytes */
export function engineErrorFromBytes(bytes) {
  const firstLine = decodeUtf8(bytes).split("\n", 1)[0];
  if (firstLine === undefined || firstLine.length === 0) return undefined;
  let value;
  try {
    value = JSON.parse(firstLine);
  } catch {
    return undefined;
  }
  return isObject(value) ? engineErrorFromObject(value) : undefined;
}

function engineErrorFromObject(value) {
  if (
    value.error_format === "v0-provisional" &&
    typeof value.error === "string" &&
    typeof value.message === "string"
  ) {
    return new EngineError(value.error, value.message);
  }
  return undefined;
}

function readExports(exports) {
  const memory = exports.memory;
  const alloc = exports.bayesite_alloc;
  const dealloc = exports.bayesite_dealloc;
  const run = exports.bayesite_run;
  if (
    !(memory instanceof WebAssembly.Memory) ||
    typeof alloc !== "function" ||
    typeof dealloc !== "function" ||
    typeof run !== "function"
  ) {
    throw new EngineError("InvalidWasm", "Bayesite wasm exports do not match the v0 ABI");
  }
  return {
    memory,
    bayesite_alloc: alloc,
    bayesite_dealloc: dealloc,
    bayesite_run: run,
  };
}

function validateMetadata(metadata) {
  if (
    metadata.engine_version.length === 0 ||
    !COMMIT_PATTERN.test(metadata.bayesite_commit) ||
    !SHA256_PATTERN.test(metadata.wasm_sha256)
  ) {
    throw new EngineError("InvalidEngineMetadata", "ENGINE.json is malformed");
  }
}

function parseObject(text, context) {
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    throw new EngineError("MalformedEngineResponse", `${context} is not JSON`);
  }
  if (!isObject(value)) {
    throw new EngineError("MalformedEngineResponse", `${context} must be a JSON object`);
  }
  return value;
}

function isObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
