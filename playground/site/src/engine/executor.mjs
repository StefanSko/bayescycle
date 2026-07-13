// Source: src/engine/executor.ts, bayesledger @ 7346d71.

import { BayesiteAbi } from "./abi.mjs";
import { artifactBytes, parseEngineOutput } from "./stream.mjs";
import {
  ENGINE_METADATA_URL,
  ENGINE_VERSION,
  ENGINE_WASM_URL,
  EngineError,
} from "./types.mjs";

export class InProcessEngine {
  constructor(abi) {
    this.abi = abi;
  }

  get engineVersion() {
    return this.abi.engineVersion;
  }

  /**
   * @param {Uint8Array} wasmBytes
   * @param {import("./types.mjs").EngineMetadata} metadata
   */
  static async create(wasmBytes, metadata) {
    await assertWasmHash(wasmBytes, metadata.wasm_sha256);
    return new InProcessEngine(
      await BayesiteAbi.instantiate(Uint8Array.from(wasmBytes).buffer, metadata),
    );
  }

  /**
   * @param {Record<string, unknown> & {command: string}} request
   * @param {{chainId?: number, onDrawBatch?: (batch: import("./types.mjs").DrawBatch) => void}} [options]
   * @returns {Promise<import("./types.mjs").EngineOutput>}
   */
  async execute(request, options = {}) {
    await Promise.resolve();
    const bytes = artifactBytes(this.abi.run(request));
    return parseEngineOutput(
      bytes,
      options.chainId ?? 0,
      isStreamCommand(request.command),
      request.command === "sample" ? options.onDrawBatch : undefined,
    );
  }
}

export class WorkerEngine {
  constructor(
    engineVersion = ENGINE_VERSION,
    wasmUrl = ENGINE_WASM_URL,
    metadataUrl = ENGINE_METADATA_URL,
    workerFactory = () => new Worker(new URL("./worker/engine-worker.mjs", import.meta.url), {
      type: "module",
    }),
  ) {
    this.engineVersion = engineVersion;
    this.wasmUrl = wasmUrl;
    this.metadataUrl = metadataUrl;
    this.workerFactory = workerFactory;
  }

  /**
   * @param {Record<string, unknown> & {command: string}} request
   * @param {{chainId?: number, onDrawBatch?: (batch: import("./types.mjs").DrawBatch) => void}} [options]
   * @returns {Promise<import("./types.mjs").EngineOutput>}
   */
  execute(request, options = {}) {
    const worker = this.workerFactory();
    const id = globalThis.crypto.randomUUID();
    const message = {
      type: "run",
      id,
      wasmUrl: this.wasmUrl,
      metadataUrl: this.metadataUrl,
      request,
      chainId: options.chainId ?? 0,
    };
    return new Promise((resolve, reject) => {
      const malformed = () => {
        worker.terminate();
        reject(new EngineError("MalformedEngineResponse", "Engine worker returned a malformed response"));
      };
      worker.onmessage = (event) => {
        const response = event.data;
        if (response === null || typeof response !== "object") {
          malformed();
          return;
        }
        if (response.id !== id) return;
        if (response.type === "batch") {
          if (!validBatch(response)) {
            malformed();
            return;
          }
          options.onDrawBatch?.({ chainId: response.chainId, draws: response.draws });
          return;
        }
        if (response.type === "started") {
          if (!validStarted(response)) malformed();
          return;
        }
        if (response.type === "error") {
          worker.terminate();
          if (!validError(response)) {
            reject(new EngineError("MalformedEngineResponse", "Engine worker returned a malformed error"));
            return;
          }
          reject(new EngineError(response.error.error, response.error.message));
          return;
        }
        if (!validResult(response)) {
          malformed();
          return;
        }
        worker.terminate();
        resolve({
          rawBytes: response.rawBytes,
          ...(response.header === undefined ? {} : { header: response.header }),
          ...(response.trailer === undefined ? {} : { trailer: response.trailer }),
          ...(response.modelDataFingerprint === undefined
            ? {}
            : { modelDataFingerprint: response.modelDataFingerprint }),
        });
      };
      worker.onerror = (event) => {
        worker.terminate();
        reject(new EngineError("WorkerFailure", event.message));
      };
      worker.postMessage(message);
    });
  }
}

function validStarted(value) {
  return Number.isInteger(value.chainId) && exactKeys(value, ["type", "id", "chainId"]);
}

function validBatch(value) {
  return Number.isInteger(value.chainId) && Array.isArray(value.draws) &&
    value.draws.every((draw) => draw !== null && typeof draw === "object" && !Array.isArray(draw)) &&
    exactKeys(value, ["type", "id", "chainId", "draws"]);
}

function validError(value) {
  const error = value.error;
  return Number.isInteger(value.chainId) && error !== null && typeof error === "object" &&
    typeof error.error_format === "string" && typeof error.error === "string" &&
    typeof error.message === "string" && exactKeys(error, ["error_format", "error", "message"]) &&
    exactKeys(value, ["type", "id", "chainId", "error"]);
}

function validResult(value) {
  if (value.type !== "result" || !(value.rawBytes instanceof Uint8Array) || !Number.isInteger(value.chainId)) return false;
  if (value.header !== undefined && (value.header === null || typeof value.header !== "object" || Array.isArray(value.header))) return false;
  if (value.trailer !== undefined && (value.trailer === null || typeof value.trailer !== "object" || Array.isArray(value.trailer))) return false;
  if (value.modelDataFingerprint !== undefined && typeof value.modelDataFingerprint !== "string") return false;
  return exactKeys(value, ["type", "id", "chainId", "rawBytes", "header", "trailer", "modelDataFingerprint"]);
}

function exactKeys(value, allowed) {
  return Object.keys(value).every((key) => allowed.includes(key));
}

/** @param {string} command */
export function isStreamCommand(command) {
  return (
    command === "sample" ||
    command === "prior-predictive" ||
    command === "posterior-predictive"
  );
}

/** @param {unknown} value @returns {import("./types.mjs").EngineMetadata} */
export function parseEngineMetadata(value) {
  if (
    typeof value !== "object" ||
    value === null ||
    !("engine_version" in value) ||
    !("bayesite_commit" in value) ||
    !("wasm_sha256" in value) ||
    typeof value.engine_version !== "string" ||
    typeof value.bayesite_commit !== "string" ||
    typeof value.wasm_sha256 !== "string"
  ) {
    throw new EngineError("InvalidEngineMetadata", "ENGINE.json is malformed");
  }
  return {
    engine_version: value.engine_version,
    bayesite_commit: value.bayesite_commit,
    wasm_sha256: value.wasm_sha256,
  };
}

/** @param {Uint8Array} bytes @param {string} expected */
export async function assertWasmHash(bytes, expected) {
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    Uint8Array.from(bytes).buffer,
  );
  const actual = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  if (actual !== expected) {
    throw new EngineError(
      "EngineHashMismatch",
      `Bayesite wasm sha256 is ${actual}, expected ${expected}`,
    );
  }
}
