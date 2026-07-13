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
  ) {
    this.engineVersion = engineVersion;
    this.wasmUrl = wasmUrl;
    this.metadataUrl = metadataUrl;
  }

  /**
   * @param {Record<string, unknown> & {command: string}} request
   * @param {{chainId?: number, onDrawBatch?: (batch: import("./types.mjs").DrawBatch) => void}} [options]
   * @returns {Promise<import("./types.mjs").EngineOutput>}
   */
  execute(request, options = {}) {
    const worker = new Worker(new URL("./worker/engine-worker.mjs", import.meta.url), {
      type: "module",
    });
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
      worker.onmessage = (event) => {
        const response = event.data;
        if (response.id !== id) return;
        if (response.type === "batch") {
          options.onDrawBatch?.({ chainId: response.chainId, draws: response.draws });
          return;
        }
        if (response.type === "started") return;
        worker.terminate();
        if (response.type === "error") {
          reject(new EngineError(response.error.error, response.error.message));
          return;
        }
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
