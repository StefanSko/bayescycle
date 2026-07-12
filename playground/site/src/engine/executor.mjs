// Source: src/engine/executor.ts, bayesledger @ 7346d71.

import { BayesiteAbi } from "./abi.mjs";
import { artifactBytes, parseEngineOutput } from "./stream.mjs";
import { EngineError } from "./types.mjs";

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
