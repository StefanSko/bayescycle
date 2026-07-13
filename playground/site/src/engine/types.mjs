// Source: src/engine/types.ts, bayesledger @ 7346d71.

export const ENGINE_VERSION = "0.2.1";
export const ENGINE_WASM_URL = new URL(
  "../../vendor/bayesite/bayesite_core.wasm",
  import.meta.url,
).href;
export const ENGINE_METADATA_URL = new URL(
  "../../vendor/bayesite/ENGINE.json",
  import.meta.url,
).href;

/**
 * @typedef {{engine_version: string, bayesite_commit: string, wasm_sha256: string}} EngineMetadata
 * @typedef {{chainId: number, draws: Array<Record<string, unknown>>}} DrawBatch
 * @typedef {{rawBytes: Uint8Array, header?: Record<string, unknown>, trailer?: Record<string, unknown>, modelDataFingerprint?: string}} EngineOutput
 * @typedef {{error_format: "v0-provisional", error: string, message: string}} EngineErrorShape
 */

export class EngineError extends Error {
  /** @param {string} error @param {string} message */
  constructor(error, message) {
    super(message);
    this.name = error;
    this.error_format = "v0-provisional";
    this.error = error;
  }
}
