// Source: src/engine/types.ts, bayesledger @ 7346d71.

export const ENGINE_VERSION = "0.3.0";
export const MAX_POSTERIOR_RESPONSE_BYTES = 64 * 1024 * 1024;
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

/** @param {number} byteLength @param {number} [maximumBytes] */
export function requirePosteriorResponseWithinLimit(
  byteLength,
  maximumBytes = MAX_POSTERIOR_RESPONSE_BYTES,
) {
  if (byteLength > maximumBytes) throw posteriorTooLargeError(maximumBytes);
}

/** @param {number} [maximumBytes] */
export function posteriorTooLargeError(maximumBytes = MAX_POSTERIOR_RESPONSE_BYTES) {
  const mebibytes = maximumBytes / (1024 * 1024);
  return new EngineError(
    "PosteriorTooLarge",
    `Posterior exceeds the ${String(mebibytes)} MiB browser limit; reduce parameters or draws.`,
  );
}
