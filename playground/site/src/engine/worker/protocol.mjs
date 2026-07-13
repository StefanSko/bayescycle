// Worker protocol documentation for the buildless ESM adapter.
// Source: src/engine/worker/protocol.ts, bayesledger @ 7346d71.

/**
 * @typedef {{
 *   type: "run",
 *   id: string,
 *   wasmUrl: string,
 *   metadataUrl: string,
 *   request: Record<string, unknown>,
 *   chainId: number,
 * }} WorkerRequest
 *
 * @typedef {{type: "started", id: string, chainId: number} |
 *   {type: "batch", id: string, chainId: number, draws: Record<string, unknown>[]} |
 *   {type: "result", id: string, chainId: number, rawBytes: Uint8Array,
 *     header?: Record<string, unknown>, trailer?: Record<string, unknown>,
 *     modelDataFingerprint?: string} |
 *   {type: "error", id: string, chainId: number,
 *     error: import("../types.mjs").EngineErrorShape}} WorkerResponse
 */

export {};
