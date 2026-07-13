// Source: src/engine/stream.ts, bayesledger @ 7346d71.

import { decodeUtf8, engineErrorFromBytes } from "./abi.mjs";
import { EngineError } from "./types.mjs";

const FINGERPRINT_PATTERN = /^sha256:[0-9a-f]{64}$/u;
export const DRAW_BATCH_SIZE = 64;

/** Bayesite's file artifacts conventionally end in LF; the wasm text ABI omits it. */
export function artifactBytes(responseBytes) {
  if (responseBytes.at(-1) === 0x0a) return responseBytes;
  const bytes = new Uint8Array(responseBytes.byteLength + 1);
  bytes.set(responseBytes);
  bytes[bytes.byteLength - 1] = 0x0a;
  return bytes;
}

/**
 * @param {Uint8Array} rawBytes
 * @param {number} chainId
 * @param {boolean} streamDraws
 * @param {(batch: import("./types.mjs").DrawBatch) => void} [onDrawBatch]
 * @returns {import("./types.mjs").EngineOutput}
 */
export function parseEngineOutput(rawBytes, chainId, streamDraws, onDrawBatch) {
  const typedError = engineErrorFromBytes(rawBytes);
  if (typedError !== undefined) throw typedError;

  const text = decodeUtf8(rawBytes);
  if (!streamDraws) {
    const report = parseLineObject(text, "engine report");
    return output(rawBytes, undefined, report, fingerprintFrom(report));
  }

  const lines = text.split("\n");
  if (lines.at(-1) === "") lines.pop();
  if (lines.length < 2) {
    throw new EngineError("MalformedEngineResponse", "NDJSON needs a header and trailer");
  }
  const first = lines[0];
  const last = lines.at(-1);
  if (first === undefined || last === undefined) {
    throw new EngineError("MalformedEngineResponse", "NDJSON response is empty");
  }
  const header = parseLineObject(first, "NDJSON header");
  const trailerEnvelope = parseLineObject(last, "NDJSON trailer");
  const trailerValue = trailerEnvelope.trailer;
  if (!isJsonObject(trailerValue)) {
    throw new EngineError("MalformedEngineResponse", "NDJSON final line has no trailer object");
  }

  let batch = [];
  for (const [offset, line] of lines.slice(1, -1).entries()) {
    const draw = parseLineObject(line, `NDJSON draw ${String(offset)}`);
    validatePerDrawV2(header, draw);
    batch.push(draw);
    if (batch.length >= DRAW_BATCH_SIZE) {
      onDrawBatch?.({ chainId, draws: batch });
      batch = [];
    }
  }
  if (batch.length > 0) onDrawBatch?.({ chainId, draws: batch });

  return output(
    rawBytes,
    header,
    trailerValue,
    fingerprintFrom(trailerValue) ?? fingerprintFrom(header),
  );
}

function validatePerDrawV2(header, draw) {
  if (header.sample_stats_mode !== "per_draw_v2") return;
  if (
    draw.sample_stats_mode !== "per_draw_v2" ||
    typeof draw.diverging !== "boolean" ||
    !Number.isInteger(draw.tree_depth) ||
    typeof draw.tree_accept !== "number" ||
    !Number.isFinite(draw.tree_accept) ||
    typeof draw.energy !== "number" ||
    !Number.isFinite(draw.energy)
  ) {
    throw new EngineError(
      "MalformedEngineResponse",
      "per_draw_v2 draw has invalid sampler stats",
    );
  }
}

function output(rawBytes, header, trailer, modelDataFingerprint) {
  return {
    rawBytes,
    ...(header === undefined ? {} : { header }),
    ...(trailer === undefined ? {} : { trailer }),
    ...(modelDataFingerprint === undefined ? {} : { modelDataFingerprint }),
  };
}

function fingerprintFrom(value) {
  const fingerprint = value.model_data_fingerprint;
  return typeof fingerprint === "string" && FINGERPRINT_PATTERN.test(fingerprint)
    ? fingerprint
    : undefined;
}

function parseLineObject(text, context) {
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    throw new EngineError("MalformedEngineResponse", `${context} is not valid JSON`);
  }
  if (!isJsonObject(value)) {
    throw new EngineError("MalformedEngineResponse", `${context} must be an object`);
  }
  return value;
}

function isJsonObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
