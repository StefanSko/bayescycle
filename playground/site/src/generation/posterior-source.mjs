const MAX_BYTES = 8 * 1024 * 1024;
const MAX_DEPTH = 64;
const UTF8 = new TextDecoder("utf-8", { fatal: true });

export class PortablePosteriorError extends Error {
  constructor(message) {
    super(message);
    this.name = "PortablePosteriorError";
  }
}

export async function validatePortablePosterior({ modelBytes, dataBytes, posteriorBytes }) {
  if (!(posteriorBytes instanceof Uint8Array) ||
      posteriorBytes.byteLength === 0 || posteriorBytes.byteLength > MAX_BYTES) {
    throw new PortablePosteriorError("portable posterior source exceeds its byte bound");
  }
  if (posteriorBytes.at(-1) !== 0x0a) {
    throw new PortablePosteriorError("portable posterior source must end in LF");
  }
  const lines = splitLines(posteriorBytes);
  if (lines.length < 3) {
    throw new PortablePosteriorError(
      "portable posterior source needs header, draws, and trailer fingerprint",
    );
  }
  const documents = lines.map((line, index) => {
    if (line.byteLength === 0 || line.byteLength + 1 > MAX_BYTES) {
      throw new PortablePosteriorError(`portable posterior line ${index + 1} is empty or oversized`);
    }
    validateDepth(line, index + 1);
    try {
      return JSON.parse(UTF8.decode(line));
    } catch (error) {
      throw new PortablePosteriorError(
        `portable posterior line ${index + 1} is invalid JSON: ${String(error)}`,
      );
    }
  });
  const header = object(documents[0], "portable posterior header");
  const envelope = object(documents.at(-1), "portable posterior trailer envelope");
  if (Object.keys(envelope).length !== 1 || envelope.trailer === undefined) {
    throw new PortablePosteriorError("portable posterior trailer envelope is invalid");
  }
  const trailer = object(envelope.trailer, "portable posterior trailer");
  if (header.draws_format !== "v0-provisional" || trailer.draws_format !== "v0-provisional") {
    throw new PortablePosteriorError("portable posterior format is invalid");
  }
  const fingerprint = await modelDataFingerprint(modelBytes, dataBytes);
  if (
    header.model_data_fingerprint !== fingerprint ||
    trailer.model_data_fingerprint !== fingerprint
  ) {
    throw new PortablePosteriorError(
      "portable posterior fingerprint does not match exact model/data bytes",
    );
  }
  const count = documents.length - 2;
  if (header.draw_count !== count || trailer.draw_count !== count || count < 1) {
    throw new PortablePosteriorError("portable posterior draw count is incomplete");
  }
  return Object.freeze({ header, draws: Object.freeze(documents.slice(1, -1)), trailer });
}

async function modelDataFingerprint(modelBytes, dataBytes) {
  const prefix = new TextEncoder().encode("bayescycle-model-data-v1\n");
  const framed = new Uint8Array(prefix.length + modelBytes.length + 1 + dataBytes.length);
  framed.set(prefix);
  framed.set(modelBytes, prefix.length);
  framed[prefix.length + modelBytes.length] = 0x0a;
  framed.set(dataBytes, prefix.length + modelBytes.length + 1);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", framed));
  return `sha256:${[...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function splitLines(bytes) {
  const lines = [];
  let start = 0;
  for (let index = 0; index < bytes.length; index += 1) {
    if (bytes[index] === 0x0a) {
      lines.push(bytes.slice(start, index));
      start = index + 1;
    }
  }
  return lines;
}

function object(value, label) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new PortablePosteriorError(`${label} must be an object`);
  }
  return value;
}

function validateDepth(line, lineNumber) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (const byte of line) {
    if (inString) {
      if (escaped) escaped = false;
      else if (byte === 0x5c) escaped = true;
      else if (byte === 0x22) inString = false;
    } else if (byte === 0x22) inString = true;
    else if (byte === 0x7b || byte === 0x5b) {
      depth += 1;
      if (depth > MAX_DEPTH) {
        throw new PortablePosteriorError(`portable posterior line ${lineNumber} exceeds depth`);
      }
    } else if (byte === 0x7d || byte === 0x5d) {
      depth -= 1;
    }
  }
  if (inString || depth !== 0) {
    throw new PortablePosteriorError(`portable posterior line ${lineNumber} has malformed nesting`);
  }
}
