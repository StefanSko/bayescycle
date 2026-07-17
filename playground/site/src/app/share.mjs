export const FRAGMENT_WARN_LENGTH = 8000;
export const MAX_COMPRESSED_PAYLOAD_CHARACTERS = 65_536;
export const MAX_DECOMPRESSED_PAYLOAD_BYTES = 1024 * 1024;

const BASE64URL = /^[A-Za-z0-9_-]+$/;

export async function encodeProject(project) {
  const json = JSON.stringify(project);
  if (exceedsUtf8Bytes(json, MAX_DECOMPRESSED_PAYLOAD_BYTES)) {
    throw invalidPayload(
      `decompressed payload exceeds ${MAX_DECOMPRESSED_PAYLOAD_BYTES} bytes`,
    );
  }
  const input = new Blob([new TextEncoder().encode(json)]).stream();
  const compressed = input.pipeThrough(new CompressionStream("deflate-raw"));
  const bytes = new Uint8Array(await new Response(compressed).arrayBuffer());
  const payload = bytesToBase64(bytes)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "");
  if (payload.length > MAX_COMPRESSED_PAYLOAD_CHARACTERS) {
    throw invalidPayload(
      `compressed payload exceeds ${MAX_COMPRESSED_PAYLOAD_CHARACTERS} characters`,
    );
  }
  return payload;
}

export async function decodeProject(payload) {
  if (typeof payload !== "string") {
    throw invalidPayload("malformed base64url");
  }
  if (payload.length > MAX_COMPRESSED_PAYLOAD_CHARACTERS) {
    throw invalidPayload(
      `compressed payload exceeds ${MAX_COMPRESSED_PAYLOAD_CHARACTERS} characters`,
    );
  }
  let bytes;
  try {
    bytes = decodeBase64url(payload);
  } catch (error) {
    throw invalidPayload("malformed base64url", error);
  }

  let json;
  try {
    const input = new Blob([bytes]).stream();
    const decompressed = input.pipeThrough(new DecompressionStream("deflate-raw"));
    const decompressedBytes = await readBounded(
      decompressed,
      MAX_DECOMPRESSED_PAYLOAD_BYTES,
    );
    json = new TextDecoder("utf-8", { fatal: true }).decode(decompressedBytes);
  } catch (error) {
    if (error instanceof DecompressedPayloadTooLarge) {
      throw invalidPayload(error.message, error);
    }
    throw invalidPayload("malformed deflate data", error);
  }

  let project;
  try {
    project = JSON.parse(json);
  } catch (error) {
    throw invalidPayload("malformed JSON", error);
  }
  if (typeof project !== "object" || project === null || Array.isArray(project)) {
    throw new Error("Invalid shared project payload: JSON must contain a project object");
  }
  if (project.v !== 1) {
    throw new Error(`Unsupported shared project version: ${String(project.v)}`);
  }
  return project;
}

async function readBounded(stream, maximumBytes) {
  const reader = stream.getReader();
  const chunks = [];
  let byteLength = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      byteLength += value.byteLength;
      if (byteLength > maximumBytes) {
        await reader.cancel("decompressed payload too large");
        throw new DecompressedPayloadTooLarge(
          `decompressed payload exceeds ${maximumBytes} bytes`,
        );
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

class DecompressedPayloadTooLarge extends Error {}

function decodeBase64url(payload) {
  if (typeof payload !== "string" || !BASE64URL.test(payload) || payload.length % 4 === 1) {
    throw new Error("invalid base64url characters or length");
  }
  const padding = "=".repeat((4 - (payload.length % 4)) % 4);
  const binary = atob(payload.replaceAll("-", "+").replaceAll("_", "/") + padding);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function exceedsUtf8Bytes(value, maximumBytes) {
  let byteLength = 0;
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    byteLength += codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 :
      codePoint <= 0xffff ? 3 : 4;
    if (byteLength > maximumBytes) return true;
  }
  return false;
}

function invalidPayload(reason, cause) {
  return new Error(`Invalid shared project payload: ${reason}`, { cause });
}
