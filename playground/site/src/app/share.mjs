export const FRAGMENT_WARN_LENGTH = 8000;

const BASE64URL = /^[A-Za-z0-9_-]+$/;

export async function encodeProject(project) {
  const json = JSON.stringify(project);
  const input = new Blob([new TextEncoder().encode(json)]).stream();
  const compressed = input.pipeThrough(new CompressionStream("deflate-raw"));
  const bytes = new Uint8Array(await new Response(compressed).arrayBuffer());
  return bytesToBase64(bytes)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "");
}

export async function decodeProject(payload) {
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
    json = new TextDecoder("utf-8", { fatal: true }).decode(
      await new Response(decompressed).arrayBuffer(),
    );
  } catch (error) {
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

function invalidPayload(reason, cause) {
  return new Error(`Invalid shared project payload: ${reason}`, { cause });
}
