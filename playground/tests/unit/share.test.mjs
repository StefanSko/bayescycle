import {
  decodeProject,
  encodeProject,
  FRAGMENT_WARN_LENGTH,
  MAX_COMPRESSED_PAYLOAD_CHARACTERS,
  MAX_DECOMPRESSED_PAYLOAD_BYTES,
} from "/site/src/app/share.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertDeepEqual(actual, expected, message) {
  if (Object.is(actual, expected)) return;
  if (typeof actual !== "object" || actual === null || typeof expected !== "object" || expected === null) {
    throw new Error(message);
  }
  const actualKeys = Object.keys(actual);
  const expectedKeys = Object.keys(expected);
  if (actualKeys.length !== expectedKeys.length || actualKeys.some((key, index) => key !== expectedKeys[index])) {
    throw new Error(message);
  }
  for (const key of actualKeys) assertDeepEqual(actual[key], expected[key], message);
}

function sampler(overrides = {}) {
  return {
    chains: 4,
    num_warmup: 1000,
    num_draws: 2000,
    seed: 0,
    target_accept: 0.8,
    max_treedepth: 10,
    ...overrides,
  };
}

function fixedNoise(length) {
  let state = 0x5eed1234;
  let value = "";
  const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()";
  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    value += alphabet[state % alphabet.length];
  }
  return value;
}

const longProject = {
  v: 1,
  source: `# ${fixedNoise(12_000)}`,
  dataMode: "observed",
  sampler: sampler({ seed: 12000 }),
};

const projects = [
  {
    v: 1,
    source: "@model\\nclass Minimal:\\n    pass\\n",
    dataMode: "observed",
    sampler: sampler(),
  },
  longProject,
  {
    v: 1,
    source: "μ ~ 正规(0,1) 🎲",
    dataMode: "design",
    sampler: sampler({ chains: 2 }),
  },
  {
    v: 1,
    source: "@model\\nclass Designed:\\n    pass\\n",
    dataMode: "design",
    design: {
      M: { low: -3.5, high: 8.25, n: 500 },
      A: { low: -10, high: 10, n: 500 },
      nested_name: { low: 0.001, high: 9999, n: 73 },
    },
    truth: { alpha: -0.25, beta_m: 1.75, beta_a: -4.5, sigma: 0.125 },
    sampler: sampler({ chains: 8, num_warmup: 2500, num_draws: 4000, seed: 991 }),
  },
  {
    v: 1,
    source: "# sampler-only settings case",
    dataMode: "observed",
    sampler: sampler({ chains: 1, num_warmup: 0, num_draws: 1, seed: 4294967295, target_accept: 0.99, max_treedepth: 18 }),
  },
  {
    v: 1,
    source: " ",
    dataMode: "observed",
    sampler: sampler({ chains: 1, num_warmup: 0, num_draws: 1 }),
  },
];

async function independentPayloadLength(project) {
  const bytes = new TextEncoder().encode(JSON.stringify(project));
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate-raw"));
  const compressed = new Uint8Array(await new Response(stream).arrayBuffer());
  let binary = "";
  for (const byte of compressed) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "").length;
}

function bytesPayload(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

async function compressedJsonPayload(value) {
  const input = new Blob([new TextEncoder().encode(JSON.stringify(value))]).stream();
  const compressed = input.pipeThrough(new CompressionStream("deflate-raw"));
  return bytesPayload(new Uint8Array(await new Response(compressed).arrayBuffer()));
}

export default [
  {
    name: "round-trips projects byte-equal",
    fn: async () => {
      for (const [index, project] of projects.entries()) {
        const payload = await encodeProject(project);
        assert(/^[A-Za-z0-9_-]+$/.test(payload), `project ${index} payload is not unpadded base64url`);
        const decoded = await decodeProject(payload);
        assertDeepEqual(decoded, project, `project ${index} deep equality differs`);
        assert(JSON.stringify(decoded) === JSON.stringify(project), `project ${index} JSON bytes differ`);
      }
    },
  },
  {
    name: "rejects garbage",
    fn: async () => {
      for (const payload of ["not-base64!!!", bytesPayload(new Uint8Array([3, 17, 99, 241, 0, 88]))]) {
        let threw = false;
        try {
          await decodeProject(payload);
        } catch {
          threw = true;
        }
        assert(threw, `garbage payload ${payload} was accepted`);
      }

      const versionTwo = {
        v: 2,
        source: "@model\\nclass Future:\\n    pass\\n",
        dataMode: "observed",
        sampler: sampler(),
      };
      let versionMessage = "";
      try {
        await decodeProject(await encodeProject(versionTwo));
      } catch (error) {
        versionMessage = String(error instanceof Error ? error.message : error);
      }
      assert(/version/i.test(versionMessage), `version rejection was unclear: ${versionMessage}`);
    },
  },
  {
    name: "rejects compressed payloads above the character cap",
    fn: async () => {
      let boundaryMessage = "";
      try {
        await decodeProject("A".repeat(MAX_COMPRESSED_PAYLOAD_CHARACTERS));
      } catch (error) { boundaryMessage = String(error); }
      assert(!boundaryMessage.includes("compressed payload exceeds"), "exact compressed character cap was rejected as oversized");

      let message = "";
      try {
        await decodeProject("A".repeat(MAX_COMPRESSED_PAYLOAD_CHARACTERS + 1));
      } catch (error) {
        message = String(error);
      }
      assert(message.includes(String(MAX_COMPRESSED_PAYLOAD_CHARACTERS)), `compressed cap was unclear: ${message}`);
      assert(message.includes("compressed payload"), `compressed cap domain was unclear: ${message}`);
    },
  },
  {
    name: "encoder never emits a payload above decoder limits",
    fn: async () => {
      const emptySize = new TextEncoder().encode(JSON.stringify({ v: 1, source: "" })).byteLength;
      const boundary = {
        v: 1,
        source: "x".repeat(MAX_DECOMPRESSED_PAYLOAD_BYTES - emptySize),
      };
      assertDeepEqual(
        await decodeProject(await encodeProject(boundary)),
        boundary,
        "exact decompressed byte cap did not round-trip",
      );

      let message = "";
      try {
        await encodeProject({
          v: 1,
          source: "x".repeat(MAX_DECOMPRESSED_PAYLOAD_BYTES + 1),
        });
      } catch (error) { message = String(error); }
      assert(message.includes(String(MAX_DECOMPRESSED_PAYLOAD_BYTES)), `encoder cap was unclear: ${message}`);
      const valid = { v: 1, source: "still shareable" };
      assertDeepEqual(await decodeProject(await encodeProject(valid)), valid, "encoder emitted an invalid payload");
    },
  },
  {
    name: "rejects streaming decompression above the byte cap",
    fn: async () => {
      const payload = await compressedJsonPayload({
        v: 1,
        source: "x".repeat(MAX_DECOMPRESSED_PAYLOAD_BYTES + 1),
      });
      assert(payload.length < MAX_COMPRESSED_PAYLOAD_CHARACTERS, "bomb did not isolate decompression cap");
      let message = "";
      try { await decodeProject(payload); } catch (error) { message = String(error); }
      assert(message.includes(String(MAX_DECOMPRESSED_PAYLOAD_BYTES)), `decompressed cap was unclear: ${message}`);
      assert(message.includes("decompressed payload"), `decompressed cap domain was unclear: ${message}`);
    },
  },
  {
    name: "warn threshold",
    fn: async () => {
      assert(FRAGMENT_WARN_LENGTH === 8000, `warning threshold is ${FRAGMENT_WARN_LENGTH}`);
      const payload = await encodeProject(longProject);
      const independentlyComputedLength = await independentPayloadLength(longProject);
      assert(payload.length === independentlyComputedLength, "12 kB payload length differs from independent encoding");
      const shouldWarn = independentlyComputedLength > FRAGMENT_WARN_LENGTH;
      assert((payload.length > FRAGMENT_WARN_LENGTH) === shouldWarn, "12 kB payload threshold comparison differs");
    },
  },
];
