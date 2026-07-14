const MAX_BYTES = 8 * 1024 * 1024;
const MAX_DEPTH = 64;
const UTF8 = new TextDecoder("utf-8", { fatal: true });

export class PortablePosteriorError extends Error {
  constructor(message) {
    super(message);
    this.name = "PortablePosteriorError";
  }
}

export async function validatePortablePosterior({
  modelBytes, dataBytes, posteriorBytes, requireFingerprint = true,
}) {
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
  kindScope(header, "posterior header");
  kindScope(trailer, "posterior trailer");
  const fingerprint = await modelDataFingerprint(modelBytes, dataBytes);
  const hasFingerprint = header.model_data_fingerprint !== undefined ||
    trailer.model_data_fingerprint !== undefined;
  if (
    (requireFingerprint || hasFingerprint) &&
    (header.model_data_fingerprint !== fingerprint ||
      trailer.model_data_fingerprint !== fingerprint)
  ) {
    throw new PortablePosteriorError(
      "portable posterior fingerprint does not match exact model/data bytes",
    );
  }
  const count = documents.length - 2;
  if (!Array.isArray(header.params) || !Array.isArray(header.parameter_order)) {
    throw new PortablePosteriorError("posterior parameter metadata is missing");
  }
  const parameters = header.params.map((raw, index) => {
    const parameter = object(raw, `posterior params[${index}]`);
    if (typeof parameter.name !== "string" || !Array.isArray(parameter.shape)) {
      throw new PortablePosteriorError(`posterior params[${index}] is invalid`);
    }
    const shape = parameter.shape.map((value) => integer(value, "posterior shape"));
    return Object.freeze({ name: parameter.name, shape: Object.freeze(shape) });
  });
  const names = parameters.map((parameter) => parameter.name);
  if (new Set(names).size !== names.length ||
      JSON.stringify(header.parameter_order) !== JSON.stringify(names) ||
      integer(header.parameter_count, "posterior parameter_count") !== names.length) {
    throw new PortablePosteriorError("posterior parameter order is invalid");
  }
  const settings = object(header.settings, "posterior settings");
  const drawsPerChain = positiveInteger(settings.num_draws, "posterior settings.num_draws");
  const chainOrder = integerArray(header.chain_order, "posterior chain_order");
  const chainCount = positiveInteger(header.chain_count, "posterior chain_count");
  if (chainOrder.length === 0 || new Set(chainOrder).size !== chainOrder.length ||
      chainCount !== chainOrder.length || count !== chainCount * drawsPerChain ||
      header.draw_count !== count) {
    throw new PortablePosteriorError("posterior chain topology or draw count is invalid");
  }
  if (
    JSON.stringify(trailer.parameter_order) !== JSON.stringify(names) ||
    integer(trailer.parameter_count, "posterior trailer parameter_count") !== names.length ||
    integer(trailer.params, "posterior trailer params") !== names.length ||
    integer(trailer.draw_count, "posterior trailer draw_count") !== count ||
    integer(trailer.draws_per_chain, "posterior trailer draws_per_chain") !== drawsPerChain ||
    integer(trailer.chain_count, "posterior trailer chain_count") !== chainCount ||
    JSON.stringify(integerArray(trailer.chain_order, "posterior trailer chain_order")) !==
      JSON.stringify(chainOrder)
  ) {
    throw new PortablePosteriorError("posterior trailer metadata disagrees with header");
  }
  const headerIdentity = header.posterior_identity_hash;
  const trailerIdentity = trailer.posterior_identity_hash;
  if ((headerIdentity === undefined) !== (trailerIdentity === undefined) ||
      (headerIdentity !== undefined && headerIdentity !== trailerIdentity)) {
    throw new PortablePosteriorError("posterior identity disagrees between header and trailer");
  }
  if (!Array.isArray(trailer.chains) || trailer.chains.length !== chainCount) {
    throw new PortablePosteriorError("posterior trailer chain statistics are incomplete");
  }
  trailer.chains.forEach((raw, index) => {
    const statistic = object(raw, `posterior trailer chains[${index}]`);
    if (integer(statistic.chain, "posterior trailer chain") !== chainOrder[index] ||
        integer(statistic.draw_count, "posterior trailer chain draw_count") !== drawsPerChain) {
      throw new PortablePosteriorError("posterior trailer chain statistics are out of order");
    }
  });
  const seen = new Set();
  const draws = documents.slice(1, -1).map((raw, sourceDrawIndex) => {
    const draw = object(raw, `posterior draw ${sourceDrawIndex}`);
    if (draw.draws_format !== "v0-provisional") {
      throw new PortablePosteriorError(`posterior draw ${sourceDrawIndex} format is invalid`);
    }
    kindScope(draw, `posterior draw ${sourceDrawIndex}`);
    if ((draw.draw_index ?? sourceDrawIndex) !== sourceDrawIndex) {
      throw new PortablePosteriorError("posterior draw indices are not contiguous");
    }
    const chain = integer(draw.chain, "posterior chain");
    const drawIndex = integer(draw.draw, "posterior draw");
    const expectedChain = chainOrder[Math.floor(sourceDrawIndex / drawsPerChain)];
    const expectedDraw = sourceDrawIndex % drawsPerChain;
    if (chain !== expectedChain || drawIndex !== expectedDraw) {
      throw new PortablePosteriorError("posterior draws are not grouped in chain order");
    }
    const coordinate = `${chain}:${drawIndex}`;
    if (seen.has(coordinate)) {
      throw new PortablePosteriorError("posterior chain/draw coordinates are duplicated");
    }
    seen.add(coordinate);
    if (JSON.stringify(draw.parameter_order) !== JSON.stringify(names) ||
        integer(draw.parameter_count, "posterior draw parameter_count") !== names.length) {
      throw new PortablePosteriorError("posterior draw parameter order is invalid");
    }
    const values = object(draw.values, `posterior draw ${sourceDrawIndex} values`);
    if (JSON.stringify(Object.keys(values)) !== JSON.stringify(names)) {
      throw new PortablePosteriorError("posterior draw values do not match parameter order");
    }
    const flattened = parameters.map((parameter) => {
      const numbers = flatten(values[parameter.name], parameter.name);
      const size = parameter.shape.length === 0
        ? 1
        : parameter.shape.reduce((left, right) => left * right, 1);
      if (numbers.length !== size) {
        throw new PortablePosteriorError(
          `posterior value ${parameter.name} does not match declared shape`,
        );
      }
      return Object.freeze([parameter.name, Object.freeze(numbers)]);
    });
    return Object.freeze({
      sourceDrawIndex, chain, draw: drawIndex, values: Object.freeze(flattened),
    });
  });
  return Object.freeze({
    parameters: Object.freeze(parameters),
    draws: Object.freeze(draws),
  });
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

function kindScope(document, label) {
  if (document.artifact_kind !== "posterior_draws" ||
      document.artifact_scope !== "observed_data_conditioned_parameter_draws") {
    throw new PortablePosteriorError(`${label} kind or scope is invalid`);
  }
}

function positiveInteger(value, label) {
  const result = integer(value, label);
  if (result < 1) throw new PortablePosteriorError(`${label} must be positive`);
  return result;
}

function integerArray(value, label) {
  if (!Array.isArray(value)) throw new PortablePosteriorError(`${label} must be an array`);
  return value.map((item) => integer(item, label));
}

function integer(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new PortablePosteriorError(`${label} must be a nonnegative safe integer`);
  }
  return value;
}

function flatten(value, label) {
  if (Array.isArray(value)) return value.flatMap((item) => flatten(item, label));
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new PortablePosteriorError(`posterior value ${label} must contain finite numbers`);
  }
  return [value];
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
