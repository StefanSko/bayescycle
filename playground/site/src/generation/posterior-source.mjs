import { MAX_GENERATION_INPUT_BYTES } from "./limits.mjs";
import { parseStrictJson } from "./strict-json.mjs";

const MAX_BYTES = MAX_GENERATION_INPUT_BYTES;
const MAX_DEPTH = 64;
const UTF8 = new TextDecoder("utf-8", { fatal: true });
const SAMPLE_WORKFLOW_PHASES = Object.freeze([
  "parse_json",
  "decode_ir",
  "bind_data",
  "build_posterior_state",
  "evaluate_logp_grad",
  "run_nuts",
  "emit_artifact",
]);

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
      const document = parseStrictJson(UTF8.decode(line), `portable posterior line ${index + 1}`, {
        integerKeys: [
          "draw_index", "chain", "draw", "draw_count", "draws_per_chain",
          "chain_count", "parameter_count", "params", "num_draws", "num_warmup",
          "max_treedepth", "tree_depth", "divergences", "chains", "seed",
        ],
        integerArrayKeys: [
          "shape", "chain_order", "coordinate_order", "treedepth_histogram",
        ],
        unrestrictedObjectKeys: ["values", "rhat", "ess"],
      });
      validateFinite(document);
      return document;
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
  if (JSON.stringify(header.workflow_phases) !== JSON.stringify(SAMPLE_WORKFLOW_PHASES) ||
      JSON.stringify(trailer.workflow_phases) !== JSON.stringify(SAMPLE_WORKFLOW_PHASES)) {
    throw new PortablePosteriorError("posterior workflow phases are invalid");
  }
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
      JSON.stringify(header.packing) !== JSON.stringify(names) ||
      integer(header.parameter_count, "posterior parameter_count") !== names.length) {
    throw new PortablePosteriorError("posterior parameter order or packing is invalid");
  }
  const headerSeed = integer(header.seed, "posterior seed");
  const settings = object(header.settings, "posterior settings");
  integer(settings.num_warmup, "posterior settings.num_warmup");
  const targetAccept = finiteNumber(
    settings.target_accept, "posterior settings.target_accept",
  );
  if (!(targetAccept > 0.0 && targetAccept < 1.0)) {
    throw new PortablePosteriorError("posterior settings.target_accept must be in (0, 1)");
  }
  const maxTreedepth = positiveInteger(
    settings.max_treedepth, "posterior settings.max_treedepth",
  );
  if (maxTreedepth > 20) {
    throw new PortablePosteriorError("posterior max_treedepth must be at most 20");
  }
  const sampleStatsMode = header.sample_stats_mode;
  if (sampleStatsMode !== "per_draw_v1" && sampleStatsMode !== "per_draw_v2") {
    throw new PortablePosteriorError("posterior sample_stats_mode is invalid");
  }
  const drawsPerChain = positiveInteger(settings.num_draws, "posterior settings.num_draws");
  const chainOrder = integerArray(header.chain_order, "posterior chain_order");
  const chainCount = positiveInteger(header.chain_count, "posterior chain_count");
  if (positiveInteger(header.chains, "posterior chains") !== chainCount ||
      chainOrder.length === 0 || new Set(chainOrder).size !== chainOrder.length ||
      chainCount !== chainOrder.length || count !== chainCount * drawsPerChain ||
      integer(header.draw_count, "posterior draw_count") !== count) {
    throw new PortablePosteriorError("posterior chain topology or draw count is invalid");
  }
  if (
    integer(trailer.seed, "posterior trailer seed") !== headerSeed ||
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
  const declaredChainStats = trailer.chains.map((raw, index) => {
    const statistic = object(raw, `posterior trailer chains[${index}]`);
    if (integer(statistic.chain, "posterior trailer chain") !== chainOrder[index] ||
        integer(statistic.draw_count, "posterior trailer chain draw_count") !== drawsPerChain) {
      throw new PortablePosteriorError("posterior trailer chain statistics are out of order");
    }
    const stepSize = finiteNumber(statistic.step_size, "posterior trailer step_size");
    if (stepSize <= 0.0) {
      throw new PortablePosteriorError("posterior trailer step_size must be positive");
    }
    const meanAccept = finiteNumber(
      statistic.mean_accept, "posterior trailer mean_accept",
    );
    if (meanAccept < 0.0 || meanAccept > 1.0) {
      throw new PortablePosteriorError("posterior trailer mean_accept must be in [0, 1]");
    }
    const divergences = integer(statistic.divergences, "posterior trailer divergences");
    const histogram = integerArray(
      statistic.treedepth_histogram, "posterior trailer treedepth_histogram",
    );
    if (histogram.length !== maxTreedepth + 1) {
      throw new PortablePosteriorError("posterior treedepth histogram length is invalid");
    }
    return { divergences, histogram, meanAccept };
  });
  const seen = new Set();
  const actualDivergences = Array(chainCount).fill(0);
  const actualHistograms = Array.from(
    { length: chainCount }, () => Array(maxTreedepth + 1).fill(0),
  );
  const actualAcceptSums = Array(chainCount).fill(0.0);
  const draws = documents.slice(1, -1).map((raw, sourceDrawIndex) => {
    const draw = object(raw, `posterior draw ${sourceDrawIndex}`);
    if (draw.draws_format !== "v0-provisional") {
      throw new PortablePosteriorError(`posterior draw ${sourceDrawIndex} format is invalid`);
    }
    kindScope(draw, `posterior draw ${sourceDrawIndex}`);
    if (draw.draw_index === undefined || draw.draw_index !== sourceDrawIndex) {
      throw new PortablePosteriorError("posterior draw_index values are required and contiguous");
    }
    if (draw.draw_index_base !== "zero_based_retained_draw_order") {
      throw new PortablePosteriorError("posterior draw_index_base is invalid");
    }
    const chain = integer(draw.chain, "posterior chain");
    const drawIndex = integer(draw.draw, "posterior draw");
    if (integer(draw.seed, "posterior draw seed") !== headerSeed) {
      throw new PortablePosteriorError("posterior draw seed disagrees with header");
    }
    if (integer(draw.draw_count, "posterior draw_count") !== count) {
      throw new PortablePosteriorError("posterior draw_count disagrees with header");
    }
    if (integer(draw.chain_count, "posterior draw chain_count") !== chainCount) {
      throw new PortablePosteriorError("posterior draw chain_count disagrees with header");
    }
    if (JSON.stringify(integerArray(draw.chain_order, "posterior draw chain_order")) !==
        JSON.stringify(chainOrder)) {
      throw new PortablePosteriorError("posterior draw chain_order disagrees with header");
    }
    const treeDepth = integer(draw.tree_depth, "posterior tree_depth");
    if (treeDepth > maxTreedepth) {
      throw new PortablePosteriorError("posterior tree_depth exceeds declared max_treedepth");
    }
    if (draw.sample_stats_mode !== sampleStatsMode) {
      throw new PortablePosteriorError(
        "posterior draw sample_stats_mode disagrees with header",
      );
    }
    if (typeof draw.diverging !== "boolean") {
      throw new PortablePosteriorError("posterior diverging must be a boolean");
    }
    const treeAccept = finiteNumber(draw.tree_accept, "posterior tree_accept");
    if (treeAccept < 0.0 || treeAccept > 1.0) {
      throw new PortablePosteriorError("posterior tree_accept must be in [0, 1]");
    }
    if (sampleStatsMode === "per_draw_v2") {
      finiteNumber(draw.energy, "posterior energy");
    } else if (Object.hasOwn(draw, "energy")) {
      throw new PortablePosteriorError("posterior per_draw_v1 must not contain energy");
    }
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
    const chainPosition = Math.floor(sourceDrawIndex / drawsPerChain);
    actualDivergences[chainPosition] += Number(draw.diverging);
    actualHistograms[chainPosition][treeDepth] += 1;
    actualAcceptSums[chainPosition] += treeAccept;
    if (JSON.stringify(draw.parameter_order) !== JSON.stringify(names) ||
        integer(draw.parameter_count, "posterior draw parameter_count") !== names.length) {
      throw new PortablePosteriorError("posterior draw parameter order is invalid");
    }
    const values = object(draw.values, `posterior draw ${sourceDrawIndex} values`);
    if (JSON.stringify(Object.keys(values)) !== JSON.stringify(names)) {
      throw new PortablePosteriorError("posterior draw values do not match parameter order");
    }
    const flattened = parameters.map((parameter) => {
      const numbers = valuesForShape(
        values[parameter.name], parameter.shape, parameter.name,
      );
      return Object.freeze([parameter.name, Object.freeze(numbers)]);
    });
    return Object.freeze({
      sourceDrawIndex, chain, draw: drawIndex, values: Object.freeze(flattened),
    });
  });
  for (let index = 0; index < chainCount; index += 1) {
    const actualMeanAccept = actualAcceptSums[index] / drawsPerChain;
    if (actualDivergences[index] !== declaredChainStats[index].divergences ||
        JSON.stringify(actualHistograms[index]) !==
          JSON.stringify(declaredChainStats[index].histogram) ||
        Math.abs(actualMeanAccept - declaredChainStats[index].meanAccept) > 1e-9) {
      throw new PortablePosteriorError(
        "posterior trailer sampler statistics disagree with retained draws",
      );
    }
  }
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

function finiteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new PortablePosteriorError(`${label} must be a finite number`);
  }
  return value;
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

function valuesForShape(value, shape, label) {
  if (shape.length > 0) {
    if (!Array.isArray(value) || value.length !== shape[0]) {
      throw new PortablePosteriorError(
        `posterior value ${label} does not match declared shape`,
      );
    }
    return value.flatMap((item) => valuesForShape(item, shape.slice(1), label));
  }
  if (Array.isArray(value) || typeof value !== "number" || !Number.isFinite(value)) {
    throw new PortablePosteriorError(
      `posterior value ${label} must be a finite scalar at declared rank`,
    );
  }
  return [value];
}

function validateFinite(value) {
  if (Array.isArray(value)) {
    for (const item of value) validateFinite(item);
  } else if (typeof value === "object" && value !== null) {
    for (const item of Object.values(value)) validateFinite(item);
  } else if (typeof value === "number" && !Number.isFinite(value)) {
    throw new PortablePosteriorError("posterior source contains a non-finite number");
  }
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
