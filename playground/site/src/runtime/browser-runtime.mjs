import {
  WorkerEngine,
  diagnose,
  mergeChainFits,
  posteriorPredictive,
  priorPredictive,
  recoverCheck,
  sample,
  simulate,
} from "../engine/index.mjs";
import { normalizeDocument, serializeDocument } from "../data/documents.mjs";

const UTF8 = new TextDecoder();
const ENCODE = new TextEncoder();

export class BrowserRuntime {
  constructor(executor = new WorkerEngine()) {
    this.executor = executor;
  }

  /** @param {Record<string, unknown> & {operation: string}} request @param {(event: Record<string, unknown>) => void} [onProgress] */
  async run(request, onProgress = () => {}) {
    switch (request.operation) {
      case "sample":
        return this.#sample(request, onProgress);
      case "diagnose":
        return oneArtifact(
          "diagnostics.json",
          "application/json",
          requireOutput(await diagnose({ fits: [asText(request.fit, "fit")], executor: this.executor })),
        );
      case "prior-predictive":
        return oneArtifact(
          "prior_predictive.ndjson",
          "application/x-ndjson",
          requireOutput(await priorPredictive({
            model: asObject(request.modelIr, "model IR"),
            data: asObject(request.data, "data"),
            settings: predictiveSettings(request.settings),
            seed: integerSetting(request.settings, "seed", 0),
            executor: this.executor,
          })),
        );
      case "posterior-predictive":
        return oneArtifact(
          "posterior_predictive.ndjson",
          "application/x-ndjson",
          requireOutput(await posteriorPredictive({
            model: asObject(request.modelIr, "model IR"),
            data: asObject(request.data, "data"),
            fit: asText(request.fit, "fit"),
            seed: integerSetting(request.settings, "seed", 0),
            executor: this.executor,
          })),
        );
      case "simulate": {
        const output = requireOutput(await simulate({
          model: asObject(request.modelIr, "model IR"),
          data: asObject(request.data, "data"),
          truth: asObject(request.truth, "truth"),
          seed: integerSetting(request.settings, "seed", 0),
          executor: this.executor,
        }));
        return {
          artifacts: [artifact(
            "simulated_data.json",
            "application/json",
            canonicalSimulatedBytes(output.rawBytes),
          )],
        };
      }
      case "recover-check":
        return oneArtifact(
          "recovery_check.json",
          "application/json",
          requireOutput(await recoverCheck({
            fit: asText(request.fit, "fit"),
            truth: asObject(request.truth, "truth"),
            executor: this.executor,
          })),
        );
      default:
        throw new RuntimeError("UnsupportedOperation", `Unsupported operation ${request.operation}`);
    }
  }

  async #sample(request, onProgress) {
    const settings = request.settings ?? {};
    const chains = integerSetting(settings, "chains", 4);
    const counts = Array.from({ length: chains }, () => ({ retainedDraws: 0, divergences: 0 }));
    const result = await sample({
      model: asObject(request.modelIr, "model IR"),
      data: asObject(request.data, "data"),
      settings: engineSettings(settings),
      seed: integerSetting(settings, "seed", 0),
      chains,
      executor: this.executor,
      onDrawBatch: ({ chainId, draws }) => {
        const count = counts[chainId];
        if (count === undefined) return;
        count.retainedDraws += draws.length;
        count.divergences += draws.filter((draw) => draw.diverging === true).length;
        onProgress({ type: "progress", chainId, ...count });
      },
    });
    if (!result.ok) throw runtimeError(result.error);
    const streams = result.outputs.map((output) => UTF8.decode(output.rawBytes));
    const merged = streams.length === 1 ? streams[0] : mergeChainFits(streams);
    return { artifacts: [artifact("posterior.ndjson", "application/x-ndjson", ENCODE.encode(merged))] };
  }
}

export class RuntimeError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = "RuntimeError";
    this.kind = kind;
  }
}

function canonicalSimulatedBytes(bytes) {
  const parsed = JSON.parse(UTF8.decode(bytes));
  let candidate = parsed;
  if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) &&
      !Object.hasOwn(parsed, "format") && !Object.hasOwn(parsed, "variables")) {
    candidate = { format: "bayescycle.data.json.v1", variables: parsed };
  }
  return ENCODE.encode(serializeDocument(normalizeDocument(candidate)));
}

function oneArtifact(name, mediaType, output) {
  return { artifacts: [artifact(name, mediaType, output.rawBytes)] };
}

function artifact(name, mediaType, bytes) {
  return Object.freeze({ name, mediaType, bytes: Uint8Array.from(bytes) });
}

function requireOutput(result) {
  if (!result.ok) throw runtimeError(result.error);
  const output = result.outputs[0];
  if (output === undefined) throw new RuntimeError("MissingArtifact", "Runtime produced no artifact");
  return output;
}

function runtimeError(error) {
  return new RuntimeError(error.error, error.message);
}

function asText(value, label) {
  if (typeof value === "string") return value;
  if (value instanceof Uint8Array) return UTF8.decode(value);
  throw new RuntimeError("InvalidRequest", `${label} must be text or bytes`);
}

function asObject(value, label) {
  if (value !== null && typeof value === "object" && !(value instanceof Uint8Array)) return value;
  const parsed = JSON.parse(asText(value, label));
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new RuntimeError("InvalidRequest", `${label} must contain a JSON object`);
  }
  return parsed;
}

function integerSetting(settings, name, fallback) {
  const value = settings?.[name] ?? fallback;
  if (!Number.isInteger(value)) throw new RuntimeError("InvalidSettings", `${name} must be an integer`);
  return value;
}

function predictiveSettings(settings = {}) {
  return { num_draws: integerSetting(settings, "num_draws", 200) };
}

function engineSettings(settings = {}) {
  return {
    num_warmup: integerSetting(settings, "num_warmup", 1000),
    num_draws: integerSetting(settings, "num_draws", 2000),
    max_treedepth: integerSetting(settings, "max_treedepth", 10),
    target_accept: typeof settings.target_accept === "number" ? settings.target_accept : 0.8,
  };
}
