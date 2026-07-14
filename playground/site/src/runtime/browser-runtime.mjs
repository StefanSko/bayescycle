import { CompilerClient } from "../compile/index.mjs";
import {
  WorkerEngine,
  diagnose,
  generate,
  mergeChainFits,
  posteriorPredictive,
  priorPredictive,
  recoverCheck,
  sample,
  simulate,
} from "../engine/index.mjs";
import { normalizeDocument, serializeDocument } from "../data/documents.mjs";
import { GenerationPlanError, validateGenerationPlan } from "../generation/plan.mjs";

const UTF8 = new TextDecoder();
const ENCODE = new TextEncoder();

export class BrowserRuntime {
  constructor(executor = new WorkerEngine(), compiler = new CompilerClient()) {
    this.executor = executor;
    this.compiler = compiler;
  }

  /** @param {string} source @param {{timeoutMs?: number, signal?: AbortSignal}} [options] */
  compile(source, options) {
    return this.compiler.compile(source, options);
  }

  /** @param {Record<string, unknown> & {operation: string}} request @param {(event: Record<string, unknown>) => void} [onProgress] */
  async run(request, onProgress = () => {}) {
    let result;
    switch (request.operation) {
      case "sample":
        result = await this.#sample(request, onProgress);
        break;
      case "generate":
        result = await this.#generate(request);
        break;
      case "diagnose":
        result = oneArtifact(
          "diagnostics.json",
          "application/json",
          requireOutput(await diagnose({ fits: [asText(request.fit, "fit")], executor: this.executor })),
        );
        break;
      case "prior-predictive":
        result = oneArtifact(
          "prior_predictive.ndjson",
          "application/x-ndjson",
          requireOutput(await priorPredictive({
            model: asIrBytes(request.modelIr),
            data: asObject(request.data, "data"),
            settings: predictiveSettings(request.settings),
            seed: integerSetting(request.settings, "seed", 0),
            executor: this.executor,
          })),
        );
        break;
      case "posterior-predictive":
        result = oneArtifact(
          "posterior_predictive.ndjson",
          "application/x-ndjson",
          requireOutput(await posteriorPredictive({
            model: asIrBytes(request.modelIr),
            data: asObject(request.data, "data"),
            fit: asText(request.fit, "fit"),
            seed: integerSetting(request.settings, "seed", 0),
            executor: this.executor,
          })),
        );
        break;
      case "simulate": {
        const output = requireOutput(await simulate({
          model: asIrBytes(request.modelIr),
          data: asObject(request.data, "data"),
          truth: asObject(request.truth, "truth"),
          seed: integerSetting(request.settings, "seed", 0),
          executor: this.executor,
        }));
        result = {
          artifacts: [artifact(
            "simulated_data.json",
            "application/json",
            canonicalSimulatedBytes(output.rawBytes),
          )],
        };
        break;
      }
      case "recover-check":
        result = oneArtifact(
          "recovery_check.json",
          "application/json",
          requireOutput(await recoverCheck({
            fit: asText(request.fit, "fit"),
            truth: asObject(request.truth, "truth"),
            executor: this.executor,
          })),
        );
        break;
      default:
        throw new RuntimeError("UnsupportedOperation", `Unsupported operation ${request.operation}`);
    }
    return { type: "artifacts", id: request.id, artifacts: result.artifacts };
  }

  async #generate(request) {
    let plan;
    try {
      plan = validateGenerationPlan(request.plan);
    } catch (error) {
      if (error instanceof GenerationPlanError) {
        throw new RuntimeError("InvalidGenerationPlan", error.message);
      }
      throw error;
    }
    const parameters = plan.distribution.parameters;
    const outcomes = plan.distribution.outcomes;
    const modelBytes = outcomes.modelIrBytes;
    const designBytes = outcomes.designBytes;
    const identities = {
      generation_model_hash: await sha256Bytes(modelBytes),
      design_hash: await sha256Bytes(designBytes),
    };
    let parameterSource;
    if (parameters.kind === "fixed") {
      const parametersBytes = parameters.parametersBytes;
      identities.parameters_hash = await sha256Bytes(parametersBytes);
      parameterSource = {
        kind: "fixed",
        parameters: exactText(parametersBytes, "fixed parameters"),
      };
    } else if (parameters.kind === "model-prior") {
      parameterSource = {
        kind: "model-prior",
        authored_provenance: parameters.authoredProvenance === null ? null : {
          claimed_source_model_hash: parameters.authoredProvenance.claimedSourceModelHash,
          claimed_outcome_model_hash: parameters.authoredProvenance.claimedOutcomeModelHash,
        },
      };
    } else {
      const fit = parameters.fitArtifact;
      const fitModelBytes = fit.modelIrBytes;
      const fitDataBytes = fit.dataBytes;
      const posteriorBytes = fit.posteriorBytes;
      identities.fit_hash = await sha256Bytes(posteriorBytes);
      identities.fit_model_hash = await sha256Bytes(fitModelBytes);
      identities.fit_data_hash = await sha256Bytes(fitDataBytes);
      parameterSource = {
        kind: "posterior",
        fit: exactText(posteriorBytes, "fit posterior"),
        fit_data: exactText(fitDataBytes, "fit data"),
      };
    }
    const output = requireOutput(await generate({
      model: exactText(modelBytes, "generation model IR"),
      design: exactText(designBytes, "generation design"),
      parameterSource,
      identities,
      count: plan.count,
      seed: plan.seed,
      executor: this.executor,
    }));
    return oneArtifact("generated_datasets.ndjson", "application/x-ndjson", output);
  }

  async #sample(request, onProgress) {
    const settings = request.settings ?? {};
    const dataBytes = asDocumentBytes(request.data);
    const chains = integerSetting(settings, "chains", 4);
    const counts = Array.from({ length: chains }, () => ({ retainedDraws: 0, divergences: 0 }));
    const result = await sample({
      model: asIrBytes(request.modelIr),
      data: asObject(dataBytes, "data"),
      settings: engineSettings(settings),
      seed: integerSetting(settings, "seed", 0),
      chains,
      executor: this.executor,
      onDrawBatch: ({ chainId, draws }) => {
        const count = counts[chainId];
        if (count === undefined) return;
        count.retainedDraws += draws.length;
        count.divergences += draws.filter((draw) => draw.diverging === true).length;
        onProgress({ type: "progress", id: request.id, chainId, ...count });
      },
    });
    if (!result.ok) throw runtimeError(result.error);
    const streams = result.outputs.map((output) => UTF8.decode(output.rawBytes));
    const merged = streams.length === 1 ? streams[0] : mergeChainFits(streams);
    return {
      artifacts: [
        artifact("model.ir.json", "application/json", asIrBytes(request.modelIr)),
        artifact("data.json", "application/json", dataBytes),
        artifact("posterior.ndjson", "application/x-ndjson", ENCODE.encode(merged)),
      ],
    };
  }
}

export class RuntimeError extends Error {
  constructor(kind, message) {
    super(`${kind}: ${message}`);
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

function exactText(value, label) {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(value);
  } catch {
    throw new RuntimeError("InvalidRequest", `${label} must be valid UTF-8 bytes`);
  }
}

async function sha256Bytes(value) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", Uint8Array.from(value)));
  return `sha256:${[...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

function asText(value, label) {
  if (typeof value === "string") return value;
  if (value instanceof Uint8Array) return UTF8.decode(value);
  throw new RuntimeError("InvalidRequest", `${label} must be text or bytes`);
}

function asBytes(value, label) {
  if (value instanceof Uint8Array) return Uint8Array.from(value);
  if (typeof value === "string") return ENCODE.encode(value);
  throw new RuntimeError("InvalidRequest", `${label} must be text or bytes`);
}

function asDocumentBytes(value) {
  if (value !== null && typeof value === "object" && !(value instanceof Uint8Array)) {
    return ENCODE.encode(serializeDocument(normalizeDocument(value)));
  }
  return asBytes(value, "data");
}

function asIrBytes(value) {
  if (value !== null && typeof value === "object" && !(value instanceof Uint8Array)) {
    return ENCODE.encode(JSON.stringify(value));
  }
  return asBytes(value, "model IR");
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
