import { CompilerClient } from "../compile/index.mjs";
import {
  WorkerEngine,
  diagnose,
  generate,
  mergeChainFits,
  recoverCheck,
  sample,
} from "../engine/index.mjs";
import {
  normalizeDocument,
  parseDocument,
  serializeDocument,
} from "../data/documents.mjs";
import {
  parseGeneratedDatasets,
  verifyGeneratedDatasets,
} from "../generation/artifact.mjs";
import {
  GenerationPlanError,
  fitArtifact,
  serializeGenerationPlan,
  validateGenerationPlan,
} from "../generation/plan.mjs";
import { validatePortablePosterior } from "../generation/posterior-source.mjs";

const UTF8 = new TextDecoder();
const ENCODE = new TextEncoder();

export class BrowserRuntime {
  #runtimeFits = new WeakSet();

  constructor(executor = new WorkerEngine(), compiler = new CompilerClient()) {
    this.executor = executor;
    this.compiler = compiler;
  }

  /** @param {string} source @param {{timeoutMs?: number, signal?: AbortSignal}} [options] */
  compile(source, options) {
    return this.compiler.compile(source, options);
  }

  /**
   * @param {Record<string, unknown> & {operation: string}} request
   * @param {((event: Record<string, unknown>) => void) | {signal?: AbortSignal}} [onProgress]
   * @param {{signal?: AbortSignal}} [options]
   */
  async run(request, onProgress = () => {}, options = {}) {
    if (typeof onProgress !== "function") {
      options = onProgress;
      onProgress = () => {};
    }
    const signal = options.signal;
    requireNotCancelled(signal);
    let result;
    switch (request.operation) {
      case "condition":
        result = await this.#condition(request, onProgress, signal);
        break;
      case "sample":
        result = await this.#sample(request, onProgress, signal);
        break;
      case "generate":
        result = await this.#generate(request, signal);
        break;
      case "diagnose":
        result = oneArtifact(
          "diagnostics.json",
          "application/json",
          requireOutput(await diagnose({
            fits: [asText(request.fit, "fit")], executor: this.executor, signal,
          })),
        );
        break;
      case "recover-check":
        result = oneArtifact(
          "recovery_check.json",
          "application/json",
          requireOutput(await recoverCheck({
            fit: asText(request.fit, "fit"),
            truth: asDocumentObject(request.truth, "truth"),
            executor: this.executor,
            signal,
          })),
        );
        break;
      default:
        throw new RuntimeError("UnsupportedOperation", `Unsupported operation ${request.operation}`);
    }
    return { type: "artifacts", id: request.id, ...result };
  }

  async #generate(request, signal) {
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
    if (parameters.kind === "posterior" && parameters.fitArtifact.association === "runtime" &&
        !this.#runtimeFits.has(parameters.fitArtifact)) {
      throw new RuntimeError(
        "InvalidFitAssociation",
        "runtime posterior association was not issued by this conditioning runtime",
      );
    }
    if (parameters.kind === "posterior" && parameters.fitArtifact.association === "portable") {
      await validatePortablePosterior({
        modelBytes: parameters.fitArtifact.modelIrBytes,
        dataBytes: parameters.fitArtifact.dataBytes,
        posteriorBytes: parameters.fitArtifact.posteriorBytes,
      });
    }
    requireNotCancelled(signal);
    const output = requireOutput(await generate({
      model: exactText(modelBytes, "generation model IR"),
      design: exactText(designBytes, "generation design"),
      parameterSource,
      identities,
      count: plan.count,
      seed: plan.seed,
      executor: this.executor,
      signal,
    }));
    requireNotCancelled(signal);
    const parsedOutput = parseGeneratedDatasets(output.rawBytes);
    await verifyGeneratedDatasets(parsedOutput, {
      modelBytes,
      designBytes,
      fixedParametersBytes: parameters.kind === "fixed"
        ? parameters.parametersBytes
        : undefined,
      modelPriorBytes: parameters.kind === "model-prior"
        ? parameters.modelIrBytes
        : undefined,
      authoredProvenance: parameters.kind === "model-prior" &&
        parameters.authoredProvenance !== null
        ? {
            claimed_source_model_hash:
              parameters.authoredProvenance.claimedSourceModelHash,
            claimed_outcome_model_hash:
              parameters.authoredProvenance.claimedOutcomeModelHash,
          }
        : null,
      posteriorBytes: parameters.kind === "posterior"
        ? parameters.fitArtifact.posteriorBytes
        : undefined,
      fitDataBytes: parameters.kind === "posterior"
        ? parameters.fitArtifact.dataBytes
        : undefined,
      posteriorAssociation: parameters.kind === "posterior"
        ? parameters.fitArtifact.association
        : undefined,
      expectedSourceKind: parameters.kind,
      expectedCount: plan.count,
      expectedSeed: plan.seed,
    });
    requireNotCancelled(signal);
    const generated = artifact(
      "generated_datasets.ndjson",
      "application/x-ndjson",
      output.rawBytes,
    );
    if (parameters.kind === "posterior" && parameters.fitArtifact.association === "runtime") {
      return { artifacts: [generated] };
    }
    const planBytes = await serializeGenerationPlan(plan);
    const published = [
      artifact("model.ir.json", "application/json", modelBytes),
      artifact("design.json", "application/json", designBytes),
      artifact("generation-plan.json", "application/json", planBytes),
    ];
    if (parameters.kind === "fixed") {
      published.push(artifact(
        "fixed-parameters.json",
        "application/json",
        parameters.parametersBytes,
      ));
    } else if (parameters.kind === "posterior") {
      published.push(
        artifact(
          "source-posterior.ndjson",
          "application/x-ndjson",
          parameters.fitArtifact.posteriorBytes,
        ),
        artifact(
          "source-fit-data.json",
          "application/json",
          parameters.fitArtifact.dataBytes,
        ),
      );
    }
    published.push(generated);
    const runBytes = await generationRunBytes(published);
    published.push(artifact("run.json", "application/json", runBytes));
    return { artifacts: published };
  }

  async #condition(request, onProgress, signal) {
    const sampled = await this.#sample(request, onProgress, signal);
    const posterior = sampled.artifacts.find(
      (entry) => entry.name === "posterior.ndjson",
    );
    if (posterior === undefined) {
      throw new RuntimeError("MissingArtifact", "Conditioning produced no posterior artifact");
    }
    const artifacts = [...sampled.artifacts];
    const warnings = [];
    try {
      const diagnosed = requireOutput(await diagnose({
        fits: [exactText(posterior.bytes, "posterior")],
        executor: this.executor,
        signal,
      }));
      artifacts.push(artifact(
        "diagnostics.json", "application/json", diagnosed.rawBytes,
      ));
    } catch (error) {
      if (signal?.aborted === true) throw error;
      warnings.push(`Posterior completed; diagnostics unavailable: ${error.message}`);
    }
    if (request.pairedParameters !== undefined) {
      try {
        const recovery = requireOutput(await recoverCheck({
          fit: exactText(posterior.bytes, "posterior"),
          truth: asDocumentObject(request.pairedParameters, "paired parameters"),
          executor: this.executor,
          signal,
        }));
        artifacts.push(artifact(
          "recovery_check.json", "application/json", recovery.rawBytes,
        ));
      } catch (error) {
        if (signal?.aborted === true) throw error;
        warnings.push(`Posterior completed; recovery check unavailable: ${error.message}`);
      }
    }
    return {
      artifacts,
      fitArtifact: sampled.fitArtifact,
      notice: warnings.length === 0 ? null : warnings.join("\n"),
    };
  }

  async #sample(request, onProgress, signal) {
    const settings = request.settings ?? {};
    const modelBytes = asIrBytes(request.modelIr);
    const dataBytes = asDocumentBytes(request.data);
    const chains = integerSetting(settings, "chains", 4);
    const counts = Array.from({ length: chains }, () => ({ retainedDraws: 0, divergences: 0 }));
    const result = await sample({
      model: modelBytes,
      data: asObject(dataBytes, "data"),
      settings: engineSettings(settings),
      seed: integerSetting(settings, "seed", 0),
      chains,
      executor: this.executor,
      signal,
      onDrawBatch: ({ chainId, draws }) => {
        const count = counts[chainId];
        if (count === undefined) return;
        count.retainedDraws += draws.length;
        count.divergences += draws.filter((draw) => draw.diverging === true).length;
        onProgress({ type: "progress", id: request.id, chainId, ...count });
      },
    });
    if (!result.ok) throw runtimeError(result.error);
    requireNotCancelled(signal);
    const streams = result.outputs.map((output) => UTF8.decode(output.rawBytes));
    const merged = streams.length === 1 ? streams[0] : mergeChainFits(streams);
    const posteriorBytes = ENCODE.encode(merged);
    const association = fitArtifact(modelBytes, dataBytes, posteriorBytes, "runtime");
    this.#runtimeFits.add(association);
    return {
      artifacts: [
        artifact("model.ir.json", "application/json", modelBytes),
        artifact("data.json", "application/json", dataBytes),
        artifact("posterior.ndjson", "application/x-ndjson", posteriorBytes),
      ],
      fitArtifact: association,
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

function oneArtifact(name, mediaType, output) {
  return { artifacts: [artifact(name, mediaType, output.rawBytes)] };
}

function artifact(name, mediaType, bytes) {
  const owned = Uint8Array.from(bytes);
  return Object.freeze({
    name,
    mediaType,
    get bytes() { return Uint8Array.from(owned); },
  });
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

async function generationRunBytes(artifacts) {
  const byName = new Map(artifacts.map((entry) => [entry.name, entry]));
  const entry = async (role, name, format) => {
    const value = byName.get(name);
    if (value === undefined) {
      throw new RuntimeError("MissingArtifact", `Generation publication needs ${name}`);
    }
    return { role, path: name, sha256: await sha256Bytes(value.bytes), format };
  };
  const inputs = [await entry("design", "design.json", "bayescycle.data.json.v1")];
  if (byName.has("fixed-parameters.json")) {
    inputs.push(await entry(
      "fixed-parameters",
      "fixed-parameters.json",
      "bayescycle.data.json.v1",
    ));
  } else if (byName.has("source-posterior.ndjson")) {
    inputs.push(
      await entry("source-posterior", "source-posterior.ndjson", "v0-provisional"),
      await entry("source-fit-data", "source-fit-data.json", "bayescycle.data.json.v1"),
    );
  }
  const plan = byName.get("generation-plan.json");
  const model = byName.get("model.ir.json");
  if (plan === undefined || model === undefined) {
    throw new RuntimeError("MissingArtifact", "Generation publication needs model and plan");
  }
  const document = {
    format: "bayescycle.generation-run.v0",
    kind: "generate",
    backend: "bayesite",
    plan: {
      path: "generation-plan.json",
      sha256: await sha256Bytes(plan.bytes),
      format: "v0-provisional",
    },
    model: {
      path: "model.ir.json",
      sha256: await sha256Bytes(model.bytes),
      format: "bayeswire_ir.v1",
    },
    inputs,
    outputs: [await entry(
      "generated-datasets",
      "generated_datasets.ndjson",
      "v0-provisional",
    )],
  };
  return ENCODE.encode(`${JSON.stringify(document)}\n`);
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
  const bytes = asBytes(value, "data");
  parseDocument(exactText(bytes, "data"));
  return bytes;
}

function asDocumentObject(value, label) {
  try {
    if (value !== null && typeof value === "object" && !(value instanceof Uint8Array)) {
      return normalizeDocument(value);
    }
    return parseDocument(exactText(asBytes(value, label), label));
  } catch (error) {
    throw new RuntimeError("InvalidRequest", `${label}: ${error.message}`);
  }
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

function requireNotCancelled(signal) {
  if (signal?.aborted === true) {
    throw new RuntimeError("Cancelled", "Runtime operation was cancelled");
  }
}

function integerSetting(settings, name, fallback) {
  const value = settings?.[name] ?? fallback;
  if (!Number.isInteger(value)) throw new RuntimeError("InvalidSettings", `${name} must be an integer`);
  return value;
}

function engineSettings(settings = {}) {
  return {
    num_warmup: integerSetting(settings, "num_warmup", 1000),
    num_draws: integerSetting(settings, "num_draws", 2000),
    max_treedepth: integerSetting(settings, "max_treedepth", 10),
    target_accept: typeof settings.target_accept === "number" ? settings.target_accept : 0.8,
  };
}
