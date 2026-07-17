import {
  parseDocument,
  projectDocument,
  serializeDocument,
} from "../data/documents.mjs";

const TEXT = new TextDecoder("utf-8", { fatal: true });
const ENCODE = new TextEncoder();

export const SCENARIO_TARGET_DIVERGENCE_ERROR =
  "The main model changed while compiling the prior scenario. Recompile the model before simulating from Another prior.";
export const EMPTY_RECOVERY_PROJECTION_NOTICE =
  "Posterior completed; recovery check unavailable: generated parameters contain no parameters from the original model.";

export function assertScenarioCompatible(originalCompile, scenarioCompile) {
  if (scenarioCompile.targetIrHash !== originalCompile.irHash) {
    throw new Error(SCENARIO_TARGET_DIVERGENCE_ERROR);
  }
  const originalSchema = originalCompile.modelSchema;
  const composedSchema = scenarioCompile.modelSchema;
  if (JSON.stringify(composedSchema.data) !== JSON.stringify(originalSchema.data)) {
    const originalNames = new Set(originalSchema.data.map((entry) => entry.name));
    const extraNames = composedSchema.data
      .map((entry) => entry.name)
      .filter((name) => !originalNames.has(name));
    if (extraNames.length > 0) {
      throw new Error(`prior-only model must not declare data slots: ${extraNames.join(", ")}`);
    }
    throw new Error("prior-only model must not change target data slots");
  }
  if (JSON.stringify(composedSchema.observed) !== JSON.stringify(originalSchema.observed)) {
    throw new Error("prior-only model must not change target observed slots");
  }
  return scenarioCompile;
}

export function projectRecoveryTruth(parametersBytes, modelSchema) {
  if (!(parametersBytes instanceof Uint8Array)) {
    throw new TypeError("paired parameters must be bytes");
  }
  const document = parseDocument(TEXT.decode(parametersBytes));
  const projected = projectDocument(
    document,
    modelSchema.parameters.map((parameter) => parameter.name),
  );
  if (Object.keys(projected.variables).length === 0) {
    return Object.freeze({ bytes: null, notice: EMPTY_RECOVERY_PROJECTION_NOTICE });
  }
  const owned = ENCODE.encode(serializeDocument(projected));
  return Object.freeze({
    get bytes() { return Uint8Array.from(owned); },
    notice: null,
  });
}
