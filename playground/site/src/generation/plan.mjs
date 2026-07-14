export const MAX_GENERATION_COUNT = 1000;
export const MAX_GENERATION_INPUT_BYTES = 8 * 1024 * 1024;

export class GenerationPlanError extends Error {
  constructor(message) {
    super(message);
    this.name = "GenerationPlanError";
  }
}

function pending() {
  throw new GenerationPlanError("functional generation plans are not implemented");
}

export function fixed(_parametersBytes) { return pending(); }
export function modelPrior(_modelIrBytes, _authoredProvenance = null) { return pending(); }
export function fitArtifact(_modelIrBytes, _dataBytes, _posteriorBytes, _association) { return pending(); }
export function posteriorOf(_fitArtifact) { return pending(); }
export function outcomesOf(_modelIrBytes, _designBytes) { return pending(); }
export function jointPredict(_parameters, _outcomes) { return pending(); }
export function draw(_distribution, _count, _seed) { return pending(); }
export function generateDatasets(_modelIrBytes, _options) { return pending(); }
export function validateGenerationPlan(_value) { return pending(); }
export async function serializeGenerationPlan(_plan) { return pending(); }
export async function parseGenerationPlanDocument(_bytes) { return pending(); }
export async function generationPlanIdentity(_plan) { return pending(); }
export async function generationInvalidationKey(_plan) { return pending(); }
