export const MAX_GENERATED_ARTIFACT_BYTES = 64 * 1024 * 1024;
export const MAX_GENERATED_LINE_BYTES = 8 * 1024 * 1024;

export class GeneratedDatasetsArtifactError extends Error {
  constructor(message) {
    super(message);
    this.name = "GeneratedDatasetsArtifactError";
  }
}

/** @param {Uint8Array} _bytes */
export function parseGeneratedDatasets(_bytes) {
  throw new GeneratedDatasetsArtifactError("generated-dataset parsing is not implemented");
}

/** @param {ReturnType<typeof parseGeneratedDatasets>} _artifact @param {Record<string, Uint8Array>} _sources */
export async function verifyGeneratedDatasets(_artifact, _sources) {
  throw new GeneratedDatasetsArtifactError("generated-dataset verification is not implemented");
}
