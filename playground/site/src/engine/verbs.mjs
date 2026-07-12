// Source: src/engine/verbs.ts, bayesledger @ 7346d71.

import { EngineError } from "./types.mjs";

/**
 * @typedef {Record<string, unknown>} EngineDocument
 * @typedef {{execute: (request: Record<string, unknown>, options?: Record<string, unknown>) => Promise<import("./types.mjs").EngineOutput>}} EngineExecutor
 * @typedef {{model: EngineDocument, data: EngineDocument, settings: EngineDocument, seed: number, chains: number, executor: EngineExecutor, onDrawBatch?: (batch: import("./types.mjs").DrawBatch) => void}} SampleInputs
 * @typedef {{fits: string[], executor: EngineExecutor}} DiagnoseInputs
 * @typedef {{model: EngineDocument, data: EngineDocument, settings: EngineDocument, seed: number, executor: EngineExecutor}} PriorPredictiveInputs
 * @typedef {{model: EngineDocument, data: EngineDocument, fit: string, seed: number, executor: EngineExecutor}} PosteriorPredictiveInputs
 * @typedef {PosteriorPredictiveInputs} PosteriorCheckInputs
 * @typedef {{model: EngineDocument, data: EngineDocument, truth: EngineDocument, seed: number, executor: EngineExecutor}} SimulateInputs
 * @typedef {{model: EngineDocument, data: EngineDocument, settings: EngineDocument, seed: number, executor: EngineExecutor}} RecoverInputs
 * @typedef {{fit: string, truth: EngineDocument, targets?: EngineDocument, interval?: number, executor: EngineExecutor}} RecoverCheckInputs
 * @typedef {{model: EngineDocument, data: EngineDocument, settings: EngineDocument, seed: number, executor: EngineExecutor}} SbcInputs
 * @typedef {{ok: true, outputs: import("./types.mjs").EngineOutput[]} | {ok: false, error: import("./types.mjs").EngineErrorShape}} RunResult
 */

/** @param {SampleInputs} inputs @returns {Promise<RunResult>} */
export async function sample(inputs) {
  return guard(async () => {
    if (!Number.isInteger(inputs.chains) || inputs.chains < 1) {
      throw new EngineError("InvalidSettings", "sample chains must be a positive integer");
    }
    const outputs = await Promise.all(
      Array.from({ length: inputs.chains }, (_, chainId) =>
        inputs.executor.execute(
          {
            command: "sample",
            model: inputs.model,
            data: inputs.data,
            settings: inputs.settings,
            seed: inputs.seed,
            chain_id: chainId,
          },
          {
            chainId,
            ...(inputs.onDrawBatch === undefined
              ? {}
              : { onDrawBatch: inputs.onDrawBatch }),
          },
        ),
      ),
    );
    return { ok: true, outputs };
  });
}

/** @param {DiagnoseInputs} inputs @returns {Promise<RunResult>} */
export async function diagnose(inputs) {
  return guard(async () => {
    if (inputs.fits.length === 0) {
      throw new EngineError("InvalidSettings", "diagnose needs at least one fit stream");
    }
    const firstFit = inputs.fits[0];
    if (firstFit === undefined) {
      throw new EngineError("InvalidSettings", "diagnose needs at least one fit stream");
    }
    const output = await inputs.executor.execute({
      command: "diagnose",
      fit: inputs.fits.length === 1 ? firstFit : mergeChainFits(inputs.fits),
    });
    return { ok: true, outputs: [output] };
  });
}

/** @param {PriorPredictiveInputs} inputs @returns {Promise<RunResult>} */
export async function priorPredictive(inputs) {
  return guard(async () => {
    const output = await inputs.executor.execute({
      command: "prior-predictive",
      model: inputs.model,
      data: inputs.data,
      settings: inputs.settings,
      seed: inputs.seed,
    });
    return { ok: true, outputs: [output] };
  });
}

/** @param {PosteriorPredictiveInputs} inputs @returns {Promise<RunResult>} */
export async function posteriorPredictive(inputs) {
  return guard(async () => {
    const output = await inputs.executor.execute({
      command: "posterior-predictive",
      model: inputs.model,
      data: inputs.data,
      fit: inputs.fit,
      seed: inputs.seed,
    });
    return { ok: true, outputs: [output] };
  });
}

/** @param {PosteriorCheckInputs} inputs @returns {Promise<RunResult>} */
export async function posteriorCheck(inputs) {
  return guard(async () => {
    const output = await inputs.executor.execute({
      command: "posterior-check",
      model: inputs.model,
      data: inputs.data,
      fit: inputs.fit,
      seed: inputs.seed,
    });
    return { ok: true, outputs: [output] };
  });
}

/** @param {SimulateInputs} inputs @returns {Promise<RunResult>} */
export async function simulate(inputs) {
  return guard(async () => {
    const output = await inputs.executor.execute({
      command: "simulate",
      model: inputs.model,
      data: inputs.data,
      truth: inputs.truth,
      seed: inputs.seed,
    });
    return { ok: true, outputs: [output] };
  });
}

/** @param {RecoverCheckInputs} inputs @returns {Promise<RunResult>} */
export async function recoverCheck(inputs) {
  return guard(async () => {
    const request = {
      command: "recover-check",
      fit: inputs.fit,
      truth: inputs.truth,
      ...(inputs.targets === undefined ? {} : { targets: inputs.targets }),
      ...(inputs.interval === undefined ? {} : { settings: { interval: inputs.interval } }),
    };
    const output = await inputs.executor.execute(request);
    return { ok: true, outputs: [output] };
  });
}

/** @param {RecoverInputs} inputs @returns {Promise<RunResult>} */
export async function recover(inputs) {
  return guard(async () => {
    const output = await inputs.executor.execute({
      command: "recover",
      model: inputs.model,
      data: inputs.data,
      settings: inputs.settings,
      seed: inputs.seed,
    });
    return { ok: true, outputs: [output] };
  });
}

/** @param {SbcInputs} inputs @returns {Promise<RunResult>} */
export async function sbc(inputs) {
  return guard(async () => {
    const output = await inputs.executor.execute({
      command: "sbc",
      model: inputs.model,
      data: inputs.data,
      settings: inputs.settings,
      seed: inputs.seed,
    });
    return { ok: true, outputs: [output] };
  });
}

export function mergeChainFits(fits) {
  const parsed = fits.map((fit, fitIndex) => {
    const lines = fit.trimEnd().split("\n");
    if (lines.length < 3) {
      throw new EngineError(
        "MalformedEngineResponse",
        `fit stream ${String(fitIndex)} is incomplete`,
      );
    }
    const documents = lines.map((line, lineIndex) => {
      let value;
      try {
        value = JSON.parse(line);
      } catch {
        throw new EngineError(
          "MalformedEngineResponse",
          `fit stream ${String(fitIndex)} line ${String(lineIndex)} is not JSON`,
        );
      }
      if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new EngineError("MalformedEngineResponse", "fit NDJSON lines must be objects");
      }
      return value;
    });
    const header = documents[0];
    const trailerEnvelope = documents.at(-1);
    const trailer = trailerEnvelope?.trailer;
    if (
      header === undefined ||
      typeof trailer !== "object" ||
      trailer === null ||
      Array.isArray(trailer)
    ) {
      throw new EngineError("MalformedEngineResponse", "fit stream has no trailer");
    }
    return { header, draws: documents.slice(1, -1), trailer };
  });
  const chainOrder = parsed.flatMap(({ header }) => arrayValue(header.chain_order));
  const totalDraws = parsed.reduce((total, fit) => total + fit.draws.length, 0);
  const common = {
    chain_count: parsed.length,
    chain_order: chainOrder,
    draw_count: totalDraws,
  };
  const first = parsed[0];
  if (first === undefined) throw new EngineError("InvalidSettings", "diagnose needs a fit");
  const header = { ...first.header, ...common, chains: parsed.length };
  let drawIndex = 0;
  const draws = parsed.flatMap((fit) =>
    fit.draws.map((draw) => ({ ...draw, ...common, draw_index: drawIndex++ })),
  );
  const trailer = {
    ...first.trailer,
    ...common,
    chains: parsed.flatMap((fit) => arrayValue(fit.trailer.chains)),
  };
  return [header, ...draws, { trailer }].map((value) => JSON.stringify(value)).join("\n") + "\n";
}

function arrayValue(value) {
  return Array.isArray(value) ? value : [];
}

async function guard(operation) {
  try {
    return await operation();
  } catch (error) {
    if (!(error instanceof EngineError)) throw error;
    return {
      ok: false,
      error: {
        error_format: error.error_format,
        error: error.error,
        message: error.message,
      },
    };
  }
}
