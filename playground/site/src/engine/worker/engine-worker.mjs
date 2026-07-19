// Source: src/engine/worker/engine-worker.ts, bayesledger @ 7346d71.

import { BayesiteAbi } from "../abi.mjs";
import {
  assertWasmHash,
  isStreamCommand,
  parseEngineMetadata,
} from "../executor.mjs";
import { artifactBytes, parseEngineOutput } from "../stream.mjs";
import {
  EngineError,
  requirePosteriorResponseWithinLimit,
} from "../types.mjs";

self.onmessage = (event) => {
  void handle(event.data);
};

async function handle(message) {
  if (message.request.command === "sample") {
    post({ type: "started", id: message.id, chainId: message.chainId });
  }
  try {
    const [wasmResponse, metadataResponse] = await Promise.all([
      fetch(message.wasmUrl),
      fetch(message.metadataUrl),
    ]);
    if (!wasmResponse.ok || !metadataResponse.ok) {
      throw new EngineError(
        "EngineLoadFailure",
        "Could not load vendored Bayesite engine assets",
      );
    }
    const [wasmBuffer, metadataValue] = await Promise.all([
      wasmResponse.arrayBuffer(),
      metadataResponse.json(),
    ]);
    const wasmBytes = new Uint8Array(wasmBuffer);
    const metadata = parseEngineMetadata(metadataValue);
    await assertWasmHash(wasmBytes, metadata.wasm_sha256);
    const abi = await BayesiteAbi.instantiate(wasmBytes, metadata);
    const rawBytes = artifactBytes(abi.run(message.request));
    if (message.request.command === "sample") {
      requirePosteriorResponseWithinLimit(rawBytes.byteLength);
    }
    const output = parseEngineOutput(
      rawBytes,
      message.chainId,
      isStreamCommand(message.request.command),
      message.request.command === "sample"
        ? (batch) => post({ type: "batch", id: message.id, ...batch })
        : undefined,
    );
    post(
      {
        type: "result",
        id: message.id,
        chainId: message.chainId,
        ...output,
      },
      [output.rawBytes.buffer],
    );
  } catch (error) {
    const typed =
      error instanceof EngineError
        ? error
        : new EngineError(
            "WorkerFailure",
            error instanceof Error ? error.message : String(error),
          );
    post({
      type: "error",
      id: message.id,
      chainId: message.chainId,
      error: {
        error_format: typed.error_format,
        error: typed.error,
        message: typed.message,
      },
    });
  }
}

function post(message, transfer = []) {
  self.postMessage(message, { transfer });
}
