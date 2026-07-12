// Source: src/compile/index.ts, bayesledger @ 7346d71.

import { compileInPyodide } from "./runtime.mjs";

/**
 * @typedef {{ok: true, irBytes: Uint8Array, irHash: string} |
 *   {ok: false, exceptionType: string, message: string, traceback: string}} CompileResult
 */

/** @param {string} source @returns {Promise<CompileResult>} */
export async function compile(source) {
  return compileInPyodide(source);
}

export { compileInPyodide } from "./runtime.mjs";
