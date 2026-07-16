const KINDS = ["compile", "generation", "conditioning"];

// Physical worker ownership follows reducer state without changing reducer
// semantics. When an accepted attempt is no longer current, its signal aborts.
export class RunControllers {
  #active = new Map();

  begin(kind, requestId) {
    if (!KINDS.includes(kind)) throw new Error(`Unknown run controller kind ${kind}`);
    this.#active.get(kind)?.controller.abort();
    const controller = new AbortController();
    this.#active.set(kind, { requestId, controller });
    return controller;
  }

  finish(kind, requestId) {
    if (this.#active.get(kind)?.requestId === requestId) this.#active.delete(kind);
  }

  reconcile(state) {
    for (const [kind, active] of this.#active) {
      if (matches(kind, active.requestId, state)) continue;
      active.controller.abort();
      this.#active.delete(kind);
    }
  }
}

function matches(kind, requestId, state) {
  if (kind === "compile") {
    return state.compile.status === "compiling" && state.compile.requestId === requestId;
  }
  if (kind === "generation") {
    return state.generation.attempt.status === "running" &&
      state.generation.attempt.requestId === requestId;
  }
  return state.run.status === "running" && state.run.requestId === requestId &&
    state.conditioning.attempt.status === "running" &&
    state.conditioning.attempt.requestId === requestId;
}
