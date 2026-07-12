function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

export default [
  {
    name: "pyodide boots",
    fn: async () => {
      const { loadPyodide } = await import("/site/vendor/pyodide/pyodide.mjs");
      const pyodide = await loadPyodide({ indexURL: "/site/vendor/pyodide/" });
      assert(pyodide.runPython("1+1") === 2, "expected Python 1+1 to equal 2");
    },
  },
  {
    name: "engine wasm compiles",
    fn: async () => {
      const response = await fetch("/site/vendor/bayesite/bayesite_core.wasm");
      assert(response.ok, `engine wasm request failed with HTTP ${response.status}`);
      const module = await WebAssembly.compile(await response.arrayBuffer());
      assert(WebAssembly.Module.exports(module).length > 0, "engine wasm has no exports");
    },
  },
];
