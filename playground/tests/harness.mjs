const params = new URLSearchParams(window.location.search);
const suiteName = params.get("suite");

window.__results = [];
window.__done = false;

try {
  if (!suiteName || !/^[a-z0-9_-]+$/i.test(suiteName)) {
    throw new Error("invalid or missing suite name");
  }

  const suite = await import(`./unit/${suiteName}.test.mjs`);
  if (!Array.isArray(suite.default)) {
    throw new Error(`suite ${suiteName} must default-export an array`);
  }

  for (const testCase of suite.default) {
    try {
      await testCase.fn();
      window.__results.push({ name: testCase.name, ok: true, error: null });
    } catch (error) {
      window.__results.push({ name: testCase.name, ok: false, error: String(error) });
    }
  }
} catch (error) {
  window.__results.push({ name: suiteName || "harness", ok: false, error: String(error) });
} finally {
  window.__done = true;
}
