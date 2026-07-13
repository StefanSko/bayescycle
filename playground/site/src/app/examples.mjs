export function exampleAssetUrl(path, moduleUrl = import.meta.url) {
  return new URL(`../../examples/${path}`, moduleUrl);
}
