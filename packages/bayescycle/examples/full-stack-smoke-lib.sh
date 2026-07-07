set -euo pipefail

WORK="examples/_showboat_work"
RUN="$WORK/run"
if [ -z "${BAYESITE_BIN:-}" ]; then
  BAYESITE_BIN="$(command -v bayesite || true)"
fi
# python_with_viz below reaches directly into bayesite-viz's own pinned
# dependency set (xarray with DataTree support, ArviZ) purely to inspect a
# fit.nc file with plain Python; it does not shell out to the bayesite-viz
# CLIs. Those CLI invocations (bayesite-idata, bayesite-viz) are now owned by
# `bayescycle idata`/`bayescycle plot`, which pin bayesite-viz once in
# BAYESITE_VIZ_SOURCE under src/bayescycle/backends/bayesite_viz/uvx_runner.py.
# Derive the default from that single pin so this script never carries its
# own, second, divergent commit.
BAYESITE_VIZ_SOURCE="${BAYESITE_VIZ_SOURCE:-$(uv --quiet run python -c 'from bayescycle.backends.bayesite_viz.uvx_runner import BAYESITE_VIZ_SOURCE as s; print(s)')}"

reset_workdir() {
  rm -rf "$WORK"
  mkdir -p "$WORK"
}

ensure_bayesite() {
  if [ -n "$BAYESITE_BIN" ] && [ -x "$BAYESITE_BIN" ]; then
    return
  fi
  printf 'bayesite executable not found; install the Bayesite CLI release or set BAYESITE_BIN\n' >&2
  return 1
}

bayescycle_idata() {
  uv --quiet run bayescycle idata "$@"
}

bayescycle_plot() {
  PYTHONWARNINGS=ignore MPLBACKEND=Agg uv --quiet run bayescycle plot "$@"
}

python_with_viz() {
  uv --quiet run --with "$BAYESITE_VIZ_SOURCE" python "$@"
}

relpath() {
  python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], start=os.getcwd()))' "$1"
}

png_summary() {
  python3 -c 'import os, sys; from pathlib import Path; verb=sys.argv[1]; path=Path(sys.argv[2]); rel=os.path.relpath(path, start=os.getcwd()); status="ok" if path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") else "bad"; print(f"{verb}: {rel} png={status}")' "$1" "$2"
}
