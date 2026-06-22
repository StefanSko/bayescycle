set -euo pipefail

WORK="examples/_showboat_work"
RUN="$WORK/run"
BAYESITE_BIN="${BAYESITE_BIN:-../bayesite/target/debug/bayesite}"
BAYESITE_VIZ_SOURCE="${BAYESITE_VIZ_SOURCE:-git+https://github.com/StefanSko/bayesite-viz.git@a280945}"

reset_workdir() {
  rm -rf "$WORK"
  mkdir -p "$WORK"
}

ensure_bayesite() {
  if [ ! -x "$BAYESITE_BIN" ]; then
    cargo build --quiet --manifest-path ../bayesite/Cargo.toml --bin bayesite
  fi
}

bayesite_idata() {
  uvx --quiet --from "$BAYESITE_VIZ_SOURCE" bayesite-idata "$@"
}

bayesite_viz() {
  PYTHONWARNINGS=ignore MPLBACKEND=Agg uvx --quiet --from "$BAYESITE_VIZ_SOURCE" bayesite-viz "$@"
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
