"""CLI for exporting bayescycle run directories to ArviZ fit files."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from bayesite_idata import from_run_dir, write_netcdf
from bayesite_idata.assemble import AssemblyError
from bayesite_idata.protocol_v0 import ProtocolError
from bayesite_idata.run_dir import RunDirError
from bayesite_idata.validate import VALIDATE_MODES, ValidationError, validate_mode
from bayesite_idata.writer import WriteError


@click.command()
@click.argument("run_dir", type=click.Path(path_type=Path))
@click.option(
    "-o", "--output", required=True, type=click.Path(path_type=Path), help="Output .nc/.idata path."
)
@click.option(
    "--validate",
    "validate_value",
    type=click.Choice(VALIDATE_MODES),
    default="warn",
    show_default=True,
    help="Whether to run `bayesite diagnose` before assembly.",
)
@click.option(
    "--bayesite",
    "bayesite_path",
    type=click.Path(path_type=Path),
    help="Bayesite binary for validation; overrides BAYESITE and PATH lookup.",
)
def cli(run_dir: Path, output: Path, validate_value: str, bayesite_path: Path | None) -> None:
    """Export RUN_DIR to an ArviZ NetCDF fit file."""
    try:
        dt = from_run_dir(run_dir, validate=validate_mode(validate_value), bayesite=bayesite_path)
        path = write_netcdf(dt, output)
    except (RunDirError, ProtocolError, AssemblyError, ValidationError, WriteError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2) from exc
    sys.stdout.write(str(path) + "\n")
    sys.stdout.flush()
