"""Output contract: write image, print exactly one line to stdout.

Chartroom-style: with ``-o`` overwrite; without it, auto-generate a per-verb
default filename and increment to avoid overwriting. The only stdout output is
the absolute path (or the selected ``-f`` format derived from it), followed by a
newline. Everything else goes to stderr.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type OutputFormat = Literal["path", "markdown", "html", "json", "alt"]
OUTPUT_FORMATS: tuple[OutputFormat, ...] = ("path", "markdown", "html", "json", "alt")
_OUTPUT_FORMATS_BY_VALUE: dict[str, OutputFormat] = {
    "path": "path",
    "markdown": "markdown",
    "html": "html",
    "json": "json",
    "alt": "alt",
}


@dataclass(frozen=True)
class OutputSpec:
    """Resolved output target and how to announce it."""

    path: Path
    fmt: OutputFormat
    alt: str | None


def output_format(value: str) -> OutputFormat:
    """Normalize a Click-validated output format into the internal semantic type."""
    try:
        return _OUTPUT_FORMATS_BY_VALUE[value]
    except KeyError as exc:
        msg = f"unknown output format: {value}"
        raise ValueError(msg) from exc


def resolve_output_path(verb: str, out: str | None, ext: str) -> Path:
    """Resolve the file to write to, never silently overwriting.

    With ``out`` given, use it (overwrite if it exists). Without it, use the
    per-verb default ``<verb>.<ext>`` in the current directory, incrementing to
    ``<verb>-2.<ext>`` etc. so an existing file is never clobbered.
    """
    if out is not None:
        return Path(out)
    candidate = Path(f"{verb}.{ext}")
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        cand = Path(f"{verb}-{i}.{ext}")
        if not cand.exists():
            return cand
        i += 1


def auto_alt(
    verb: str,
    fit: Path,
    var_names: tuple[str, ...] | None = None,
    kind: str | None = None,
) -> str:
    """Generate deterministic alt text when --alt is omitted.

    Describes the verb, the selected variables (or "all variables"), the ppc
    kind if any, and the source fit filename. Good enough for accessibility
    and for an agent embedding the image in a report.
    """
    vars_desc = ", ".join(var_names) if var_names else "all variables"
    if verb == "ppc" and kind:
        head = f"{verb} ({kind}) plot"
    else:
        head = f"{verb} plot"
    return f"{head} of {vars_desc} from {fit.name}"


def announce(spec: OutputSpec) -> None:
    """Print the one line of stdout the contract allows."""
    import html
    from urllib.parse import quote

    abs_path = str(spec.path.resolve())
    if spec.fmt == "path":
        sys.stdout.write(abs_path + "\n")
    elif spec.fmt == "markdown":
        alt = (spec.alt or "").replace("\\", "\\\\").replace("]", "\\]")
        url = quote(abs_path, safe="/")
        sys.stdout.write(f"![{alt}]({url})\n")
    elif spec.fmt == "html":
        alt = html.escape(spec.alt or "")
        url = html.escape(quote(abs_path, safe="/"), quote=True)
        sys.stdout.write(f'<img src="{url}" alt="{alt}">\n')
    elif spec.fmt == "json":
        import json

        sys.stdout.write(json.dumps({"path": abs_path, "alt": spec.alt or ""}) + "\n")
    elif spec.fmt == "alt":
        sys.stdout.write((spec.alt or "") + "\n")
    else:  # pragma: no cover - validated upstream
        msg = f"unknown output format: {spec.fmt}"
        raise ValueError(msg)
    sys.stdout.flush()
