"""Tiny model.ir.json helpers for exporter metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from bayesite_idata.errors import AssemblyError

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def observed_data_names(path: Path) -> frozenset[str] | None:
    """Return observed data names encoded in model.ir.json when discoverable."""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = cast(JsonValue, json.load(f))
    except json.JSONDecodeError as exc:
        msg = f"{path}: invalid model.ir.json: {exc.msg}"
        raise AssemblyError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"{path}: model.ir.json must be an object"
        raise AssemblyError(msg)
    model = raw.get("model")
    if not isinstance(model, dict):
        return None
    names: set[str] = set()
    observed_nodes = model.get("observed_nodes")
    if isinstance(observed_nodes, list):
        for node in observed_nodes:
            if isinstance(node, dict):
                name = node.get("name")
                if isinstance(name, str):
                    names.add(name)
    stochastic_sites = model.get("stochastic_sites")
    if isinstance(stochastic_sites, list):
        for site in stochastic_sites:
            if not isinstance(site, dict):
                continue
            value = site.get("value")
            if isinstance(value, dict) and value.get("node") == "DataRef":
                name = value.get("name")
                if isinstance(name, str):
                    names.add(name)
    return frozenset(names) if names else None
