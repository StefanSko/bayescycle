from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayescycle.data import (
    DATA_DOC_FORMAT,
    DataDocError,
    data_doc_to_plain_json,
    normalize_data_doc,
    read_data_doc,
    write_data_doc,
)


def test_canonical_data_doc_round_trips_and_materializes() -> None:
    doc = normalize_data_doc(
        {
            "format": DATA_DOC_FORMAT,
            "variables": {
                "x": {"dtype": "float64", "shape": [2, 2], "values": [0.1, 0.2, 0.3, 0.4]},
                "y": {"dtype": "int64", "shape": [2], "values": [0, 1]},
            },
        }
    )

    assert doc.to_json()["format"] == DATA_DOC_FORMAT
    assert data_doc_to_plain_json(doc) == {"x": [[0.1, 0.2], [0.3, 0.4]], "y": [0, 1]}


def test_legacy_plain_json_data_normalizes_to_canonical(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"flag": true, "x": [[1, 2], [3, 4]], "y": 1.25}\n')

    doc = read_data_doc(path)
    out = tmp_path / "canonical.json"
    write_data_doc(out, doc)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == {
        "format": DATA_DOC_FORMAT,
        "variables": {
            "flag": {"dtype": "bool", "shape": [], "values": [True]},
            "x": {"dtype": "int64", "shape": [2, 2], "values": [1, 2, 3, 4]},
            "y": {"dtype": "float64", "shape": [], "values": [1.25]},
        },
    }


def test_legacy_variables_named_format_and_variables_are_preserved() -> None:
    doc = normalize_data_doc({"format": 1, "variables": 2})

    assert doc.to_json() == {
        "format": DATA_DOC_FORMAT,
        "variables": {
            "format": {"dtype": "int64", "shape": [], "values": [1]},
            "variables": {"dtype": "int64", "shape": [], "values": [2]},
        },
    }


def test_canonical_data_doc_rejects_shape_value_mismatch() -> None:
    with pytest.raises(DataDocError, match="values length 2 does not match shape"):
        normalize_data_doc(
            {
                "format": DATA_DOC_FORMAT,
                "variables": {"x": {"dtype": "float64", "shape": [3], "values": [1.0, 2.0]}},
            }
        )


def test_legacy_ragged_arrays_are_rejected() -> None:
    with pytest.raises(DataDocError, match="rectangular"):
        normalize_data_doc({"x": [[1.0], [2.0, 3.0]]})
