from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from bayescycle.backends.bayesjax.runner import (
    InProcessBackendError,
    _load_data,
    _load_sample_function,
)


def test_load_data_materializes_canonical_data_doc(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(
        '{"format":"bayescycle.data.json.v1",'
        '"variables":{"x":{"dtype":"float64","shape":[2],"values":[1.0,2.0]},'
        '"y":{"dtype":"int64","shape":[],"values":[3]}}}\n',
        encoding="utf-8",
    )

    assert _load_data(path) == {"x": [1.0, 2.0], "y": 3}


def test_inprocess_backend_missing_extra_error_is_repair_oriented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_bayesjax_inference(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "bayesjax.inference":
            raise ImportError("simulated missing BlackJAX")
        importer = original_import
        return importer(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(
        builtins,
        "__import__",
        fail_bayesjax_inference,
    )

    with pytest.raises(InProcessBackendError, match=r"bayescycle\[inproc\]"):
        _load_sample_function()
