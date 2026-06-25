from __future__ import annotations

import builtins

import pytest

from bayescycle._inproc import InProcessBackendError, _load_sample_function


def test_inprocess_backend_missing_extra_error_is_repair_oriented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fail_jaxstanv5_inference(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "jaxstanv5.inference":
            raise ImportError("simulated missing BlackJAX")
        importer = original_import
        return importer(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(
        builtins,
        "__import__",
        fail_jaxstanv5_inference,
    )

    with pytest.raises(InProcessBackendError, match=r"bayescycle\[inproc\]"):
        _load_sample_function()
