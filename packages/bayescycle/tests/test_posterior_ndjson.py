from __future__ import annotations

import pytest

pytest.importorskip("bayesjax", reason="posterior artifact writer is inproc-backend code")

from bayescycle._run_artifacts.posterior_ndjson import (  # noqa: E402
    PosteriorArtifactError,
    _probability_float,
)


def test_probability_float_clamps_float32_acceptance_roundoff() -> None:
    assert _probability_float(1.0000001192092896) == 1.0
    assert _probability_float(-1e-8) == 0.0


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probability_float_rejects_non_probability_values(value: float) -> None:
    with pytest.raises(PosteriorArtifactError, match=r"\[0, 1\]"):
        _probability_float(value)
