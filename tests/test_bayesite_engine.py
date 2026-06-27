from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bayescycle._errors import WorkflowError
from bayescycle.backends.bayesite_engine import (
    BayesiteCommandRequirement,
    preflight_bayesite_engine,
)


def _write_engine(tmp_path: Path, usage: str) -> Path:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\nimport sys\nprint({usage!r})\nraise SystemExit(0)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)
    return engine


def test_preflight_accepts_required_command(tmp_path: Path) -> None:
    engine = _write_engine(tmp_path, "usage: bayesite sample simulate recover-check")

    info = preflight_bayesite_engine(
        str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
    )

    assert info.executable == engine.resolve(strict=False)


def test_preflight_does_not_invoke_zero_argument_engine_path(tmp_path: Path) -> None:
    side_effect = tmp_path / "zero-args-ran"
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "if sys.argv[1:] == ['--help']:\n"
        "    print('usage: bayesite sample simulate recover-check')\n"
        "    raise SystemExit(0)\n"
        f"Path({str(side_effect)!r}).write_text('ran')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    preflight_bayesite_engine(str(engine), (BayesiteCommandRequirement("simulate", "simulate"),))

    assert not side_effect.exists()


def test_preflight_accepts_engine_that_lists_commands_only_on_help(tmp_path: Path) -> None:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['--help']:\n"
        "    print('usage: bayesite sample simulate recover-check')\n"
        "    raise SystemExit(0)\n"
        "print('missing command', file=sys.stderr)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    preflight_bayesite_engine(str(engine), (BayesiteCommandRequirement("simulate", "simulate"),))


def test_preflight_rejects_missing_engine(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="does not exist"):
        preflight_bayesite_engine(
            str(tmp_path / "missing-bayesite"),
            (BayesiteCommandRequirement("simulate", "simulate"),),
        )


def test_preflight_rejects_stale_engine_without_required_command(tmp_path: Path) -> None:
    engine = _write_engine(tmp_path, "usage: bayesite sample diagnose recover sbc")

    with pytest.raises(WorkflowError, match="does not support required command: simulate"):
        preflight_bayesite_engine(
            str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
        )
