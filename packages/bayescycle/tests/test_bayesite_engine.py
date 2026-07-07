from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bayescycle._errors import WorkflowError
from bayescycle.backends.bayesite.preflight import (
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
    engine = _write_engine(
        tmp_path,
        "usage: bayesite sample\nusage: bayesite simulate\nusage: bayesite recover-check",
    )

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
        "    print('usage: bayesite sample\\nusage: bayesite simulate')\n"
        "    raise SystemExit(0)\n"
        "if not sys.argv[1:]:\n"
        f"    Path({str(side_effect)!r}).write_text('ran')\n"
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
        "    print("
        "'usage: bayesite sample\\nusage: bayesite simulate\\n'"
        "'usage: bayesite recover-check'"
        ")\n"
        "    raise SystemExit(0)\n"
        "print('missing command', file=sys.stderr)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    preflight_bayesite_engine(str(engine), (BayesiteCommandRequirement("simulate", "simulate"),))


def test_preflight_accepts_clap_style_commands_section(tmp_path: Path) -> None:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['--help']:\n"
        "    print("
        "'Usage: bayesite <COMMAND>\\n\\n'"
        "'Commands:\\n  sample    run sampler\\n  simulate  simulate data\\n\\n'"
        "'Options:\\n  -h, --help'"
        ")\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    preflight_bayesite_engine(str(engine), (BayesiteCommandRequirement("simulate", "simulate"),))


def test_preflight_does_not_parse_later_help_sections_as_commands(tmp_path: Path) -> None:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['--help']:\n"
        "    print("
        "'Usage: bayesite <COMMAND>\\n\\n'"
        "'Commands:\\n  sample    run sampler\\n\\n'"
        "'Examples:\\nsimulate --model model.ir.json\\n\\n'"
        "'Options:\\n  -h, --help'"
        ")\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    with pytest.raises(WorkflowError, match="does not support required command: simulate"):
        preflight_bayesite_engine(
            str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
        )


def test_preflight_rejects_help_text_that_only_mentions_command(tmp_path: Path) -> None:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "print('usage: wrapper can maybe run simulate')\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    with pytest.raises(WorkflowError, match="did not advertise|does not support"):
        preflight_bayesite_engine(
            str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
        )


def test_preflight_rejects_missing_engine(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="does not exist"):
        preflight_bayesite_engine(
            str(tmp_path / "missing-bayesite"),
            (BayesiteCommandRequirement("simulate", "simulate"),),
        )


def test_preflight_rejects_stale_engine_without_required_command(tmp_path: Path) -> None:
    engine = _write_engine(
        tmp_path,
        "\n".join(
            (
                "usage: bayesite sample",
                "usage: bayesite diagnose",
                "usage: bayesite recover",
                "usage: bayesite sbc",
            )
        ),
    )

    with pytest.raises(WorkflowError, match="does not support required command: simulate"):
        preflight_bayesite_engine(
            str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
        )


def test_preflight_uses_structured_capabilities_when_available(tmp_path: Path) -> None:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import sys\n"
        "if sys.argv[1:] == ['capabilities']:\n"
        "    payload = {\n"
        "        'capabilities_format': 'v0-provisional',\n"
        "        'commands': ['sample', 'simulate', 'recover-check'],\n"
        "        'version': '0.3.0',\n"
        "        'ir': {'bayeswire_ir': 1},\n"
        "        'schemas': {'ir': 'bayeswire_ir_v1.json'},\n"
        "        'unknown_field': 'ignored',\n"
        "    }\n"
        "    print(json.dumps(payload))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    info = preflight_bayesite_engine(
        str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
    )

    assert info.commands == ("sample", "simulate", "recover-check")
    assert info.capabilities is not None
    assert info.capabilities.capabilities_format == "v0-provisional"
    assert info.capabilities.commands == ("sample", "simulate", "recover-check")
    assert info.capabilities.version == "0.3.0"
    assert dict(info.capabilities.ir) == {"bayeswire_ir": 1}
    assert dict(info.capabilities.schemas) == {"ir": "bayeswire_ir_v1.json"}


def test_preflight_falls_back_to_regex_when_capabilities_subcommand_unknown(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['--help']:\n"
        "    print('usage: bayesite sample\\nusage: bayesite simulate')\n"
        "    raise SystemExit(0)\n"
        "print('unknown subcommand: capabilities', file=sys.stderr)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    info = preflight_bayesite_engine(
        str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
    )

    assert info.capabilities is None
    assert "simulate" in info.commands


def test_preflight_falls_back_when_capabilities_output_is_invalid_json(tmp_path: Path) -> None:
    engine = tmp_path / "bayesite.py"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['capabilities']:\n"
        "    print('not json')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:] == ['--help']:\n"
        "    print('usage: bayesite sample\\nusage: bayesite simulate')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    engine.chmod(0o755)

    info = preflight_bayesite_engine(
        str(engine), (BayesiteCommandRequirement("simulate", "simulate"),)
    )

    assert info.capabilities is None
    assert "simulate" in info.commands


def test_preflight_parses_single_line_json_error_probe(tmp_path: Path) -> None:
    """The real engine answers probes with one JSON object whose message embeds usage."""
    engine = tmp_path / "bayesite.py"
    usage = (
        "unknown command; "
        "usage: bayesite sample --model <ir.json|-> --data <data.json|->\n"
        "usage: bayesite diagnose --fit <fit.jsonl|->\n"
        "usage: bayesite posterior-predictive --model <ir.json|->"
    )
    body = (
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"message = {usage!r}\n"
        'sys.stderr.write(json.dumps({"error_format": "v0-provisional", '
        '"error": "InvalidSettings", "message": message}))\n'
        "sys.exit(2)\n"
    )
    engine.write_text(body, encoding="utf-8")
    engine.chmod(0o755)

    info = preflight_bayesite_engine(
        str(engine),
        (
            BayesiteCommandRequirement("sample", "sample"),
            BayesiteCommandRequirement("diagnose", "diagnose"),
            BayesiteCommandRequirement("posterior-predictive", "posterior-predictive"),
        ),
    )

    assert set(info.commands) >= {"sample", "diagnose", "posterior-predictive"}
