from __future__ import annotations

from pathlib import Path

from bayesite_viz.output import OutputSpec, announce, auto_alt, resolve_output_path


def test_resolve_with_output_overwrites_target(tmp_path: Path) -> None:
    # -o is used verbatim; existing file would be overwritten (caller's choice)
    p = tmp_path / "out.png"
    p.write_bytes(b"x")
    got = resolve_output_path("trace", str(p), "png")
    assert got == p


def test_resolve_default_uses_verb_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    got = resolve_output_path("trace", None, "png")
    assert got == Path("trace.png")


def test_resolve_default_increments_to_avoid_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trace.png").write_bytes(b"x")
    (tmp_path / "trace-2.png").write_bytes(b"x")
    got = resolve_output_path("trace", None, "png")
    assert got == Path("trace-3.png")


def test_announce_path_prints_absolute(tmp_path: Path, capsys) -> None:
    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="path", alt=None))
    out = capsys.readouterr().out
    assert out.strip() == str(p.resolve())
    assert out.endswith("\n")


def test_announce_markdown(tmp_path: Path, capsys) -> None:
    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="markdown", alt="rank plot"))
    assert capsys.readouterr().out == f"![rank plot]({p.resolve()})\n"


def test_announce_html(tmp_path: Path, capsys) -> None:
    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="html", alt="a"))
    assert capsys.readouterr().out == f'<img src="{p.resolve()}" alt="a">\n'


def test_announce_html_escapes_alt(tmp_path: Path, capsys) -> None:
    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="html", alt='x" onerror="evil'))
    out = capsys.readouterr().out
    assert '"onerror' not in out
    assert "&quot;" in out


def test_announce_html_escapes_path(tmp_path: Path, capsys) -> None:
    p = tmp_path / 'weird&<".png'
    announce(OutputSpec(path=p, fmt="html", alt="a"))
    out = capsys.readouterr().out
    # path is URL-quoted (so & < " become %XX) then html-escaped for the attr;
    # none of the raw delimiters survive into the src attribute.
    src = out.split('src="')[1].split('"')[0]
    assert "&" not in src
    assert "<" not in src
    assert '"' not in src


def test_announce_markdown_escapes_alt(tmp_path: Path, capsys) -> None:
    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="markdown", alt="a]b"))
    out = capsys.readouterr().out
    # the ] in alt must not close the image alt early
    assert out.startswith("![a\\]b](")


def test_announce_markdown_escapes_backslash_before_bracket(tmp_path: Path, capsys) -> None:
    # alt ending in \] must not let the ] close the image alt early after a
    # markdown parser unescapes the doubled backslash. The injected
    # ](https://evil) must stay inside the alt, not become the image URL.
    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="markdown", alt=r"x\](https://evil)"))
    out = capsys.readouterr().out
    # the ] before (https://evil) is backslash-escaped (\\]), so a markdown
    # parser keeps it in the alt; the real URL is the resolved path.
    assert out.startswith(r"![x\\\](https://evil)](")
    assert out.endswith(f"]({p.resolve()})\n")


def test_announce_json_escapes(tmp_path: Path, capsys) -> None:
    import json

    p = tmp_path / 'q".png'
    announce(OutputSpec(path=p, fmt="json", alt='x"y'))
    got = json.loads(capsys.readouterr().out)
    assert got["alt"] == 'x"y'


def test_announce_markdown_encodes_spaces_in_url(tmp_path: Path, capsys) -> None:
    # a path containing whitespace must not leave raw spaces in the markdown
    # image destination (CommonMark rejects unencoded spaces).
    p = tmp_path / "my charts" / "trace.png"
    p.parent.mkdir()
    p.write_bytes(b"")
    announce(OutputSpec(path=p, fmt="markdown", alt="t"))
    out = capsys.readouterr().out
    assert "]( " not in out
    assert "%20" in out
    assert str(p.resolve()).replace(" ", "%20") in out


def test_announce_markdown_url_quotes_fragment_and_query(tmp_path: Path, capsys) -> None:
    p = tmp_path / "trace#2.png?v=1"
    announce(OutputSpec(path=p, fmt="markdown", alt="t"))
    out = capsys.readouterr().out
    # # and ? must be percent-encoded so they aren't treated as fragment/query
    assert "%23" in out
    assert "%3F" in out
    assert "#" not in out.split("](")[1]


def test_announce_html_url_quotes_fragment(tmp_path: Path, capsys) -> None:
    p = tmp_path / "trace#2.png"
    announce(OutputSpec(path=p, fmt="html", alt="t"))
    out = capsys.readouterr().out
    assert "%23" in out
    assert "#" not in out.split('src="')[1].split('"')[0]


def test_announce_json(tmp_path: Path, capsys) -> None:
    import json

    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="json", alt="a"))
    got = json.loads(capsys.readouterr().out)
    assert got == {"path": str(p.resolve()), "alt": "a"}


def test_announce_alt(tmp_path: Path, capsys) -> None:
    p = tmp_path / "rank.png"
    announce(OutputSpec(path=p, fmt="alt", alt="only alt"))
    assert capsys.readouterr().out == "only alt\n"


def test_auto_alt_basic(tmp_path: Path) -> None:
    fit = tmp_path / "fit.nc"
    assert auto_alt("trace", fit, var_names=("mu", "tau")) == "trace plot of mu, tau from fit.nc"


def test_auto_alt_all_vars(tmp_path: Path) -> None:
    fit = tmp_path / "run.nc"
    assert auto_alt("forest", fit) == "forest plot of all variables from run.nc"


def test_auto_alt_ppc_kind(tmp_path: Path) -> None:
    fit = tmp_path / "fit.nc"
    assert (
        auto_alt("ppc", fit, var_names=("obs",), kind="dist")
        == "ppc (dist) plot of obs from fit.nc"
    )
