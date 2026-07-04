"""The shared model/data fingerprint: exact bytes in, sha256 string out."""

from __future__ import annotations

import hashlib

from bayeswire.ir import model_data_fingerprint


def test_fingerprint_is_the_framed_sha256_of_the_exact_bytes() -> None:
    model_document = b'{"bayeswire_ir":1,"model":{}}'
    data_document = b'{"format":"bayescycle.data.json.v1","variables":{}}'

    expected = hashlib.sha256(
        b"bayescycle-model-data-v1\n" + model_document + b"\n" + data_document
    ).hexdigest()

    assert model_data_fingerprint(model_document, data_document) == f"sha256:{expected}"


def test_fingerprint_known_answer_vector() -> None:
    # Pinned independently of the implementation: any change to the framing
    # prefix, separator, digest, or rendering breaks this exact string.
    assert model_data_fingerprint(b"model-bytes", b"data-bytes") == (
        "sha256:51e42ece670eac6b5c9a3f83e1f6b9f0856f5a05feca1c7a2d58a0ad63697d81"
    )


def test_fingerprint_is_byte_exact_not_semantic() -> None:
    model_document = b'{"bayeswire_ir":1,"model":{}}'
    data_document = b'{"x": [1.0]}'

    baseline = model_data_fingerprint(model_document, data_document)

    assert model_data_fingerprint(model_document, data_document + b"\n") != baseline
    assert model_data_fingerprint(model_document + b" ", data_document) != baseline


def test_fingerprint_format_shape() -> None:
    fingerprint = model_data_fingerprint(b"", b"")

    prefix, _, digest = fingerprint.partition(":")
    assert prefix == "sha256"
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
