"""Closed immutable functional generation plans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

MAX_GENERATION_COUNT = 1000
MAX_GENERATION_INPUT_BYTES = 8 * 1024 * 1024
MAX_GENERATION_PLAN_BYTES = 1024 * 1024
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_MAX_DEPTH = 64
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class GenerationPlanError(ValueError):
    """Raised when a functional generation plan is invalid."""


class _DuplicateKey(ValueError):
    pass


class FitAssociation(StrEnum):
    """Authority carried by a posterior fit artifact."""

    RUNTIME = "runtime"
    PORTABLE = "portable"


@dataclass(frozen=True)
class AuthoredProvenance:
    claimed_source_model_hash: str
    claimed_outcome_model_hash: str

    def __post_init__(self) -> None:
        _hash(self.claimed_source_model_hash, "claimed_source_model_hash")
        _hash(self.claimed_outcome_model_hash, "claimed_outcome_model_hash")


@dataclass(frozen=True)
class Fixed:
    parameters_bytes: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters_bytes",
            _copy_bytes(self.parameters_bytes, "fixed parameters"),
        )


@dataclass(frozen=True)
class ModelPrior:
    model_ir_bytes: bytes
    authored_provenance: AuthoredProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_ir_bytes",
            _copy_bytes(self.model_ir_bytes, "model-prior model IR"),
        )
        if self.authored_provenance is not None and not isinstance(
            self.authored_provenance, AuthoredProvenance
        ):
            raise GenerationPlanError("authored provenance must be an immutable value")


@dataclass(frozen=True)
class FitArtifact:
    model_ir_bytes: bytes
    data_bytes: bytes
    posterior_bytes: bytes
    association: FitAssociation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_ir_bytes",
            _copy_bytes(self.model_ir_bytes, "fit model IR"),
        )
        object.__setattr__(self, "data_bytes", _copy_bytes(self.data_bytes, "fit data"))
        object.__setattr__(
            self,
            "posterior_bytes",
            _copy_bytes(self.posterior_bytes, "fit posterior"),
        )
        if not isinstance(self.association, FitAssociation):
            raise GenerationPlanError("fit association must be runtime or portable")


@dataclass(frozen=True)
class PosteriorOf:
    fit_artifact: FitArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.fit_artifact, FitArtifact):
            raise GenerationPlanError("posterior source requires a fit artifact")


type ParameterSource = Fixed | ModelPrior | PosteriorOf


@dataclass(frozen=True)
class OutcomesOf:
    model_ir_bytes: bytes
    design_bytes: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_ir_bytes",
            _copy_bytes(self.model_ir_bytes, "outcomes model IR"),
        )
        object.__setattr__(
            self,
            "design_bytes",
            _copy_bytes(self.design_bytes, "generation design"),
        )


@dataclass(frozen=True)
class JointPredict:
    parameters: ParameterSource
    outcomes: OutcomesOf

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Fixed | ModelPrior | PosteriorOf):
            raise GenerationPlanError("parameter source has unknown kind")
        if not isinstance(self.outcomes, OutcomesOf):
            raise GenerationPlanError("joint prediction requires model outcomes")
        source_model = _source_model(self.parameters)
        if source_model is not None and source_model != self.outcomes.model_ir_bytes:
            raise GenerationPlanError("parameter source model must equal the outcomes model bytes")


@dataclass(frozen=True)
class Draw:
    distribution: JointPredict
    count: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, JointPredict):
            raise GenerationPlanError("draw requires a joint-predict distribution")
        _integer(self.count, "count", minimum=1, maximum=MAX_GENERATION_COUNT)
        _integer(self.seed, "seed", minimum=0, maximum=_MAX_SAFE_INTEGER)


@dataclass(frozen=True)
class GenerationPlanDocument:
    bytes: bytes
    identity_hash: str
    invalidation_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes", bytes(self.bytes))


def generate_datasets(
    model_ir: bytes,
    *,
    design: bytes,
    parameter_source: ParameterSource,
    count: int = 100,
    seed: int = 0,
) -> Draw:
    """Build the user-facing dataset generation plan."""
    return Draw(
        distribution=JointPredict(
            parameters=parameter_source,
            outcomes=OutcomesOf(model_ir_bytes=model_ir, design_bytes=design),
        ),
        count=count,
        seed=seed,
    )


def serialize_generation_plan(plan: Draw) -> bytes:
    """Serialize hash-only versioned generation provenance."""
    if not isinstance(plan, Draw):
        raise GenerationPlanError("generation plan must be a Draw value")
    parameters = _source_document(plan.distribution.parameters)
    outcomes = plan.distribution.outcomes
    document: dict[str, JsonValue] = {
        "generation_plan_format": "v0-provisional",
        "kind": "draw",
        "count": plan.count,
        "seed": plan.seed,
        "distribution": {
            "kind": "joint-predict",
            "parameters": parameters,
            "outcomes": {
                "kind": "model-outcomes",
                "model_hash": _sha256(outcomes.model_ir_bytes),
                "design_hash": _sha256(outcomes.design_bytes),
            },
        },
    }
    data = json.dumps(document, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(data) > MAX_GENERATION_PLAN_BYTES:
        raise GenerationPlanError("serialized generation plan exceeds byte limit")
    return data


def parse_generation_plan_document(data: bytes) -> GenerationPlanDocument:
    """Parse strict hash-only generation provenance."""
    source = _copy_bytes(data, "generation plan", maximum=MAX_GENERATION_PLAN_BYTES)
    if not source.endswith(b"\n"):
        raise GenerationPlanError("generation plan must end in one LF")
    _validate_depth(source)
    try:
        value = cast(
            JsonValue,
            json.loads(source.decode("utf-8"), object_pairs_hook=_unique_object),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey) as exc:
        raise GenerationPlanError(f"generation plan is not valid JSON: {exc}") from exc
    document = _object(value, "generation plan")
    _exact_keys(
        document,
        ("generation_plan_format", "kind", "count", "seed", "distribution"),
        "generation plan",
    )
    if document["generation_plan_format"] != "v0-provisional":
        raise GenerationPlanError("generation plan format must be v0-provisional")
    if document["kind"] != "draw":
        raise GenerationPlanError("generation plan kind must be draw")
    _integer(document["count"], "count", minimum=1, maximum=MAX_GENERATION_COUNT)
    _integer(document["seed"], "seed", minimum=0, maximum=_MAX_SAFE_INTEGER)
    distribution = _object(document["distribution"], "distribution")
    _exact_keys(distribution, ("kind", "parameters", "outcomes"), "distribution")
    if distribution["kind"] != "joint-predict":
        raise GenerationPlanError("distribution kind must be joint-predict")
    source_model_hash = _parse_source_document(distribution["parameters"])
    outcomes = _object(distribution["outcomes"], "outcomes")
    _exact_keys(outcomes, ("kind", "model_hash", "design_hash"), "outcomes")
    if outcomes["kind"] != "model-outcomes":
        raise GenerationPlanError("outcomes kind must be model-outcomes")
    outcome_model_hash = _hash(outcomes["model_hash"], "model_hash")
    _hash(outcomes["design_hash"], "design_hash")
    if source_model_hash is not None and source_model_hash != outcome_model_hash:
        raise GenerationPlanError("parameter source model hash must equal outcomes model hash")
    identity = _sha256(source)
    return GenerationPlanDocument(
        bytes=source,
        identity_hash=identity,
        invalidation_key=_sha256(b"bayescycle-generation-key-v0\n" + source),
    )


def resolve_generation_plan_document(
    data: bytes,
    *,
    model_ir_bytes: bytes,
    design_bytes: bytes,
    fixed_parameters_bytes: bytes | None = None,
    fit_data_bytes: bytes | None = None,
    posterior_bytes: bytes | None = None,
) -> Draw:
    """Resolve a strict hash-only plan against co-travelling exact payload bytes."""
    source = parse_generation_plan_document(data).bytes
    document = cast(dict[str, JsonValue], json.loads(source.decode("utf-8")))
    distribution = cast(dict[str, JsonValue], document["distribution"])
    source_document = cast(dict[str, JsonValue], distribution["parameters"])
    kind = source_document["kind"]
    parameter_source: ParameterSource
    if kind == "fixed":
        if fixed_parameters_bytes is None:
            raise GenerationPlanError("fixed plan requires fixed parameter payload bytes")
        parameter_source = Fixed(fixed_parameters_bytes)
    elif kind == "model-prior":
        raw_provenance = source_document["authored_provenance"]
        provenance = None
        if raw_provenance is not None:
            claims = cast(dict[str, JsonValue], raw_provenance)
            provenance = AuthoredProvenance(
                claimed_source_model_hash=cast(str, claims["claimed_source_model_hash"]),
                claimed_outcome_model_hash=cast(str, claims["claimed_outcome_model_hash"]),
            )
        parameter_source = ModelPrior(model_ir_bytes, provenance)
    elif kind == "posterior":
        if fit_data_bytes is None or posterior_bytes is None:
            raise GenerationPlanError(
                "posterior plan requires source posterior and fit-data payload bytes"
            )
        parameter_source = PosteriorOf(
            FitArtifact(
                model_ir_bytes=model_ir_bytes,
                data_bytes=fit_data_bytes,
                posterior_bytes=posterior_bytes,
                association=FitAssociation.PORTABLE,
            )
        )
    else:
        raise GenerationPlanError(f"parameter source has unknown kind {kind!r}")
    plan = generate_datasets(
        model_ir_bytes,
        design=design_bytes,
        parameter_source=parameter_source,
        count=cast(int, document["count"]),
        seed=cast(int, document["seed"]),
    )
    if serialize_generation_plan(plan) != source:
        raise GenerationPlanError("generation plan payload hashes do not match resolved bytes")
    return plan


def generation_plan_identity(plan: Draw) -> str:
    """Return the deterministic serialized-plan identity."""
    return _sha256(serialize_generation_plan(plan))


def generation_invalidation_key(plan: Draw) -> str:
    """Return the deterministic generation dependency key."""
    return _sha256(b"bayescycle-generation-key-v0\n" + serialize_generation_plan(plan))


def _source_model(source: ParameterSource) -> bytes | None:
    if isinstance(source, ModelPrior):
        return source.model_ir_bytes
    if isinstance(source, PosteriorOf):
        return source.fit_artifact.model_ir_bytes
    return None


def _source_document(source: ParameterSource) -> dict[str, JsonValue]:
    if isinstance(source, Fixed):
        return {"kind": "fixed", "parameters_hash": _sha256(source.parameters_bytes)}
    if isinstance(source, ModelPrior):
        provenance: JsonValue = None
        if source.authored_provenance is not None:
            provenance = {
                "claimed_source_model_hash": source.authored_provenance.claimed_source_model_hash,
                "claimed_outcome_model_hash": source.authored_provenance.claimed_outcome_model_hash,
            }
        return {
            "kind": "model-prior",
            "model_hash": _sha256(source.model_ir_bytes),
            "authored_provenance": provenance,
        }
    fit = source.fit_artifact
    return {
        "kind": "posterior",
        "fit_hash": _sha256(fit.posterior_bytes),
        "fit_model_hash": _sha256(fit.model_ir_bytes),
        "fit_data_hash": _sha256(fit.data_bytes),
    }


def _parse_source_document(value: JsonValue) -> str | None:
    source = _object(value, "parameter source")
    kind = source.get("kind")
    if kind == "fixed":
        _exact_keys(source, ("kind", "parameters_hash"), "fixed parameter source")
        _hash(source["parameters_hash"], "parameters_hash")
        return None
    if kind == "model-prior":
        _exact_keys(
            source,
            ("kind", "model_hash", "authored_provenance"),
            "model-prior parameter source",
        )
        _hash(source["model_hash"], "model_hash")
        provenance = source["authored_provenance"]
        if provenance is not None:
            claims = _object(provenance, "authored provenance")
            _exact_keys(
                claims,
                ("claimed_source_model_hash", "claimed_outcome_model_hash"),
                "authored provenance",
            )
            _hash(claims["claimed_source_model_hash"], "claimed_source_model_hash")
            _hash(claims["claimed_outcome_model_hash"], "claimed_outcome_model_hash")
        return _hash(source["model_hash"], "model_hash")
    if kind == "posterior":
        _exact_keys(
            source,
            ("kind", "fit_hash", "fit_model_hash", "fit_data_hash"),
            "posterior parameter source",
        )
        for name in ("fit_hash", "fit_model_hash", "fit_data_hash"):
            _hash(source[name], name)
        return _hash(source["fit_model_hash"], "fit_model_hash")
    raise GenerationPlanError(f"parameter source has unknown kind {kind!r}")


def _copy_bytes(value: object, label: str, *, maximum: int = MAX_GENERATION_INPUT_BYTES) -> bytes:
    if not isinstance(value, bytes | bytearray | memoryview):
        raise GenerationPlanError(f"{label} must be bytes")
    copied = bytes(memoryview(value))
    if not copied:
        raise GenerationPlanError(f"{label} bytes must not be empty")
    if len(copied) > maximum:
        raise GenerationPlanError(f"{label} exceeds {maximum} bytes")
    return copied


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate object key {key}")
        result[key] = value
    return result


def _validate_depth(data: bytes) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > _MAX_DEPTH:
                raise GenerationPlanError(f"generation plan exceeds nesting depth {_MAX_DEPTH}")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise GenerationPlanError("generation plan has malformed nesting")
    if in_string or depth != 0:
        raise GenerationPlanError("generation plan has malformed nesting")


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise GenerationPlanError(f"{label} must be an object")
    return value


def _exact_keys(value: dict[str, JsonValue], expected: tuple[str, ...], label: str) -> None:
    actual = tuple(value)
    if actual != expected:
        raise GenerationPlanError(
            f"{label} has unknown, missing, or out-of-order fields: {list(actual)!r}"
        )


def _integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise GenerationPlanError(f"{label} must be an integer in {minimum}..{maximum}")
    return value


def _hash(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise GenerationPlanError(f"{label} must be a lowercase sha256 hash")
    return value


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
