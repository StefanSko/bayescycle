"""Model declaration and resolved metadata."""

from bayeswire.model.core import Data, Observed, Param, PartiallyObserved
from bayeswire.model.decorator import (
    ModelMeta,
    attached_model_dimensions,
    is_model_class,
    model,
    model_meta,
    resolved_free_values,
    resolved_stochastic_sites,
)
from bayeswire.model.dimensions import (
    Dim,
    dimension_metadata_from_dict,
    dimension_metadata_to_dict,
    model_dimensions,
)

__all__ = [
    "Data",
    "Dim",
    "ModelMeta",
    "Observed",
    "Param",
    "PartiallyObserved",
    "attached_model_dimensions",
    "dimension_metadata_from_dict",
    "dimension_metadata_to_dict",
    "is_model_class",
    "model",
    "model_dimensions",
    "model_meta",
    "resolved_free_values",
    "resolved_stochastic_sites",
]
