def test_package_imports() -> None:
    import bayeswire
    import bayeswire.constraints
    import bayeswire.distributions
    import bayeswire.ir
    import bayeswire.math
    import bayeswire.model

    # Sanity: __init__ should not leak random names
    assert hasattr(bayeswire, "__all__")


def test_top_level_exports_the_declaration_language() -> None:
    import bayeswire

    assert set(bayeswire.__all__) == {
        "Data",
        "Dim",
        "Observed",
        "Param",
        "PartiallyObserved",
        "Submodel",
        "dimension_metadata_to_dict",
        "model",
        "model_dependencies",
        "model_dimensions",
        "with_prior",
    }
