def test_package_imports() -> None:
    import bayesjax
    import bayesjax.compiler
    import bayesjax.diagnostics
    import bayesjax.inference
    import bayesjax.model

    # Sanity: __init__ should not leak random names
    assert hasattr(bayesjax, "__all__")


def test_top_level_exports_the_binding_surface() -> None:
    import bayesjax

    assert set(bayesjax.__all__) == {"BoundModel", "bind_model"}


def test_compiler_exports_compile_log_density() -> None:
    import bayesjax.compiler as compiler
    from bayesjax.compiler import compile_log_density
    from bayesjax.compiler.core import compile_log_density as core_compile_log_density

    assert compile_log_density is core_compile_log_density
    assert compiler.__all__ == ["compile_log_density"]
