"""Exact structural equality for resolved metadata roles."""

from __future__ import annotations

from dataclasses import fields, is_dataclass


def _structurally_equal(left: object, right: object) -> bool:
    """Compare every runtime field without dataclass equality shortcuts."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        left_items = tuple(left.items())
        right_items = tuple(right.items())
        return len(left_items) == len(right_items) and all(
            _structurally_equal(left_key, right_key)
            and _structurally_equal(left_value, right_value)
            for (left_key, left_value), (right_key, right_value) in zip(
                left_items,
                right_items,
                strict=True,
            )
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _structurally_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if is_dataclass(left) and not isinstance(left, type):
        if not is_dataclass(right) or isinstance(right, type):
            return False
        return all(
            _structurally_equal(
                getattr(left, value_field.name),
                getattr(right, value_field.name),
            )
            for value_field in fields(left)
        )
    return left == right
