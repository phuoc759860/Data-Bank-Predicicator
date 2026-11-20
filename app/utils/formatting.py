"""Formatting helpers shared by pages and KPI cards."""

from __future__ import annotations


def int_text(value: int | float) -> str:
    """Format a count with thousands separators."""
    return f"{int(value):,}"


def pct_text(value: float, digits: int = 1) -> str:
    """Format a rate already expressed in percentage points."""
    return f"{value:.{digits}f}%"


def probability_text(value: float) -> str:
    """Format a model probability as a percentage."""
    return f"{100.0 * value:.1f}%"
