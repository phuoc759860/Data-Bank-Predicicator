"""Dashboard utility, filter, and formatting tests."""

from __future__ import annotations

import pandas as pd
import pytest

from app.components.filters import apply_common_filters
from app.utils.formatting import int_text, pct_text, probability_text
from src import app_utils as au


def test_overall_summary(app_data: pd.DataFrame) -> None:
    summary = au.overall_summary(app_data)
    assert summary["total"] == 45_211
    assert summary["subscribers"] == 5_289
    assert summary["non_subscribers"] == 39_922
    assert summary["subscription_rate"] == pytest.approx(11.698480458295547)


def test_subscription_rate_by_returns_counts_and_rates(app_data: pd.DataFrame) -> None:
    summary = au.subscription_rate_by(app_data, "job")
    assert set(summary.columns) == {
        "job",
        "n_customers",
        "subscribers",
        "subscription_rate_pct",
    }
    assert int(summary["n_customers"].sum()) == 45_211
    assert int(summary["subscribers"].sum()) == 5_289
    assert summary["subscription_rate_pct"].between(0, 100).all()
    assert summary["subscription_rate_pct"].is_monotonic_decreasing


def test_subscription_rate_minimum_group_filter(app_data: pd.DataFrame) -> None:
    summary = au.subscription_rate_by(app_data, "job", min_customers=5_000)
    assert len(summary) > 0
    assert (summary["n_customers"] >= 5_000).all()


def test_apply_common_filters_does_not_mutate_source(app_data: pd.DataFrame) -> None:
    source_length = len(app_data)
    filtered = apply_common_filters(
        app_data,
        jobs=["management"],
        education=["tertiary"],
        marital=["married"],
        contact=["cellular"],
        age_range=(30, 50),
    )
    assert len(app_data) == source_length
    assert set(filtered["job"]) == {"management"}
    assert set(filtered["education"]) == {"tertiary"}
    assert set(filtered["marital"]) == {"married"}
    assert set(filtered["contact"]) == {"cellular"}
    assert filtered["age"].between(30, 50).all()
    assert len(filtered) <= source_length


def test_empty_filter_result_is_safe(app_data: pd.DataFrame) -> None:
    filtered = apply_common_filters(
        app_data,
        jobs=["management"],
        education=["unknown"],
        age_range=(18, 18),
    )
    assert filtered.empty
    assert len(app_data) == 45_211


def test_categorical_options_and_numeric_ranges(app_data: pd.DataFrame) -> None:
    options = au.load_categorical_options()
    ranges = au.load_numeric_ranges()
    assert options["job"] == sorted(app_data["job"].unique().tolist())
    assert options["poutcome"] == ["failure", "other", "success", "unknown"]
    assert ranges["age"] == (18.0, 95.0)
    assert ranges["balance"][0] == -8_019.0
    assert ranges["balance"][1] == 102_127.0
    assert ranges["pdays"] == (-1.0, 871.0)


def test_format_helpers() -> None:
    assert int_text(45_211) == "45,211"
    assert pct_text(11.698, 2) == "11.70%"
    assert pct_text(11.698) == "11.7%"
    assert probability_text(0.1511) == "15.1%"


def test_model_test_metrics_are_from_metadata(metadata: dict) -> None:
    assert au.get_model_test_metrics() == metadata["test_metrics_at_selected_threshold"]
