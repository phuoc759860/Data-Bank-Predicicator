"""Dataset loading, data-quality, and feature-engineering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import app_utils as au
from src import phase2_preprocessing as p2

RAW_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]


def test_raw_dataset_shape_and_schema(raw_df: pd.DataFrame) -> None:
    assert raw_df.shape == (45_211, 17)
    assert list(raw_df.columns) == RAW_COLUMNS
    assert raw_df["y"].value_counts().to_dict() == {"no": 39_922, "yes": 5_289}


def test_raw_dataset_missing_values_and_unknowns(raw_df: pd.DataFrame) -> None:
    assert int(raw_df.isna().sum().sum()) == 0
    assert int(raw_df.duplicated().sum()) == 0
    assert "unknown" in set(raw_df["job"])
    assert "unknown" in set(raw_df["education"])
    assert "unknown" in set(raw_df["contact"])
    assert "unknown" in set(raw_df["poutcome"])
    assert int((raw_df["pdays"] == -1).sum()) == 36_954


def test_app_data_loads_and_preserves_observed_counts(app_data: pd.DataFrame) -> None:
    assert len(app_data) == 45_211
    assert int(app_data["subscribed"].sum()) == 5_289
    assert app_data["subscribed"].nunique() == 2
    summary = au.overall_summary(app_data)
    assert summary["total"] == 45_211
    assert summary["subscribers"] == 5_289
    assert summary["non_subscribers"] == 39_922
    assert summary["subscription_rate"] == pytest.approx(11.6984804583)


def test_feature_engineering_matches_phase2(raw_sample: pd.DataFrame) -> None:
    actual = au.build_engineered_features(raw_sample)
    expected = p2.engineer_features(raw_sample)
    common = [column for column in expected.columns if column in actual.columns]
    pd.testing.assert_frame_equal(
        actual[common].reset_index(drop=True),
        expected[common].reset_index(drop=True),
        check_dtype=False,
    )
    assert "duration" not in actual.columns
    assert "pdays" not in actual.columns
    assert all(feature in actual.columns for feature in au.ENGINEERED_FEATURES)


def test_feature_engineering_sentinel_and_bands(raw_df: pd.DataFrame) -> None:
    sample = raw_df.head(2).copy()
    sample.loc[sample.index[0], "pdays"] = -1
    sample.loc[sample.index[1], "pdays"] = 10
    sample.loc[sample.index[0], "campaign"] = 1
    sample.loc[sample.index[1], "campaign"] = 4
    sample.loc[sample.index[0], "age"] = 20
    sample.loc[sample.index[1], "age"] = 66
    sample.loc[sample.index[0], "poutcome"] = "unknown"
    sample.loc[sample.index[1], "poutcome"] = "success"

    engineered = au.build_engineered_features(sample)
    assert engineered["has_previous_contact"].tolist() == [0, 1]
    assert engineered["pdays_since_previous_contact"].tolist() == [0, 10]
    assert engineered["previous_campaign_success"].tolist() == [0, 1]
    assert engineered["age_group"].tolist() == ["18-24", "65+"]
    assert engineered["contact_frequency_group"].tolist() == ["1", "4-5"]


def test_feature_engineering_accepts_binary_numeric_encoding(raw_sample: pd.DataFrame) -> None:
    sample = raw_sample.copy()
    for column in ["default", "housing", "loan"]:
        sample[column] = np.where(sample[column] == "yes", 1, 0)
    engineered = au.build_engineered_features(sample)
    assert set(engineered["default"].unique()).issubset({0, 1})
    assert set(engineered["housing"].unique()).issubset({0, 1})
    assert set(engineered["loan"].unique()).issubset({0, 1})


def test_unknown_categories_are_flagged_not_crash_prediction(raw_sample: pd.DataFrame) -> None:
    sample = raw_sample.copy()
    sample.loc[sample.index[0], "job"] = "astronaut"
    unseen = au.find_unseen_categories(sample)
    assert unseen["job"] == ["astronaut"]
    predictions = au.predict_batch(sample)
    assert len(predictions) == len(sample)
    assert predictions["predicted_probability"].between(0, 1).all()
