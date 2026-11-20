"""Model artifact, preprocessing, prediction-output, and validation tests."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import app_utils as au

EXPECTED_TEST_METRICS = {
    "accuracy": 0.8705075749198274,
    "precision": 0.4543987086359968,
    "recall": 0.5321361058601134,
    "f1": 0.49020461471484544,
    "roc_auc": 0.8046476557534035,
    "average_precision": 0.45572110223378215,
}


def test_artifact_structure_and_feature_contract(artifact: dict, metadata: dict) -> None:
    required = {
        "model",
        "preprocessing_pipeline",
        "feature_names",
        "threshold",
        "model_name",
    }
    assert required.issubset(artifact)
    assert artifact["model_name"] == "Random Forest"
    assert artifact["threshold"] == 0.2
    assert metadata["selected_model"] == "Random Forest"
    assert metadata["threshold"] == 0.2
    assert metadata["duration_included"] is False
    assert metadata["test_set_used_for_selection"] is False
    assert len(artifact["feature_names"]) == 59
    assert artifact["model"].n_features_in_ == 59
    assert list(artifact["feature_names"]) == metadata["model_input_features"]
    assert not any("duration" in str(name) for name in artifact["feature_names"])


def test_processed_matrices_match_model_contract(artifact: dict) -> None:
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")["y"]
    y_test = pd.read_csv("data/processed/y_test.csv")["y"]
    assert X_train.shape == (36_168, 59)
    assert X_test.shape == (9_043, 59)
    assert list(X_test.columns) == list(artifact["feature_names"])
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert int(y_test.sum()) == 1_058


def test_saved_pipeline_matches_exported_pipeline(
    artifact: dict, raw_sample: pd.DataFrame
) -> None:
    engineered = au.build_engineered_features(raw_sample)
    external_pipeline = joblib.load("models/preprocessing_pipeline.joblib")
    transformed = artifact["preprocessing_pipeline"].transform(engineered)
    expected = external_pipeline.transform(engineered)
    assert transformed.shape == (len(raw_sample), 59)
    assert np.allclose(transformed, expected)


def test_reloaded_model_reproduces_phase4_test_metrics(artifact: dict) -> None:
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")["y"]
    probabilities = artifact["model"].predict_proba(X_test)[:, 1]
    labels = (probabilities >= float(artifact["threshold"])).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, labels),
        "precision": precision_score(y_test, labels),
        "recall": recall_score(y_test, labels),
        "f1": f1_score(y_test, labels),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
    }
    for name, expected in EXPECTED_TEST_METRICS.items():
        assert metrics[name] == pytest.approx(expected, abs=1e-12)
    assert int(labels.sum()) == 1_239


def test_individual_prediction_output(sample_record: dict) -> None:
    result = au.predict_customer(sample_record)
    assert set(result) == {
        "probability",
        "prediction",
        "predicted_label",
        "threshold",
        "model_name",
    }
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in {"Yes", "No"}
    assert result["predicted_label"] in {0, 1}
    assert result["threshold"] == 0.2
    assert result["model_name"] == "Random Forest"
    assert (result["prediction"] == "Yes") == (result["predicted_label"] == 1)


def test_batch_prediction_output_format(raw_sample: pd.DataFrame) -> None:
    predictions = au.predict_batch(raw_sample)
    assert len(predictions) == len(raw_sample)
    assert list(predictions.columns) == [
        "predicted_probability",
        "predicted_subscription",
        "predicted_label",
        "decision_threshold",
        "model_name",
    ]
    assert predictions["predicted_probability"].between(0, 1).all()
    assert predictions["predicted_subscription"].isin(["Yes", "No"]).all()
    assert predictions["predicted_label"].isin([0, 1]).all()
    assert (predictions["decision_threshold"] == 0.2).all()
    assert (predictions["model_name"] == "Random Forest").all()
    assert (
        predictions["predicted_label"].to_numpy()
        == (predictions["predicted_probability"].to_numpy() >= 0.2).astype(int)
    ).all()


def test_batch_prediction_without_duration(raw_sample: pd.DataFrame) -> None:
    without_duration = raw_sample.drop(columns=["duration"]).copy()
    predictions = au.predict_batch(without_duration)
    assert len(predictions) == len(without_duration)
    assert predictions["predicted_probability"].notna().all()


def test_missing_required_columns_rejected(raw_sample: pd.DataFrame) -> None:
    invalid = raw_sample.drop(columns=["job", "marital", "poutcome"])
    with pytest.raises(ValueError, match="Missing required columns"):
        au.validate_raw_input(invalid)


def test_incorrect_data_type_rejected(raw_sample: pd.DataFrame) -> None:
    invalid = raw_sample.copy()
    invalid["age"] = invalid["age"].astype(object)
    invalid.loc[invalid.index[0], "age"] = "not-a-number"
    with pytest.raises(ValueError, match="must contain numeric values"):
        au.validate_raw_input(invalid)


def test_invalid_binary_value_rejected(raw_sample: pd.DataFrame) -> None:
    invalid = raw_sample.copy()
    invalid["housing"] = invalid["housing"].astype(object)
    invalid.loc[invalid.index[0], "housing"] = "maybe"
    with pytest.raises(ValueError, match="yes/no"):
        au.validate_raw_input(invalid)


def test_blank_category_rejected(raw_sample: pd.DataFrame) -> None:
    invalid = raw_sample.copy()
    invalid.loc[invalid.index[0], "education"] = ""
    with pytest.raises(ValueError, match="blank"):
        au.validate_raw_input(invalid)


@pytest.mark.parametrize(
    "column,value",
    [
        ("age", 17),
        ("age", 121),
        ("day", 0),
        ("day", 32),
        ("campaign", 0),
        ("previous", -1),
        ("pdays", -2),
    ],
)
def test_out_of_range_fields_rejected(
    raw_sample: pd.DataFrame, column: str, value: int
) -> None:
    invalid = raw_sample.copy()
    invalid.loc[invalid.index[0], column] = value
    with pytest.raises(ValueError, match=column):
        au.validate_raw_input(invalid)


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="no customer rows"):
        au.validate_raw_input(pd.DataFrame())


def test_binary_numeric_input_is_accepted(raw_sample: pd.DataFrame) -> None:
    sample = raw_sample.copy()
    for column in ["default", "housing", "loan"]:
        sample[column] = np.where(sample[column] == "yes", 1, 0)
    cleaned = au.validate_raw_input(sample)
    assert set(cleaned["default"].unique()).issubset({"yes", "no"})
    predictions = au.predict_batch(sample)
    assert len(predictions) == len(sample)
