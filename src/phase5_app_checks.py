"""Reproducible Phase 5 smoke checks for the real artifacts and Streamlit pages.

Run from the project root:

    python src/phase5_app_checks.py

The checks use the real Phase 4 model, the real Phase 2 processed test set, and
the real Phase 3 K-Means artifact. They do not retrain or replace any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import f1_score
from streamlit.testing.v1 import AppTest

from src import app_utils as au
from src import phase2_preprocessing as p2

PAGES = {
    "Home": str(PROJECT_ROOT / "app" / "Home.py"),
    "Executive Dashboard": str(PROJECT_ROOT / "app" / "pages" / "1_Executive_Dashboard.py"),
    "Customer Prediction": str(PROJECT_ROOT / "app" / "pages" / "2_Customer_Prediction.py"),
    "Batch Prediction": str(PROJECT_ROOT / "app" / "pages" / "3_Batch_Prediction.py"),
    "Customer Segmentation": str(PROJECT_ROOT / "app" / "pages" / "4_Customer_Segmentation.py"),
    "Marketing Analytics": str(PROJECT_ROOT / "app" / "pages" / "5_Marketing_Analytics.py"),
}


def check_artifacts() -> None:
    au.check_required_artifacts()
    metadata = au.load_model_metadata()
    assert metadata["selected_model"] == "Random Forest"
    assert float(metadata["threshold"]) == 0.2
    assert metadata["duration_included"] is False
    assert metadata["test_set_used_for_selection"] is False
    print("PASS artifacts and metadata")


def check_model_reload_and_metrics() -> None:
    artifact = au.load_best_model_bundle()
    X_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_test.csv")
    y_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_test.csv")["y"]
    probabilities = artifact["model"].predict_proba(X_test)[:, 1]
    labels = (probabilities >= float(artifact["threshold"])).astype(int)
    score = f1_score(y_test, labels)
    assert abs(score - 0.49020461471484544) < 1e-10
    assert int(labels.sum()) == 1239
    print(f"PASS model reload and prediction (F1={score:.4f}, positives={int(labels.sum()):,})")


def check_feature_parity() -> None:
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "bank-full.csv", sep=";").head(100)
    ours = au.build_engineered_features(raw)
    expected = p2.engineer_features(raw)
    common = [column for column in expected.columns if column in ours.columns]
    pd.testing.assert_frame_equal(
        ours[common].reset_index(drop=True),
        expected[common].reset_index(drop=True),
        check_dtype=False,
    )
    assert "duration" not in ours.columns
    print("PASS Phase 2 feature-engineering parity; duration absent")


def check_individual_prediction() -> None:
    record = {
        "age": 40,
        "job": "management",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1500,
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }
    result = au.predict_customer(record)
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in {"Yes", "No"}
    assert result["threshold"] == 0.2
    print(
        f"PASS individual prediction ({result['prediction']}, "
        f"probability={result['probability']:.4f})"
    )


def check_batch_prediction() -> None:
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "bank-full.csv", sep=";").head(5)
    predictions = au.predict_batch(raw)
    assert len(predictions) == 5
    for column in [
        "predicted_probability",
        "predicted_subscription",
        "predicted_label",
        "decision_threshold",
        "model_name",
    ]:
        assert column in predictions.columns
    assert predictions["predicted_probability"].between(0, 1).all()
    print("PASS batch prediction (5 rows)")


def check_invalid_inputs() -> None:
    try:
        au.validate_raw_input(pd.DataFrame({"age": [40]}))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing columns were not rejected.")
    bad = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "bank-full.csv", sep=";").head(2).copy()
    bad["age"] = bad["age"].astype(object)
    bad.loc[0, "age"] = "not-a-number"
    try:
        au.validate_raw_input(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-numeric values were not rejected.")
    print("PASS invalid batch input handling")


def check_segmentation() -> None:
    clusters = au.load_training_cluster_data()
    profiles = au.load_cluster_profiles()
    counts = clusters["cluster"].value_counts().sort_index().to_dict()
    assert counts == {0: 11833, 1: 890, 2: 5696, 3: 17749}
    assert len(profiles) == 4
    assert not {"y", "duration"}.intersection(au.get_kmeans_features())
    print(f"PASS segmentation ({counts})")


def check_pages() -> None:
    for name, path in PAGES.items():
        at = AppTest.from_file(path, default_timeout=240).run()
        assert not at.exception, f"{name} raised: {[e.value for e in at.exception]}"
        print(f"PASS page: {name}")


def check_prediction_form_and_batch_ui() -> None:
    at = AppTest.from_file(PAGES["Customer Prediction"], default_timeout=240).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) >= 2
    assert at.metric[1].value == "0.20"
    print("PASS individual prediction form submission")

    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "bank-full.csv", sep=";").head(5)
    valid_content = raw.to_csv(index=False, sep=";").encode("utf-8")
    at = AppTest.from_file(PAGES["Batch Prediction"], default_timeout=240).run()
    at.file_uploader[0].upload("customers.csv", valid_content, mime_type="text/csv").run()
    assert not at.exception
    assert len(at.error) == 0
    assert len(at.dataframe) == 1
    assert len(at.download_button) == 1
    print("PASS batch upload and download UI")

    invalid_content = b"age,job\n40,management\n"
    at = AppTest.from_file(PAGES["Batch Prediction"], default_timeout=240).run()
    at.file_uploader[0].upload("invalid.csv", invalid_content, mime_type="text/csv").run()
    assert not at.exception
    assert len(at.error) > 0
    assert len(at.download_button) == 0
    print("PASS batch invalid-CSV error UI")


def main() -> int:
    print("=" * 78)
    print("PHASE 5 APPLICATION CHECKS")
    print("=" * 78)
    checks = [
        check_artifacts,
        check_model_reload_and_metrics,
        check_feature_parity,
        check_individual_prediction,
        check_batch_prediction,
        check_invalid_inputs,
        check_segmentation,
        check_pages,
        check_prediction_form_and_batch_ui,
    ]
    for check in checks:
        check()
    print("=" * 78)
    print("ALL PHASE 5 CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
