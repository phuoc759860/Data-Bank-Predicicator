"""Streamlit page, form, filter, upload, and download tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "Home": PROJECT_ROOT / "app" / "Home.py",
    "Executive Dashboard": PROJECT_ROOT / "app" / "pages" / "1_Executive_Dashboard.py",
    "Customer Prediction": PROJECT_ROOT / "app" / "pages" / "2_Customer_Prediction.py",
    "Batch Prediction": PROJECT_ROOT / "app" / "pages" / "3_Batch_Prediction.py",
    "Customer Segmentation": PROJECT_ROOT / "app" / "pages" / "4_Customer_Segmentation.py",
    "Marketing Analytics": PROJECT_ROOT / "app" / "pages" / "5_Marketing_Analytics.py",
}


@pytest.mark.parametrize("name,path", PAGES.items(), ids=PAGES.keys())
def test_page_renders_without_exception(name: str, path: Path) -> None:
    at = AppTest.from_file(str(path), default_timeout=240).run()
    assert not at.exception, f"{name} raised: {[e.value for e in at.exception]}"
    assert len(at.markdown) > 0
    assert at.sidebar is not None


def test_executive_dashboard_filters_update_without_error() -> None:
    at = AppTest.from_file(str(PAGES["Executive Dashboard"]), default_timeout=240).run()
    assert not at.exception
    assert len(at.multiselect) == 4
    at.multiselect[0].set_value(["management"]).run()
    assert not at.exception
    assert len(at.dataframe) == 1


def test_prediction_form_submission() -> None:
    at = AppTest.from_file(str(PAGES["Customer Prediction"]), default_timeout=240).run()
    labels = [element.label for element in list(at.number_input) + list(at.selectbox)]
    assert not any("duration" in label.lower() for label in labels)
    assert len(at.button) == 1
    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) >= 2
    assert {metric.label for metric in at.metric} >= {
        "Predicted probability",
        "Decision threshold",
    }
    assert at.metric[1].value == "0.20"


def test_batch_upload_and_download_ui(raw_df: pd.DataFrame) -> None:
    content = raw_df.head(5).to_csv(index=False, sep=";").encode("utf-8")
    at = AppTest.from_file(str(PAGES["Batch Prediction"]), default_timeout=240).run()
    at.file_uploader[0].upload("customers.csv", content, mime_type="text/csv").run()
    assert not at.exception
    assert len(at.error) == 0
    assert len(at.dataframe) == 1
    assert at.dataframe[0].value.shape == (5, 22)
    assert len(at.download_button) == 1
    assert at.download_button[0].label == "Download predictions as CSV"


def test_batch_missing_columns_ui() -> None:
    content = b"age,job\n40,management\n"
    at = AppTest.from_file(str(PAGES["Batch Prediction"]), default_timeout=240).run()
    at.file_uploader[0].upload("invalid.csv", content, mime_type="text/csv").run()
    assert not at.exception
    assert len(at.error) >= 1
    assert "Missing required columns" in " ".join(error.value for error in at.error)
    assert len(at.download_button) == 0


def test_batch_incorrect_type_ui(raw_df: pd.DataFrame) -> None:
    invalid = raw_df.head(5).copy()
    invalid["age"] = invalid["age"].astype(object)
    invalid.loc[invalid.index[0], "age"] = "not-a-number"
    content = invalid.to_csv(index=False, sep=";").encode("utf-8")
    at = AppTest.from_file(str(PAGES["Batch Prediction"]), default_timeout=240).run()
    at.file_uploader[0].upload("invalid.csv", content, mime_type="text/csv").run()
    assert not at.exception
    assert len(at.error) >= 1
    assert "must contain numeric values" in " ".join(error.value for error in at.error)
    assert len(at.download_button) == 0
