"""Shared pytest fixtures for Phase 6 integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import app_utils as au


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Original semicolon-separated dataset, read without modification."""
    return pd.read_csv(PROJECT_ROOT / "data" / "raw" / "bank-full.csv", sep=";")


@pytest.fixture(scope="session")
def app_data() -> pd.DataFrame:
    """Dataset enriched with the descriptive columns used by the dashboard."""
    return au.load_raw_data()


@pytest.fixture(scope="session")
def artifact() -> dict:
    return au.load_best_model_bundle()


@pytest.fixture(scope="session")
def metadata() -> dict:
    return au.load_model_metadata()


@pytest.fixture
def raw_sample(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.head(10).copy()


@pytest.fixture
def sample_record() -> dict:
    return {
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
