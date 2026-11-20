"""
Shared application utilities for the Phase 5 Bank Marketing Intelligence app.

This module deliberately uses the real Phase 2 feature engineering rules, the
real Phase 4 selected model artifact, and the real Phase 3 K-Means pipeline.
It does not retrain or replace any model.

Prediction inputs are pre-call customer and campaign fields. `duration` is
excluded from prediction and from the engineered model features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
KMEANS_PATH = PROJECT_ROOT / "models" / "kmeans.joblib"
CLUSTERING_METRICS_PATH = PROJECT_ROOT / "reports" / "phase3" / "clustering_metrics.csv"
CLUSTER_PROFILES_PATH = PROJECT_ROOT / "reports" / "phase3" / "cluster_profiles.csv"

TARGET_COLUMN = "y"
DURATION_COLUMN = "duration"

# Raw fields used by the Phase 2 engineering rules. `duration` is deliberately
# absent: it is only known after a call and is excluded from the pre-call model.
REQUIRED_RAW_COLUMNS = [
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
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]
NUMERIC_RAW_COLUMNS = ["age", "balance", "day", "campaign", "pdays", "previous"]
BINARY_RAW_COLUMNS = ["default", "housing", "loan"]
CATEGORICAL_RAW_COLUMNS = ["job", "marital", "education", "contact", "month", "poutcome"]

# Phase 2 feature engineering constants. Keeping these identical is important:
# the saved preprocessing pipeline and model were fitted on these definitions.
AGE_BINS = [17, 24, 34, 44, 54, 64, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
CONTACT_FREQUENCY_BINS = [0, 1, 3, 5, np.inf]
CONTACT_FREQUENCY_LABELS = ["1", "2-3", "4-5", "6+"]

ENGINEERED_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "previous",
    "pdays_since_previous_contact",
    "default",
    "housing",
    "loan",
    "has_previous_contact",
    "previous_campaign_success",
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "poutcome",
    "age_group",
    "contact_frequency_group",
]


# ---------------------------------------------------------------------------
# Artifact checks
# ---------------------------------------------------------------------------


def check_required_artifacts() -> None:
    """Fail clearly if the Phase 3/4 application artifacts are not present."""
    required = {
        "raw dataset": RAW_DATA_PATH,
        "Phase 4 best model": BEST_MODEL_PATH,
        "Phase 4 model metadata": MODEL_METADATA_PATH,
        "Phase 3 K-Means model": KMEANS_PATH,
        "Phase 3 clustering metrics": CLUSTERING_METRICS_PATH,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Required Phase 3/4 artifacts are missing. Phase 5 must use the "
            "real saved models and data; no substitute model will be created.\n"
            + "\n".join(missing)
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_raw_data() -> pd.DataFrame:
    """Read the original semicolon-separated UCI dataset without modifying it."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, sep=";")
    df.columns = [str(column).strip().lower() for column in df.columns]
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Raw dataset is missing the target column '{TARGET_COLUMN}'.")
    return df


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the real raw dataset and add descriptive, non-model columns."""
    df = _read_raw_data().copy()
    df["subscribed"] = df[TARGET_COLUMN].astype(str).str.lower().eq("yes").astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)
    df["campaign_group"] = pd.cut(
        df["campaign"],
        bins=CONTACT_FREQUENCY_BINS,
        labels=CONTACT_FREQUENCY_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)
    df["has_previous_contact"] = (df["pdays"] != -1).astype(int)
    df["previous_campaign_success"] = (df["poutcome"] == "success").astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_categorical_options() -> dict[str, list[str]]:
    """Observed categorical levels from the real dataset, for form controls."""
    df = load_raw_data()
    return {
        column: sorted(df[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_RAW_COLUMNS
    }


@st.cache_data(show_spinner=False)
def load_numeric_ranges() -> dict[str, tuple[float, float]]:
    """Observed numeric ranges from the real dataset, for sensible form limits."""
    df = load_raw_data()
    return {
        column: (float(df[column].min()), float(df[column].max()))
        for column in NUMERIC_RAW_COLUMNS
    }


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict[str, Any]:
    """Load the Phase 4 model metadata (selected model, threshold, metrics)."""
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {MODEL_METADATA_PATH}")
    with MODEL_METADATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_best_model_bundle() -> dict[str, Any]:
    """Load and validate the real Phase 4 selected model artifact."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Phase 4 best model not found: {BEST_MODEL_PATH}")
    artifact = joblib.load(BEST_MODEL_PATH)
    required_keys = {
        "model",
        "preprocessing_pipeline",
        "threshold",
        "model_name",
        "feature_names",
    }
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError(
            "The Phase 4 artifact does not have the expected model/pipeline/"
            "threshold structure."
        )
    if not hasattr(artifact["model"], "predict_proba"):
        raise ValueError("The selected Phase 4 model does not expose predict_proba.")
    if not hasattr(artifact["preprocessing_pipeline"], "transform"):
        raise ValueError("The saved preprocessing pipeline is not fitted.")
    if not 0 <= float(artifact["threshold"]) <= 1:
        raise ValueError("The saved decision threshold is outside [0, 1].")
    if len(artifact["feature_names"]) != artifact["model"].n_features_in_:
        raise ValueError("Saved feature names do not match the selected model.")
    if any(DURATION_COLUMN in str(name) for name in artifact["feature_names"]):
        raise ValueError("duration unexpectedly appears in the selected model features.")
    return artifact


@st.cache_resource(show_spinner=False)
def load_kmeans_bundle() -> Any:
    """Load and validate the real Phase 3 K-Means pipeline."""
    if not KMEANS_PATH.exists():
        raise FileNotFoundError(f"Phase 3 K-Means model not found: {KMEANS_PATH}")
    kmeans = joblib.load(KMEANS_PATH)
    if not hasattr(kmeans, "predict"):
        raise ValueError("The Phase 3 K-Means artifact is not a fitted estimator.")
    if not hasattr(kmeans, "named_steps") or "preprocessor" not in kmeans.named_steps:
        raise ValueError("The K-Means pipeline is missing its preprocessing step.")
    features = list(getattr(kmeans.named_steps["preprocessor"], "feature_names_in_", []))
    if not features:
        raise ValueError("The K-Means preprocessing step has no recorded input features.")
    if {TARGET_COLUMN, DURATION_COLUMN}.intersection(features):
        raise ValueError("K-Means inputs must not include y or duration.")
    return kmeans


def get_kmeans_features() -> list[str]:
    """Return the 12 engineered customer features expected by K-Means."""
    kmeans = load_kmeans_bundle()
    return [str(name) for name in kmeans.named_steps["preprocessor"].feature_names_in_]


# ---------------------------------------------------------------------------
# Validation and feature engineering
# ---------------------------------------------------------------------------


def _prepare_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise header names and perform basic structural checks."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The input contains no customer rows.")
    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names are not allowed: {duplicated}")
    return data


def validate_raw_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw customer input and return a cleaned copy.

    Missing required columns, non-numeric values, invalid binary values, and
    out-of-range campaign fields produce a friendly ValueError for the UI.
    """
    data = _prepare_input_frame(frame)
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". The prediction form expects raw pre-call customer fields."
        )

    for column in NUMERIC_RAW_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = converted.isna()
        if invalid.any():
            examples = data.loc[invalid, column].head(3).astype(str).tolist()
            raise ValueError(
                f"Column '{column}' must contain numeric values. "
                f"Invalid examples: {examples}"
            )
        data[column] = converted

    for column in BINARY_RAW_COLUMNS:
        normalised = data[column].astype(str).str.strip().str.lower()
        mapped = normalised.map(
            {"yes": "yes", "no": "no", "1": "yes", "0": "no", "true": "yes", "false": "no"}
        )
        invalid = mapped.isna()
        if invalid.any():
            examples = data.loc[invalid, column].head(3).astype(str).tolist()
            raise ValueError(
                f"Column '{column}' must be yes/no (or 1/0). "
                f"Invalid examples: {examples}"
            )
        data[column] = mapped

    for column in CATEGORICAL_RAW_COLUMNS:
        missing_values = data[column].isna() | data[column].astype(str).str.strip().eq("")
        if missing_values.any():
            raise ValueError(f"Column '{column}' contains blank category values.")

    checks = {
        "age": (data["age"].between(18, 120), "age must be between 18 and 120"),
        "day": (data["day"].between(1, 31), "day must be between 1 and 31"),
        "campaign": (data["campaign"] >= 1, "campaign must be at least 1"),
        "previous": (data["previous"] >= 0, "previous must be at least 0"),
        "pdays": (data["pdays"] >= -1, "pdays must be -1 or a non-negative day count"),
    }
    for column, (valid_mask, message) in checks.items():
        if not bool(valid_mask.all()):
            raise ValueError(f"Invalid values in '{column}': {message}.")
    return data


def find_unseen_categories(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return categorical values not present in the real training dataset."""
    raw = load_raw_data()
    unseen: dict[str, list[str]] = {}
    for column in CATEGORICAL_RAW_COLUMNS:
        allowed = set(raw[column].dropna().astype(str).unique())
        observed = set(frame[column].astype(str).unique())
        extra = sorted(observed - allowed)
        if extra:
            unseen[column] = extra
    return unseen


def build_engineered_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact Phase 2 pre-call feature engineering rules.

    This mirrors `src/phase2_preprocessing.py:engineer_features` but does not
    require a `duration` column. Duration is always dropped if present.
    """
    data = _prepare_input_frame(raw)
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Missing required fields for feature engineering: " + ", ".join(missing))

    for column in NUMERIC_RAW_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    for column in BINARY_RAW_COLUMNS:
        normalised = data[column].astype(str).str.strip().str.lower()
        mapped = normalised.map(
            {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
        )
        if mapped.isna().any():
            raise ValueError(f"Unexpected binary values in '{column}'.")
        data[column] = mapped.astype("int8")

    data["has_previous_contact"] = (data["pdays"] != -1).astype("int8")
    data["previous_campaign_success"] = (
        data["poutcome"].astype(str).str.lower() == "success"
    ).astype("int8")

    data["age_group"] = pd.cut(
        data["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)

    data["contact_frequency_group"] = pd.cut(
        data["campaign"],
        bins=CONTACT_FREQUENCY_BINS,
        labels=CONTACT_FREQUENCY_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)

    data["pdays_since_previous_contact"] = (
        data["pdays"].where(data["pdays"] != -1, 0).astype("int16")
    )
    data = data.drop(columns=["pdays"])
    if DURATION_COLUMN in data.columns:
        data = data.drop(columns=[DURATION_COLUMN])

    missing_engineered = [
        column for column in ENGINEERED_FEATURES if column not in data.columns
    ]
    if missing_engineered:
        raise ValueError(
            "Feature engineering did not produce required fields: "
            + ", ".join(missing_engineered)
        )
    if data.isna().any().any():
        bad_columns = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Feature engineering produced missing values: {bad_columns}")
    return data


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_batch(raw: pd.DataFrame) -> pd.DataFrame:
    """Predict subscription probability for a batch of raw customer records."""
    artifact = load_best_model_bundle()
    validated = validate_raw_input(raw)
    engineered = build_engineered_features(validated)

    try:
        matrix = artifact["preprocessing_pipeline"].transform(engineered)
    except Exception as exc:  # noqa: BLE001 - surface a helpful UI error
        raise ValueError(f"The saved preprocessing pipeline could not transform the input: {exc}") from exc

    matrix = np.asarray(matrix)
    expected_features = len(artifact["feature_names"])
    if matrix.ndim != 2 or matrix.shape[1] != expected_features:
        raise ValueError(
            "The saved preprocessing pipeline produced an unexpected number of "
            f"features: {matrix.shape[1] if matrix.ndim == 2 else 'unknown'}; "
            f"expected {expected_features}."
        )

    model_input = pd.DataFrame(
        matrix,
        columns=artifact["feature_names"],
        index=validated.index,
    )
    probabilities = artifact["model"].predict_proba(model_input)[:, 1]
    threshold = float(artifact["threshold"])
    labels = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "predicted_probability": probabilities,
            "predicted_subscription": np.where(labels == 1, "Yes", "No"),
            "predicted_label": labels,
            "decision_threshold": threshold,
            "model_name": artifact["model_name"],
        },
        index=validated.index,
    )


def predict_customer(record: dict[str, Any]) -> dict[str, Any]:
    """Predict one customer, returning the operational decision and probability."""
    if not isinstance(record, dict):
        raise ValueError("A customer record must be provided as a dictionary.")
    result = predict_batch(pd.DataFrame([record])).iloc[0]
    return {
        "probability": float(result["predicted_probability"]),
        "prediction": str(result["predicted_subscription"]),
        "predicted_label": int(result["predicted_label"]),
        "threshold": float(result["decision_threshold"]),
        "model_name": str(result["model_name"]),
    }


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_training_cluster_data() -> pd.DataFrame:
    """Assign the real K-Means clusters to the real Phase 2 training split.

    The split is reproduced with the same 80/20 stratified seed used in Phase 2.
    Subscription labels are retained only for post-hoc profile interpretation.
    """
    raw = _read_raw_data()
    y_raw = raw[TARGET_COLUMN].astype(str).str.lower().eq("yes").astype(int)
    X_raw = raw.drop(columns=[TARGET_COLUMN])
    X_train_raw, _, y_train, _ = train_test_split(
        X_raw,
        y_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_raw,
    )

    engineered = build_engineered_features(X_train_raw)
    features = get_kmeans_features()
    engineered = engineered.copy()
    engineered["subscribed"] = y_train.to_numpy()
    engineered["cluster"] = load_kmeans_bundle().predict(engineered[features])
    return engineered


@st.cache_data(show_spinner=False)
def load_cluster_profiles() -> pd.DataFrame:
    """Compute actual Phase 2 training-split statistics for each cluster."""
    df = load_training_cluster_data()
    overall_rate = float(df["subscribed"].mean() * 100)
    rows: list[dict[str, Any]] = []
    for cluster, group in df.groupby("cluster", sort=True):
        top_job = str(group["job"].mode().iat[0])
        top_education = str(group["education"].mode().iat[0])
        rows.append(
            {
                "cluster": int(cluster),
                "n_customers": int(len(group)),
                "share_pct": 100.0 * len(group) / len(df),
                "age_mean": float(group["age"].mean()),
                "balance_median": float(group["balance"].median()),
                "top_job": top_job,
                "top_job_pct": float((group["job"] == top_job).mean() * 100),
                "top_education": top_education,
                "pct_housing_loan": float(group["housing"].mean() * 100),
                "pct_previous_contact": float(group["has_previous_contact"].mean() * 100),
                "observed_subscription_rate_pct": float(group["subscribed"].mean() * 100),
            }
        )
    profiles = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    profiles["overall_subscription_rate_pct"] = overall_rate
    return profiles


@st.cache_data(show_spinner=False)
def load_clustering_metrics() -> pd.DataFrame:
    """Load the real Phase 3 k-selection metrics."""
    if not CLUSTERING_METRICS_PATH.exists():
        raise FileNotFoundError(f"Clustering metrics not found: {CLUSTERING_METRICS_PATH}")
    return pd.read_csv(CLUSTERING_METRICS_PATH)


@st.cache_data(show_spinner=False)
def build_cluster_projection(max_points: int = 7000) -> tuple[pd.DataFrame, np.ndarray]:
    """Project the actual K-Means feature space to 2D PCA for display only."""
    df = load_training_cluster_data()
    kmeans = load_kmeans_bundle()
    features = get_kmeans_features()
    matrix = np.asarray(kmeans.named_steps["preprocessor"].transform(df[features]))
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(matrix)

    plot_df = df[["cluster", "age", "balance", "job", "subscribed"]].copy()
    plot_df["pc1"] = coordinates[:, 0]
    plot_df["pc2"] = coordinates[:, 1]
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=42).copy()
    return plot_df, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Small reporting helpers used by the UI
# ---------------------------------------------------------------------------


def overall_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return actual customer and subscription summary values."""
    total = len(df)
    subscribers = int(df["subscribed"].sum()) if "subscribed" in df else 0
    return {
        "total": total,
        "subscribers": subscribers,
        "non_subscribers": total - subscribers,
        "subscription_rate": 100.0 * subscribers / total if total else 0.0,
    }


def subscription_rate_by(
    df: pd.DataFrame,
    column: str,
    min_customers: int = 1,
) -> pd.DataFrame:
    """Observed subscription rate and sample size by a categorical field."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' is not available.")
    summary = (
        df.groupby(column, dropna=False)
        .agg(
            n_customers=("subscribed", "size"),
            subscribers=("subscribed", "sum"),
            subscription_rate_pct=("subscribed", lambda values: 100.0 * values.mean()),
        )
        .reset_index()
    )
    summary = summary[summary["n_customers"] >= min_customers]
    return summary.sort_values("subscription_rate_pct", ascending=False).reset_index(drop=True)


def get_model_test_metrics() -> dict[str, Any]:
    """Return the real Phase 4 test metrics stored with the selected artifact."""
    metadata = load_model_metadata()
    return metadata.get("test_metrics_at_selected_threshold", {})
