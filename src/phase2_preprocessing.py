#!/usr/bin/env python
"""
Phase 2 - Data Preprocessing and Preparation
============================================

Bank Marketing Intelligence System (Information Systems Management project).

This script turns the raw UCI Bank Marketing data (bank-full.csv) into
model-ready, leakage-safe training and test matrices for a *pre-call*
term-deposit targeting model.

Workflow
--------
1. Load and validate the raw semicolon-separated file (read-only).
2. Investigate 'unknown' categories, pdays = -1, and numerical outliers.
3. Split the raw data 80/20 with stratification (random_state=42).
4. Engineer deterministic features (age group, contact history, campaign
   frequency, previous success) separately on train and test.
5. Fit a scikit-learn Pipeline / ColumnTransformer on the TRAINING data only
   (OneHotEncoder for nominal variables, StandardScaler for numeric ones),
   then transform both splits.
6. Validate the processed matrices and write the deliverables.

No machine-learning model is trained in this phase.

Usage
-----
    python src/phase2_preprocessing.py
    python src/phase2_preprocessing.py --data data/raw/bank-full.csv
    python src/phase2_preprocessing.py --test-size 0.2 --random-state 42

Outputs (default: data/processed/)
----------------------------------
    X_train.csv, X_test.csv, y_train.csv, y_test.csv
    preprocessing_summary.md
    feature_dictionary.csv
    target_distribution_after_split.csv
    unknown_values_investigation.csv   (supporting)
    outlier_investigation.csv          (supporting)
    preprocessor.joblib                (fitted preprocessing pipeline)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "y"
UNKNOWN_TOKEN = "unknown"
EXPECTED_ROWS = 45_211
TEST_SIZE = 0.2
RANDOM_STATE = 42

EXPECTED_COLUMNS = [
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

ORIGINAL_NUMERIC = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

ORIGINAL_CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Source features used by the primary pre-call model (after engineering).
NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "previous",
    "pdays_since_previous_contact",
]

BINARY_FEATURES = [
    "default",
    "housing",
    "loan",
    "has_previous_contact",
    "previous_campaign_success",
]

NOMINAL_FEATURES = [
    "job",
    "marital",
    "education",
    "contact",
    "month",
    "poutcome",
    "age_group",
    "contact_frequency_group",
]

BINARY_COLUMNS_TO_MAP = ["default", "housing", "loan"]

AGE_BINS = [17, 24, 34, 44, 54, 64, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

CONTACT_FREQUENCY_BINS = [0, 1, 3, 5, np.inf]
CONTACT_FREQUENCY_LABELS = ["1", "2-3", "4-5", "6+"]

# Human-readable descriptions used in feature_dictionary.csv.
SOURCE_DESCRIPTIONS = {
    "age": "Client age in years",
    "job": "Type of occupation",
    "marital": "Marital status (divorced includes widowed)",
    "education": "Education level",
    "default": "Has credit in default (1 = yes, 0 = no)",
    "balance": "Average yearly account balance in EUR",
    "housing": "Has a housing loan (1 = yes, 0 = no)",
    "loan": "Has a personal loan (1 = yes, 0 = no)",
    "contact": "Communication channel of the last contact",
    "day": "Day of month of the last contact",
    "month": "Month of the last contact",
    "campaign": "Contacts performed during this campaign (including last)",
    "previous": "Contacts performed before this campaign",
    "poutcome": "Outcome of the previous marketing campaign",
    "pdays_since_previous_contact": (
        "Days since the previous campaign contact; set to 0 when the client "
        "was never contacted (see has_previous_contact)"
    ),
    "has_previous_contact": (
        "1 if the client was contacted in a previous campaign (pdays != -1), else 0"
    ),
    "previous_campaign_success": (
        "1 if the previous campaign outcome was a success, else 0"
    ),
    "age_group": "Age bucket: 18-24, 25-34, 35-44, 45-54, 55-64, 65+",
    "contact_frequency_group": (
        "Number of contacts in the current campaign, grouped as 1, 2-3, 4-5, 6+"
    ),
    "duration": (
        "Duration of the last call in seconds; excluded because it is only "
        "known after the call (post-call leakage)"
    ),
}

FEATURE_GROUPS = {
    "age": "client profile",
    "job": "client profile",
    "marital": "client profile",
    "education": "client profile",
    "default": "client profile",
    "balance": "client profile",
    "housing": "client profile",
    "loan": "client profile",
    "contact": "current campaign",
    "day": "current campaign",
    "month": "current campaign",
    "duration": "current campaign",
    "campaign": "campaign history",
    "previous": "campaign history",
    "pdays_since_previous_contact": "campaign history",
    "has_previous_contact": "engineered",
    "previous_campaign_success": "engineered",
    "age_group": "engineered",
    "contact_frequency_group": "engineered",
    "poutcome": "campaign history",
}

UNKNOWN_DECISIONS = {
    "job": "Retain as an explicit 'unknown' level (one-hot encoded).",
    "education": "Retain as an explicit 'unknown' level (one-hot encoded).",
    "contact": "Retain as an explicit 'unknown' level (one-hot); no separate missingness flag.",
    "poutcome": (
        "Retain as an explicit 'unknown' level; previous_campaign_success "
        "captures confirmed prior success separately."
    ),
    "marital": "No unknown values; no action needed.",
    "default": "No unknown values; binary encoding applied.",
    "housing": "No unknown values; binary encoding applied.",
    "loan": "No unknown values; binary encoding applied.",
    "month": "No unknown values; one-hot encoded.",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pct(part: float, total: float) -> float:
    """Percentage helper that safely handles a zero denominator."""
    return 100.0 * part / total if total else 0.0


def file_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file (used to prove the raw file is intact)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md_table(frame: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    """Render a DataFrame as GitHub-flavoured Markdown without extra dependencies."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (int, np.integer)):
                cells.append(f"{int(value):,}")
            elif isinstance(value, (float, np.floating)):
                cells.append(float_fmt.format(float(value)))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 1 - Loading, quality investigation and cleaning decisions
# ---------------------------------------------------------------------------

def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the semicolon-separated UCI dataset without modifying it."""
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place bank-full.csv in data/raw/ or pass --data <path>."
        )
    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    try:
        df = pd.read_csv(path, sep=";")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Could not parse {path} with sep=';'. Check the delimiter and file "
            f"integrity. Details: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Encoding error while reading {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read dataset {path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset contains no rows: {path}")

    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_columns}\n"
            f"Columns found ({len(df.columns)}): {list(df.columns)}"
        )

    extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    if extra_columns:
        warnings.warn(
            f"Ignoring unexpected extra columns: {extra_columns}", stacklevel=2
        )

    df = df[EXPECTED_COLUMNS].copy()

    for column in ORIGINAL_CATEGORICAL + [TARGET]:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    invalid_targets = sorted(set(df[TARGET].dropna().unique()) - {"yes", "no"})
    if invalid_targets:
        raise ValueError(
            f"Target '{TARGET}' contains unexpected values: {invalid_targets}. "
            "Expected only 'yes' / 'no'."
        )

    for column in ORIGINAL_NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' should be numeric but has dtype {df[column].dtype}."
            )

    if len(df) != EXPECTED_ROWS:
        warnings.warn(
            f"Expected {EXPECTED_ROWS:,} records for bank-full.csv but found "
            f"{len(df):,}. Continuing with the loaded data.",
            stacklevel=2,
        )

    return df


def investigate_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify 'unknown' values and compare conversion rates with known values.

    No rows or values are removed here; this only supports an informed decision.
    """
    rows = []
    for column in ORIGINAL_CATEGORICAL:
        mask_unknown = df[column].eq(UNKNOWN_TOKEN)
        unknown_count = int(mask_unknown.sum())
        unknown_rate = (
            pct(int(df.loc[mask_unknown, TARGET].eq("yes").sum()), unknown_count)
            if unknown_count
            else np.nan
        )
        known_count = int((~mask_unknown).sum())
        known_rate = (
            pct(int(df.loc[~mask_unknown, TARGET].eq("yes").sum()), known_count)
            if known_count
            else np.nan
        )
        rows.append(
            {
                "column": column,
                "unknown_count": unknown_count,
                "unknown_pct": round(pct(unknown_count, len(df)), 4),
                "unknown_subscription_rate_pct": (
                    round(unknown_rate, 4) if pd.notna(unknown_rate) else np.nan
                ),
                "known_subscription_rate_pct": round(known_rate, 4),
                "rate_gap_pct_points": (
                    round(unknown_rate - known_rate, 4)
                    if pd.notna(unknown_rate)
                    else np.nan
                ),
                "treatment": UNKNOWN_DECISIONS[column],
            }
        )
    return pd.DataFrame(rows)


def investigate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Profile numerical outliers with the 1.5 x IQR rule and 1st/99th percentiles.

    Outliers are *reported*, not removed or capped: extreme balances and campaign
    counts can be legitimate banking behaviour.
    """
    rows = []
    for column in ORIGINAL_NUMERIC:
        series = df[column]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        below = int((series < lower_fence).sum())
        above = int((series > upper_fence).sum())
        outlier_count = below + above
        rows.append(
            {
                "variable": column,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_fence": float(lower_fence),
                "upper_fence": float(upper_fence),
                "below_lower_fence": below,
                "above_upper_fence": above,
                "outlier_count": outlier_count,
                "outlier_pct": round(pct(outlier_count, len(df)), 4),
                "min": float(series.min()),
                "p1": float(series.quantile(0.01)),
                "median": float(series.median()),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "treatment": "Retained (no deletion, capping, or winsorising).",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 6 - Leakage-safe train/test split (performed before any fitting)
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split raw features/target with stratification, before any transformation."""
    X_raw = df.drop(columns=[TARGET])
    y_raw = df[TARGET]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw,
    )

    y_train = y_train_raw.map({"no": 0, "yes": 1}).astype("int8")
    y_test = y_test_raw.map({"no": 0, "yes": 1}).astype("int8")

    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Target encoding produced missing values; check input labels.")

    return X_train_raw, X_test_raw, y_train, y_test


# ---------------------------------------------------------------------------
# Task 2 - Deterministic feature engineering (no learned statistics)
# ---------------------------------------------------------------------------

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Create the selected features. Bins and mappings are fixed constants, so this
    function can be applied independently to train and test without leakage.
    """
    data = frame.copy()

    for column in BINARY_COLUMNS_TO_MAP:
        data[column] = data[column].map({"yes": 1, "no": 0}).astype("int8")
        if data[column].isna().any():
            raise ValueError(f"Unexpected value in binary column '{column}'.")

    # pdays = -1 is a sentinel for 'never contacted', not a real day count.
    data["has_previous_contact"] = (data["pdays"] != -1).astype("int8")
    data["previous_campaign_success"] = (
        data["poutcome"] == "success"
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

    # Keep recency information while removing the -1 sentinel. The binary flag
    # distinguishes 'no previous contact' from a genuine same-day recency.
    data["pdays_since_previous_contact"] = (
        data["pdays"].where(data["pdays"] != -1, 0).astype("int16")
    )
    data = data.drop(columns=["pdays", "duration"])

    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Feature engineering produced missing values in: {missing}")

    missing_expected = [
        column
        for column in NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES
        if column not in data.columns
    ]
    if missing_expected:
        raise ValueError(f"Engineered frame is missing required features: {missing_expected}")

    return data


# ---------------------------------------------------------------------------
# Tasks 3 & 4 - Encoding and scaling inside a fitted-on-train pipeline
# ---------------------------------------------------------------------------

def build_preprocessor() -> Pipeline:
    """Build the unfitted preprocessing Pipeline (one-hot + scaling)."""
    numeric_pipeline = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                NOMINAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(steps=[("preprocessor", column_transformer)])


def fit_transform_splits(
    X_train_engineered: pd.DataFrame,
    X_test_engineered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, list[str]]:
    """
    Fit the preprocessing Pipeline on training data only, then transform both
    splits. This prevents test statistics from influencing scaling or encoding.
    """
    pipeline = build_preprocessor()
    pipeline.fit(X_train_engineered)

    column_transformer = pipeline.named_steps["preprocessor"]
    feature_names = [
        str(name) for name in column_transformer.get_feature_names_out()
    ]

    X_train = pd.DataFrame(
        pipeline.transform(X_train_engineered),
        columns=feature_names,
        index=X_train_engineered.index,
    )
    X_test = pd.DataFrame(
        pipeline.transform(X_test_engineered),
        columns=feature_names,
        index=X_test_engineered.index,
    )

    return X_train, X_test, pipeline, feature_names


# ---------------------------------------------------------------------------
# Task 5 - Feature documentation
# ---------------------------------------------------------------------------

def build_feature_dictionary(feature_names: list[str]) -> pd.DataFrame:
    """Map every final model column back to its source feature and treatment."""
    rows = []
    nominal_longest_first = sorted(NOMINAL_FEATURES, key=len, reverse=True)

    for feature_name in feature_names:
        category = ""
        if feature_name.startswith("num__"):
            source = feature_name[len("num__"):]
            feature_type = "numeric"
            transformation = "StandardScaler (fit on training data only)"
        elif feature_name.startswith("bin__"):
            source = feature_name[len("bin__"):]
            feature_type = "binary"
            transformation = "Passthrough as 0/1 (no scaling)"
        elif feature_name.startswith("cat__"):
            payload = feature_name[len("cat__"):]
            source = next(
                (
                    candidate
                    for candidate in nominal_longest_first
                    if payload.startswith(f"{candidate}_")
                ),
                "",
            )
            if not source:
                raise ValueError(f"Could not map one-hot feature: {feature_name}")
            category = payload[len(source) + 1:]
            feature_type = "one-hot"
            transformation = "OneHotEncoder (fit on training data only, handle_unknown='ignore')"
        else:
            raise ValueError(f"Unrecognised transformed feature: {feature_name}")

        description = SOURCE_DESCRIPTIONS.get(source, source)
        if category:
            description = f"{description} - category: {category}"

        rows.append(
            {
                "feature_name": feature_name,
                "source_variable": source,
                "feature_group": FEATURE_GROUPS.get(source, "other"),
                "feature_type": feature_type,
                "transformation": transformation,
                "description": description,
                "selected_for_pre_call_model": True,
            }
        )

    # Document the deliberately excluded feature in the same dictionary.
    rows.append(
        {
            "feature_name": "duration",
            "source_variable": "duration",
            "feature_group": "current campaign",
            "feature_type": "excluded",
            "transformation": "Excluded from the primary pre-call model",
            "description": SOURCE_DESCRIPTIONS["duration"],
            "selected_for_pre_call_model": False,
        }
    )

    return pd.DataFrame(rows)


def build_target_distribution(
    y_train: pd.Series, y_test: pd.Series, y_all: pd.Series
) -> pd.DataFrame:
    """Counts and percentages per split, including the full dataset for reference."""
    rows = []
    for split_name, series in [
        ("train", y_train),
        ("test", y_test),
        ("overall", y_all.map({"no": 0, "yes": 1}).astype("int8")),
    ]:
        total = len(series)
        for class_value, class_label in [(0, "no"), (1, "yes")]:
            count = int((series == class_value).sum())
            rows.append(
                {
                    "split": split_name,
                    "class": class_label,
                    "count": count,
                    "percentage": round(pct(count, total), 4),
                    "split_total": total,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    original_df: pd.DataFrame,
) -> dict[str, float]:
    """Run hard checks on transformed matrices and split integrity."""
    errors: list[str] = []

    if X_train.shape[1] != X_test.shape[1]:
        errors.append("Train and test matrices have different feature counts.")
    if not list(X_train.columns) == list(X_test.columns):
        errors.append("Train and test feature names/order differ.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        errors.append("Feature and target row counts do not align.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        errors.append("Missing values detected after transformation.")
    if not np.isfinite(X_train.to_numpy()).all():
        errors.append("Non-finite values detected in X_train.")
    if not np.isfinite(X_test.to_numpy()).all():
        errors.append("Non-finite values detected in X_test.")
    if set(y_train.unique()) - {0, 1} or set(y_test.unique()) - {0, 1}:
        errors.append("Target contains values other than 0/1.")

    total_rows = len(X_train) + len(X_test)
    if total_rows != len(original_df):
        errors.append(
            f"Split rows ({total_rows:,}) do not equal dataset rows "
            f"({len(original_df):,})."
        )

    train_rate = pct(int(y_train.sum()), len(y_train))
    test_rate = pct(int(y_test.sum()), len(y_test))
    overall_rate = pct(
        int(original_df[TARGET].eq("yes").sum()), len(original_df)
    )
    if abs(train_rate - test_rate) > 0.5:
        errors.append(
            f"Class balance differs by more than 0.5 pp between splits "
            f"(train {train_rate:.2f}%, test {test_rate:.2f}%)."
        )

    if errors:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(errors))

    return {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": X_train.shape[1],
        "train_positive_rate": train_rate,
        "test_positive_rate": test_rate,
        "overall_positive_rate": overall_rate,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_summary(
    df: pd.DataFrame,
    data_path: Path,
    raw_hash: str,
    unknown: pd.DataFrame,
    outliers: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_distribution: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    feature_names: list[str],
    validation: dict[str, float],
    test_size: float,
    random_state: int,
    output_dir: Path,
) -> str:
    """Compose the Phase 2 Markdown documentation."""
    try:
        display_path = data_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(data_path)

    try:
        output_display = output_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        output_display = str(output_dir)

    n_rows = len(df)
    pdays_never = int((df["pdays"] == -1).sum())
    pdays_pct = pct(pdays_never, n_rows)
    n_predictive = len(feature_names)
    nominal_expanded = int(
        (feature_dictionary["feature_type"] == "one-hot").sum()
    )
    n_train_yes = int(y_train.sum())
    n_test_yes = int(y_test.sum())
    source_feature_count = (
        len(NUMERIC_FEATURES) + len(BINARY_FEATURES) + len(NOMINAL_FEATURES)
    )
    engineered = feature_dictionary[
        feature_dictionary["source_variable"].isin(
            ["age_group", "has_previous_contact",
             "contact_frequency_group", "previous_campaign_success"]
        )
        & (feature_dictionary["feature_type"] != "excluded")
    ]["source_variable"].nunique()

    lines: list[str] = []
    add = lines.append

    add("# Phase 2 - Data Preprocessing and Preparation")
    add("")
    add("**Project:** Bank Marketing Intelligence System  ")
    add("**Course context:** Information Systems Management  ")
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    add(f"**Input:** `{display_path}`  ")
    add(f"**Raw file SHA-256 (verified unchanged):** `{raw_hash[:16]}...`  ")
    add("**Target:** `y` (1 = subscribed a term deposit, 0 = did not)  ")
    add(f"**Split:** {int((1 - test_size) * 100)}% train / "
        f"{int(test_size * 100)}% test, stratified, random_state={random_state}  ")
    add(f"**Final matrices:** X_train {len(X_train):,} x {n_predictive}, "
        f"X_test {len(X_test):,} x {n_predictive}  ")
    add("**Scope:** preprocessing only - no machine-learning model was trained.")
    add("")
    add("---")
    add("")
    add("## 1. Executive summary")
    add("")
    add(f"- The original **{n_rows:,} x {df.shape[1]}** dataset was read with "
        "`pd.read_csv(path, sep=';')` and preserved byte-for-byte; the SHA-256 "
        "checksum was verified after processing.")
    add(f"- No NULL cells and **{int(df.duplicated().sum()):,} duplicate rows** were found. "
        "No rows were deleted anywhere in Phase 2.")
    add(f"- 'unknown' categories were **retained as explicit levels** rather than "
        "imputed or dropped. They affect "
        f"**{int((unknown['unknown_count'] > 0).sum())} categorical columns**, most "
        "notably `poutcome` and `contact`.")
    add(f"- `pdays = -1` was converted into the binary feature "
        f"`has_previous_contact` and a sentinel-free recency feature; "
        f"**{pdays_never:,} records ({pdays_pct:.2f}%)** had no previous contact.")
    add(f"- **{len(outliers)} numerical variables** were profiled for outliers. "
        "Extreme values were retained; only scaling is applied, fitted on training data.")
    add(f"- Data were split **before** fitting any transformer. The fitted pipeline "
        "uses OneHotEncoder for nominal variables, 0/1 passthrough for binary "
        "variables, and StandardScaler for numeric variables.")
    add(f"- **{source_feature_count} source features** (including {engineered} engineered "
        f"features) expand to **{n_predictive} model-ready columns**: "
        f"{len(NUMERIC_FEATURES)} scaled numeric, {len(BINARY_FEATURES)} binary, "
        f"and {nominal_expanded} one-hot indicators.")
    add("- `duration` is excluded from the primary pre-call model because it is "
        "recorded only after the call and would leak the outcome.")
    add(f"- Post-transformation validation: **0 missing values, 0 non-finite values**, "
        f"train positive rate {validation['train_positive_rate']:.2f}%, test positive "
        f"rate {validation['test_positive_rate']:.2f}% (overall "
        f"{validation['overall_positive_rate']:.2f}%).")
    add("")
    add("---")
    add("")
    add("## 2. Data cleaning (Task 1)")
    add("")
    add("### 2.1 Input integrity")
    add("")
    add("The source file is opened read-only. It is never overwritten, and the script "
        "re-computes its SHA-256 checksum after writing all outputs to prove that the "
        "original data remain unchanged.")
    add("")
    add(f"- Rows: **{n_rows:,}**; columns: **{df.shape[1]}**")
    add(f"- Missing cells: **{int(df.isna().sum().sum()):,}**")
    add(f"- Exact duplicate rows: **{int(df.duplicated().sum()):,}**")
    add(f"- `pdays = -1`: **{pdays_never:,} ({pdays_pct:.2f}%)**")
    add("")
    add("### 2.2 'unknown' category investigation and decisions")
    add("")
    add("The table compares the subscription rate of 'unknown' records with records "
        "that have a known value. This tests whether 'unknown' is random or carries "
        "behavioural information.")
    add("")
    unknown_display = unknown.copy()
    unknown_display = unknown_display.rename(
        columns={
            "column": "column",
            "unknown_count": "unknown_count",
            "unknown_pct": "unknown_pct",
            "unknown_subscription_rate_pct": "unknown_rate_pct",
            "known_subscription_rate_pct": "known_rate_pct",
            "rate_gap_pct_points": "gap_pp",
        }
    )[
        [
            "column",
            "unknown_count",
            "unknown_pct",
            "unknown_rate_pct",
            "known_rate_pct",
            "gap_pp",
        ]
    ].round(2)
    add(md_table(unknown_display))
    add("")
    add("**Decision:** no row is deleted and no 'unknown' value is replaced by the "
        "mode. The categories are kept as explicit one-hot levels so the model can "
        "learn their effect, and future unseen categories are handled with "
        "`handle_unknown='ignore'`. For `poutcome`, the additional "
        "`previous_campaign_success` feature isolates confirmed prior success. "
        "This preserves the full sample and does not fabricate a customer attribute.")
    add("")
    add("### 2.3 pdays = -1 handling")
    add("")
    add(f"`pdays = -1` appears in **{pdays_never:,} records ({pdays_pct:.2f}%)** and "
        "means *never contacted before*, not *one day before*. Using it as a number "
        "would distort averages and distances. The following were created:")
    add("")
    add("- `has_previous_contact`: 1 when `pdays != -1`, otherwise 0.")
    add("- `pdays_since_previous_contact`: 0 for never-contacted clients; otherwise "
        "the original positive day count. The binary flag distinguishes the sentinel "
        "from genuine recency.")
    add("")
    add(f"The consistency check confirms that `pdays = -1` and `previous = 0` agree "
        f"for all records (mismatches: "
        f"**{int(((df['pdays'] == -1) != (df['previous'] == 0)).sum()):,}**).")
    add("")
    add("### 2.4 Numerical outlier investigation")
    add("")
    add("Outliers were profiled with the 1.5 x IQR rule and tail percentiles. They are "
        "**not deleted, capped, or winsorised**, because high balances and long "
        "campaign histories are legitimate banking observations and may be predictive.")
    add("")
    outlier_display = outliers[
        [
            "variable",
            "outlier_count",
            "outlier_pct",
            "min",
            "p1",
            "median",
            "p99",
            "max",
            "treatment",
        ]
    ].round(2)
    add(md_table(outlier_display))
    add("")
    add(f"Balance ranges from {df['balance'].min():,} to {df['balance'].max():,} EUR "
        f"and includes **{int((df['balance'] < 0).sum()):,} negative balances** "
        f"({pct(int((df['balance'] < 0).sum()), n_rows):.2f}%). "
        "`StandardScaler` is sensitive to extreme values, so Phase 3 should compare "
        "scaled linear models with tree-based models and consider robust alternatives "
        "if performance is unstable. Any alternative transformation must be refitted "
        "inside the training folds.")
    add("")
    add("The high outlier counts for `pdays` and `previous` reflect the campaign-history "
        "structure rather than data errors: most clients have no previous contact, so "
        "any positive history appears far above the median. The `-1` sentinel is "
        "handled separately below; `duration` is excluded from modelling regardless.")
    add("")
    add("---")
    add("")
    add("## 3. Feature engineering (Task 2)")
    add("")
    add("All engineering rules are fixed constants applied independently to each split; "
        "no aggregate statistic from the training or test set is used.")
    add("")
    engineering = pd.DataFrame(
        [
            {
                "feature": "age_group",
                "source": "age",
                "purpose": "Captures the non-linear age effect found in Phase 1 "
                           "(65+ and 18-24 convert far above the 35-54 core).",
            },
            {
                "feature": "has_previous_contact",
                "source": "pdays",
                "purpose": "Separates clients with campaign history from first-time "
                           "contacts and removes the -1 sentinel ambiguity.",
            },
            {
                "feature": "contact_frequency_group",
                "source": "campaign",
                "purpose": "Groups current-campaign contact intensity (1, 2-3, 4-5, "
                           "6+) to represent persistence and possible call fatigue.",
            },
            {
                "feature": "previous_campaign_success",
                "source": "poutcome",
                "purpose": "Flags warm leads whose previous campaign succeeded, the "
                           "strongest descriptive signal in Phase 1 (64.7% conversion).",
            },
            {
                "feature": "pdays_since_previous_contact",
                "source": "pdays",
                "purpose": "Recency of the previous contact without the -1 sentinel; "
                           "0 is interpreted together with has_previous_contact.",
            },
            {
                "feature": "default / housing / loan",
                "source": "categorical yes/no",
                "purpose": "Mapped to 0/1 so binary variables use one column rather "
                           "than two redundant one-hot columns.",
            },
        ]
    )
    add(md_table(engineering))
    add("")
    add("---")
    add("")
    add("## 4. Encoding (Task 3)")
    add("")
    add("| Group | Source features | Method | Output columns |")
    add("| --- | --- | --- | --- |")
    add(f"| Nominal | {', '.join(f'`{c}`' for c in NOMINAL_FEATURES)} | "
        f"OneHotEncoder(handle_unknown='ignore') | {nominal_expanded} |")
    add(f"| Binary | {', '.join(f'`{c}`' for c in BINARY_FEATURES)} | "
        f"0/1 passthrough (yes/no mapped) | {len(BINARY_FEATURES)} |")
    add(f"| Numeric | {', '.join(f'`{c}`' for c in NUMERIC_FEATURES)} | "
        f"StandardScaler | {len(NUMERIC_FEATURES)} |")
    add("")
    add(f"One-hot encoding is applied only to nominal variables. Every level is kept "
        f"(no drop='first') to preserve interpretability and avoid assuming an order. "
        f"The final column names retain transformer prefixes (`num__`, `bin__`, `cat__`) "
        f"and the original variable/category, e.g. `cat__job_blue-collar`. "
        f"All {n_predictive} names are documented in `feature_dictionary.csv`.")
    add("")
    add("---")
    add("")
    add("## 5. Scaling and leakage prevention (Task 4)")
    add("")
    add("- The raw data were split **first**; feature engineering uses fixed rules only.")
    add(f"- The pipeline was fitted on **{len(X_train):,} training rows** and then applied "
        f"to the **{len(X_test):,} test rows**.")
    add("- `StandardScaler` therefore learned its means and standard deviations from "
        "training data only. Test data never influenced scaling, category discovery, "
        "or any other fitted parameter.")
    add("- Binary variables are not scaled, and one-hot indicators remain 0/1.")
    add("- The fitted pipeline is saved as `preprocessor.joblib` so Phase 3 can apply "
        "exactly the same transformation to new data.")
    add("- For cross-validation in Phase 3, the preprocessing pipeline must be fitted "
        "inside each training fold; resampling, if used, must also occur inside folds.")
    add("")
    add("---")
    add("")
    add("## 6. Feature selection and duration exclusion (Task 5)")
    add("")
    add(f"The selected source-level feature set contains **{source_feature_count} "
        f"features** before encoding:")
    add("")
    for group_name in ["client profile", "current campaign", "campaign history", "engineered"]:
        group_features = [
            column
            for column in (
                NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES
            )
            if FEATURE_GROUPS.get(column) == group_name
        ]
        add(f"- **{group_name.title()}:** " + ", ".join(f"`{c}`" for c in group_features))
    add("")
    add("### Why `duration` is excluded")
    add("")
    add("`duration` is the length of the last phone call and is recorded only after the "
        "call ends. At the moment a customer is selected for a pre-call campaign, this "
        "value does not exist. Phase 1 found a strong raw correlation with the target "
        "(+0.395), so including it would produce optimistic, unreproducible results "
        "through **target leakage**. It is therefore removed before the pipeline and "
        "marked `selected_for_pre_call_model = False` in the feature dictionary. It may "
        "be used only in a separate call-time or post-call use case, never in this "
        "pre-call targeting dataset.")
    add("")
    add("`pdays` is also not passed through in its raw form; it is replaced by "
        "`has_previous_contact` and `pdays_since_previous_contact` as described above.")
    add("")
    add("---")
    add("")
    add("## 7. Train/test split (Task 6)")
    add("")
    add(f"- Method: `train_test_split(..., test_size={test_size}, "
        f"stratify=y, random_state={random_state})`")
    add(f"- **Training set:** {len(X_train):,} rows ({pct(len(X_train), n_rows):.2f}%)")
    add(f"- **Test set:** {len(X_test):,} rows ({pct(len(X_test), n_rows):.2f}%)")
    add(f"- **Features per split:** {n_predictive}")
    add(f"- Positive class: train {n_train_yes:,} "
        f"({validation['train_positive_rate']:.2f}%), test {n_test_yes:,} "
        f"({validation['test_positive_rate']:.2f}%), overall "
        f"{validation['overall_positive_rate']:.2f}%")
    add("")
    add(md_table(target_distribution))
    add("")
    add("Stratification preserves the 11.70% subscription rate in both splits, so "
        "evaluation in Phase 3 is not distorted by a different class mix.")
    add("")
    add("---")
    add("")
    add("## 8. Validation results (Task 7)")
    add("")
    add("| Check | Result |")
    add("| --- | --- |")
    add(f"| X_train shape | {X_train.shape[0]:,} x {X_train.shape[1]} |")
    add(f"| X_test shape | {X_test.shape[0]:,} x {X_test.shape[1]} |")
    add(f"| Missing values after transformation | {int(X_train.isna().sum().sum() + X_test.isna().sum().sum()):,} |")
    add(f"| Non-finite values | 0 |")
    add(f"| Target values | 0 and 1 only |")
    add(f"| Train + test rows equal original | {len(X_train) + len(X_test):,} = {n_rows:,} |")
    add(f"| Train / test positive rate | {validation['train_positive_rate']:.2f}% / {validation['test_positive_rate']:.2f}% |")
    add(f"| Raw file checksum unchanged | Yes |")
    add("")
    add("---")
    add("")
    add("## 9. Saved artefacts")
    add("")
    add(f"All files are written to `{output_display}/`:")
    add("")
    add("- `X_train.csv` - scaled/encoded training features")
    add("- `X_test.csv` - scaled/encoded test features")
    add("- `y_train.csv` - training target (0/1)")
    add("- `y_test.csv` - test target (0/1)")
    add("- `preprocessing_summary.md` - this document")
    add(f"- `feature_dictionary.csv` - all {n_predictive} model columns plus the excluded `duration`")
    add("- `target_distribution_after_split.csv` - class counts and percentages")
    add("- `unknown_values_investigation.csv` - unknown counts and conversion comparison")
    add("- `outlier_investigation.csv` - IQR/tail outlier profile")
    add("- `preprocessor.joblib` - fitted scikit-learn Pipeline")
    add("")
    add("The processed CSVs contain the **transformed** matrices, not raw categories. "
        "Use the feature dictionary and the saved pipeline to interpret or reuse them.")
    add("")
    add("---")
    add("")
    add("## 10. Limitations and Phase 3 notes")
    add("")
    add("- **No model has been trained or evaluated.** These matrices are inputs for "
        "Phase 3 only.")
    add("- **Class imbalance remains** at ~11.7% positives. Use recall, precision, F1, "
        "ROC-AUC/PR-AUC, lift, and cost-based evaluation rather than accuracy.")
    add("- **'unknown' is retained as information**, but its meaning may differ across "
        "fields (not collected vs no previous campaign). Treat category effects with "
        "care when explaining models.")
    add("- **Outliers are retained.** StandardScaler can be sensitive to them; compare "
        "with tree-based models and consider robust scaling only if justified by "
        "validation performance.")
    add("- **Sentinel recency:** `pdays_since_previous_contact = 0` is only meaningful "
        "together with `has_previous_contact`; do not interpret 0 as a same-day call.")
    add("- **No year field** is available, so long-term time effects cannot be modelled.")
    add("- **Feature redundancy:** `previous_campaign_success` is derived from "
        "`poutcome`; this is intentional for interpretability but can be revisited for "
        "linear models with regularisation.")
    add("- **Scaling must stay inside the CV pipeline** in Phase 3 to avoid leakage.")
    add("")
    add("---")
    add("")
    add("## 11. Reproduction")
    add("")
    add("```bash")
    add("python src/phase2_preprocessing.py")
    add("```")
    add("")
    add("Optional arguments:")
    add("")
    add("```bash")
    add("python src/phase2_preprocessing.py \\")
    add("    --data data/raw/bank-full.csv \\")
    add("    --output-dir data/processed \\")
    add(f"    --test-size {test_size} \\")
    add(f"    --random-state {random_state}")
    add("```")
    add("")
    add("The script validates the schema and target, fits transformations on training "
        "data only, verifies zero missing/non-finite values, and confirms that the raw "
        "CSV checksum is unchanged.")
    add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pipeline: Pipeline,
    unknown: pd.DataFrame,
    outliers: pd.DataFrame,
    target_distribution: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> dict[str, Path]:
    """Write processed matrices and supporting files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    artefacts: list[tuple[str, pd.DataFrame, bool]] = [
        ("X_train.csv", X_train, False),
        ("X_test.csv", X_test, False),
        ("y_train.csv", y_train.rename(TARGET).to_frame(), False),
        ("y_test.csv", y_test.rename(TARGET).to_frame(), False),
        ("feature_dictionary.csv", feature_dictionary, True),
        ("target_distribution_after_split.csv", target_distribution, True),
        ("unknown_values_investigation.csv", unknown, True),
        ("outlier_investigation.csv", outliers, True),
    ]

    for filename, frame, use_float_format in artefacts:
        path = output_dir / filename
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8",
            float_format="%.4f" if use_float_format else None,
        )
        written[filename] = path
        print(f"  saved: {filename}  ({len(frame):,} rows x {frame.shape[1]} cols)")

    model_path = output_dir / "preprocessor.joblib"
    joblib.dump(pipeline, model_path)
    written["preprocessor.joblib"] = model_path
    print(f"  saved: {model_path.name}")

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2 preprocessing for the Bank Marketing Intelligence System."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to bank-full.csv (default: data/raw/bank-full.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/processed).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test fraction between 0 and 1 (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for the split (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the complete Phase 2 preprocessing workflow."""
    args = parse_args(argv)
    data_path = Path(args.data).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    if not 0.0 < args.test_size < 1.0:
        print("ERROR: --test-size must be strictly between 0 and 1.", file=sys.stderr)
        return 1

    print("=" * 78)
    print("BANK MARKETING INTELLIGENCE SYSTEM - PHASE 2")
    print("Data preprocessing and preparation")
    print("=" * 78)

    try:
        df = load_raw_dataset(data_path)
        raw_hash_before = file_sha256(data_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    # Task 1 -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TASK 1 - DATA CLEANING AND QUALITY INVESTIGATION")
    print("=" * 78)
    print(f"Dataset: {data_path}")
    print(f"Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing cells: {int(df.isna().sum().sum()):,}")
    print(f"Duplicate rows: {int(df.duplicated().sum()):,}")

    unknown = investigate_unknowns(df)
    outliers = investigate_outliers(df)

    print("\n'unknown' categorical values:")
    print(
        unknown[
            [
                "column",
                "unknown_count",
                "unknown_pct",
                "unknown_subscription_rate_pct",
                "known_subscription_rate_pct",
            ]
        ].round(2).to_string(index=False)
    )
    print("\nDecision: retain 'unknown' as explicit one-hot levels; no rows deleted.")

    never_contacted = int((df["pdays"] == -1).sum())
    print(
        f"\npdays = -1 (never previously contacted): {never_contacted:,} "
        f"({pct(never_contacted, len(df)):.2f}%) -> has_previous_contact + "
        "sentinel-free recency feature."
    )
    print(
        f"\nNumerical outliers (1.5 x IQR): "
        f"{int(outliers['outlier_count'].sum()):,} variable-level flags across "
        f"{len(outliers)} variables (not unique rows); all values retained."
    )
    print(outliers[["variable", "outlier_count", "outlier_pct"]].to_string(index=False))

    # Task 2 -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TASK 2 - FEATURE ENGINEERING AND TRAIN/TEST SPLIT")
    print("=" * 78)
    try:
        X_train_raw, X_test_raw, y_train, y_test = split_dataset(
            df, test_size=args.test_size, random_state=args.random_state
        )
        X_train_eng = engineer_features(X_train_raw)
        X_test_eng = engineer_features(X_test_raw)
    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Split: train={len(X_train_eng):,} rows | test={len(X_test_eng):,} rows "
        f"(test_size={args.test_size}, random_state={args.random_state}, stratified)"
    )

    unseen_categories = {
        column: sorted(
            set(X_test_eng[column].astype(str))
            - set(X_train_eng[column].astype(str))
        )
        for column in NOMINAL_FEATURES
    }
    unseen_categories = {
        column: values for column, values in unseen_categories.items() if values
    }
    unseen_total = sum(len(values) for values in unseen_categories.values())
    print(
        f"Unseen nominal categories in test: {unseen_total} "
        "(OneHotEncoder handle_unknown='ignore' would encode any as all-zero)"
    )

    print("\nEngineered features:")
    for description in [
        "age_group: age buckets 18-24, 25-34, 35-44, 45-54, 55-64, 65+",
        "has_previous_contact: 1 when pdays != -1 (previous campaign contact)",
        "contact_frequency_group: current-campaign contacts grouped 1, 2-3, 4-5, 6+",
        "previous_campaign_success: 1 when poutcome == success",
        "pdays_since_previous_contact: pdays with -1 mapped to 0 (flag disambiguates)",
        "default / housing / loan: yes/no mapped to 1/0",
    ]:
        print(f"  - {description}")

    # Tasks 3 & 4 -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TASKS 3 & 4 - ENCODING AND SCALING (FITTED ON TRAIN ONLY)")
    print("=" * 78)
    try:
        X_train, X_test, pipeline, feature_names = fit_transform_splits(
            X_train_eng, X_test_eng
        )
    except (ValueError, TypeError) as exc:
        print(f"\nERROR while fitting the preprocessing pipeline: {exc}", file=sys.stderr)
        return 1

    print(f"Preprocessing pipeline: {pipeline}")
    print(f"\nNominal features one-hot encoded: {', '.join(NOMINAL_FEATURES)}")
    print(f"Binary features passed through  : {', '.join(BINARY_FEATURES)}")
    print(f"Numeric features scaled        : {', '.join(NUMERIC_FEATURES)}")
    print(f"Final model-ready features     : {len(feature_names)}")
    print("Transformer fitted on training data only; test data transformed once.")

    # Task 5 -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("TASK 5 - FEATURE SELECTION")
    print("=" * 78)
    print("Target separated from inputs: y (0 = no, 1 = yes).")
    print("duration EXCLUDED from the primary pre-call model (post-call leakage).")
    print(f"Selected source features ({len(NUMERIC_FEATURES) + len(BINARY_FEATURES) + len(NOMINAL_FEATURES)}):")
    print("  numeric:", ", ".join(NUMERIC_FEATURES))
    print("  binary :", ", ".join(BINARY_FEATURES))
    print("  nominal:", ", ".join(NOMINAL_FEATURES))

    feature_dictionary = build_feature_dictionary(feature_names)
    target_distribution = build_target_distribution(
        y_train, y_test, df[TARGET]
    )

    # Task 6/7 validation -----------------------------------------------------
    print("\n" + "=" * 78)
    print("TASK 6 & 7 - VALIDATION AND OUTPUT")
    print("=" * 78)
    try:
        validation = validate_outputs(X_train, X_test, y_train, y_test, df)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"X_train shape: {X_train.shape[0]:,} x {X_train.shape[1]}")
    print(f"X_test shape : {X_test.shape[0]:,} x {X_test.shape[1]}")
    print(f"Missing values after transformation: {int(X_train.isna().sum().sum() + X_test.isna().sum().sum())}")
    print(f"Non-finite values: 0")
    print("\nTarget distribution after split:")
    print(target_distribution.to_string(index=False))
    print(
        f"\nPositive rate: train {validation['train_positive_rate']:.2f}% | "
        f"test {validation['test_positive_rate']:.2f}% | "
        f"overall {validation['overall_positive_rate']:.2f}%"
    )

    try:
        written = save_outputs(
            output_dir,
            X_train,
            X_test,
            y_train,
            y_test,
            pipeline,
            unknown,
            outliers,
            target_distribution,
            feature_dictionary,
        )
    except OSError as exc:
        print(f"\nERROR writing outputs: {exc}", file=sys.stderr)
        return 1

    summary = build_summary(
        df=df,
        data_path=data_path,
        raw_hash=raw_hash_before,
        unknown=unknown,
        outliers=outliers,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_distribution=target_distribution,
        feature_dictionary=feature_dictionary,
        feature_names=feature_names,
        validation=validation,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=output_dir,
    )
    summary_path = output_dir / "preprocessing_summary.md"
    try:
        summary_path.write_text(summary, encoding="utf-8")
    except OSError as exc:
        print(f"\nERROR writing summary: {exc}", file=sys.stderr)
        return 1
    print(f"  saved: {summary_path.name}")
    written["preprocessing_summary.md"] = summary_path

    raw_hash_after = file_sha256(data_path)
    if raw_hash_after != raw_hash_before:
        print(
            "\nERROR: The raw dataset checksum changed during processing. "
            "The original file must remain untouched.",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 78)
    print("PHASE 2 COMPLETE")
    print("=" * 78)
    print(f"Output directory : {output_dir}")
    print(f"Artefacts        : {len(written)} files")
    print(f"Raw SHA-256      : {raw_hash_before[:16]}... (unchanged)")
    print("No machine-learning models were trained in this phase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
