#!/usr/bin/env python
"""
Phase 3 - Data Mining Model Development
=======================================

Bank Marketing Intelligence System (Information Systems Management project).

This script trains, tunes, evaluates, and persists three models:

1. Decision Tree classifier   - interpretable pre-call subscription model
2. Random Forest classifier   - ensemble pre-call subscription model
3. K-Means customer segmentation - independent unsupervised task

It consumes the Phase 2 artefacts:
    data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
    data/processed/preprocessor.joblib
    data/processed/feature_dictionary.csv

Key methodological safeguards
-----------------------------
- The same stratified 80/20 split from Phase 2 is reused for both classifiers.
- Hyperparameters are selected with 5-fold StratifiedKFold CV on TRAIN only.
- The test set is evaluated once per final model and never used for tuning.
- Phase 2 already fitted encoding/scaling on training data; it is reused, not
  refitted, for classification.
- K-Means excludes y and duration. It is fitted on the training rows only,
  using its own scaling/encoding pipeline, and uses y only for post-hoc
  interpretation of observed subscription rates.
- All random seeds are fixed at 42.

No Phase 4/5 work is performed here.

Usage
-----
    python src/phase3_models.py
    python src/phase3_models.py --processed-dir data/processed --models-dir models
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "phase3"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"

TARGET = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
POSITIVE_LABEL = 1
CLASS_NAMES = ["no", "yes"]

# Clustering uses stable customer/profile attributes and prior campaign history.
# Current-campaign execution fields (contact, day, month, campaign) and duration
# are deliberately excluded so segments describe customers, not campaign tactics.
CLUSTER_NUMERIC = ["age", "balance", "previous"]
CLUSTER_BINARY = [
    "default",
    "housing",
    "loan",
    "has_previous_contact",
    "previous_campaign_success",
]
CLUSTER_NOMINAL = ["job", "marital", "education", "poutcome"]
CLUSTER_FEATURES = CLUSTER_NUMERIC + CLUSTER_BINARY + CLUSTER_NOMINAL

K_RANGE = list(range(2, 11))
MIN_CLUSTER_SHARE_PCT = 2.0
SILHOUETTE_SAMPLE_SIZE = 5_000
PCA_SAMPLE_SIZE = 8_000

FEATURE_IMPORTANCE_TOP_N = 15
RULE_TOP_N = 10

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pct(part: float, total: float) -> float:
    """Percentage helper with a safe zero denominator."""
    return 100.0 * part / total if total else 0.0


def md_table(frame: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    """Render a DataFrame as GitHub-flavoured Markdown."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
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


def set_plot_style() -> None:
    """Apply a clean, consistent visual style for report charts."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="deep",
        rc={
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E5E5E5",
            "legend.framealpha": 0.9,
        },
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure to disk and close it."""
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved chart: {path.name}")


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with a clear error if the Phase 2 artefact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run Phase 2 first: python src/phase2_preprocessing.py"
        )
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def load_phase2_inputs(
    processed_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    Pipeline,
]:
    """Load the Phase 2 matrices, target vectors, dictionary, and pipeline."""
    required = {
        "X_train": processed_dir / "X_train.csv",
        "X_test": processed_dir / "X_test.csv",
        "y_train": processed_dir / "y_train.csv",
        "y_test": processed_dir / "y_test.csv",
        "feature_dictionary": processed_dir / "feature_dictionary.csv",
        "preprocessor": processed_dir / "preprocessor.joblib",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Phase 2 artefacts: " + ", ".join(missing) + "\n"
            "Run: python src/phase2_preprocessing.py"
        )

    X_train = load_csv(required["X_train"])
    X_test = load_csv(required["X_test"])
    feature_dictionary = load_csv(required["feature_dictionary"])

    try:
        y_train_frame = load_csv(required["y_train"])
        y_test_frame = load_csv(required["y_test"])
    except ValueError as exc:
        raise ValueError(f"Could not read target files: {exc}") from exc

    if TARGET not in y_train_frame.columns or TARGET not in y_test_frame.columns:
        raise ValueError("Target CSV files must contain a 'y' column.")
    if len(X_train) != len(y_train_frame) or len(X_test) != len(y_test_frame):
        raise ValueError("Feature and target row counts do not align.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test feature columns/order differ.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Phase 2 features contain missing values.")
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Phase 2 features contain non-finite values.")
    if set(y_train_frame[TARGET].unique()) - {0, 1}:
        raise ValueError("y_train must contain only 0/1.")
    if set(y_test_frame[TARGET].unique()) - {0, 1}:
        raise ValueError("y_test must contain only 0/1.")

    try:
        preprocessing_pipeline = joblib.load(required["preprocessor"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load Phase 2 pipeline: {exc}") from exc
    if not hasattr(preprocessing_pipeline, "transform"):
        raise ValueError("Loaded Phase 2 preprocessor is not a fitted transformer.")

    return (
        X_train,
        X_test,
        y_train_frame[TARGET].astype("int8"),
        y_test_frame[TARGET].astype("int8"),
        feature_dictionary,
        preprocessing_pipeline,
    )


def load_clustering_raw_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Recreate the exact Phase 2 split on raw data for K-Means.

    The saved target vector is used to prove that the reproduced training rows
    are identical to the Phase 2 split before clustering features are built.
    Using y here is for split verification only; y is never a clustering input.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}\n"
            "K-Means needs raw customer attributes for profiling."
        )

    raw = pd.read_csv(RAW_DATA_PATH, sep=";")
    if len(raw) != len(X_train) + len(X_test):
        raise ValueError("Raw dataset row count does not match Phase 2 outputs.")

    X_raw = raw.drop(columns=[TARGET])
    y_raw = raw[TARGET]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    y_train_reproduced = y_train_raw.map({"no": 0, "yes": 1}).astype("int8")
    if len(y_train_reproduced) != len(y_train) or not np.array_equal(
        y_train_reproduced.to_numpy(), y_train.to_numpy()
    ):
        raise RuntimeError(
            "Could not reproduce the Phase 2 split for clustering. "
            "Verify random_state/test_size and rerun Phase 2 if needed."
        )

    return X_train_raw, X_test_raw, y_train_raw, y_test_raw


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def build_cv() -> StratifiedKFold:
    """Stratified 5-fold CV with a fixed seed."""
    return StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


def tune_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """Tune tree depth and leaf-size controls with stratified CV (F1)."""
    param_grid = {
        "max_depth": [3, 4, 5, 6, 8, 10, 12],
        "min_samples_leaf": [25, 50, 100, 200],
        "min_samples_split": [2, 50, 100],
    }
    estimator = DecisionTreeClassifier(
        criterion="gini",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=build_cv(),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=0,
    )
    print(f"Decision Tree: fitting {len(param_grid['max_depth']) * len(param_grid['min_samples_leaf']) * len(param_grid['min_samples_split'])} "
          f"candidates x {CV_FOLDS} folds...")
    search.fit(X_train, y_train)
    print(f"  best CV F1: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")
    return search


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """Tune n_estimators / depth / leaf size with stratified CV (F1)."""
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 14, None],
        "min_samples_leaf": [5, 10, 20, 30],
    }
    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=build_cv(),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=0,
    )
    print(f"Random Forest: fitting {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_leaf'])} "
          f"candidates x {CV_FOLDS} folds...")
    search.fit(X_train, y_train)
    print(f"  best CV F1: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")
    return search


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_score: float,
    best_params: dict,
) -> dict[str, object]:
    """Compute test metrics once, including positive-class diagnostics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive": int((y_pred == 1).sum()),
        "predicted_positive_pct": pct(int((y_pred == 1).sum()), len(y_pred)),
        "cv_f1_mean": float(cv_score),
        "best_params": json.dumps(best_params, sort_keys=True),
    }
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
        ),
    }


def plot_confusion_matrix(
    cm: np.ndarray, model_name: str, path: Path
) -> None:
    """Plot an annotated confusion matrix for the positive-class analysis."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=0.8,
        linecolor="white",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(f"{model_name}: Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.text(
        0.01,
        -0.02,
        "Rows = actual, columns = predicted. Positive class = subscribed (yes).",
        fontsize=8,
        color="#666666",
    )
    save_figure(fig, path)


def plot_roc_curves(
    evaluations: dict[str, dict[str, object]],
    y_test: pd.Series,
    path: Path,
) -> None:
    """Overlay test ROC curves for both classifiers."""
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for (name, evaluation), color in zip(
        evaluations.items(), ["#4C72B0", "#DD8452"]
    ):
        fpr, tpr, _ = roc_curve(y_test, evaluation["y_proba"])
        auc_value = evaluation["metrics"]["roc_auc"]
        ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=color,
            label=f"{name} (AUC = {auc_value:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#999999",
            label="Random classifier")
    ax.set_title("ROC Curves - Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    save_figure(fig, path)


def build_feature_importance_frame(
    models: dict[str, object], feature_names: list[str]
) -> pd.DataFrame:
    """Combine decision-tree and random-forest impurity importances."""
    frames = []
    for model_name, model in models.items():
        importance = np.asarray(model.feature_importances_, dtype=float)
        frame = pd.DataFrame(
            {
                "model": model_name,
                "feature": feature_names,
                "importance": importance,
            }
        )
        frame["rank"] = frame["importance"].rank(
            ascending=False, method="first"
        ).astype(int)
        frames.append(frame.sort_values("importance", ascending=False))
    return pd.concat(frames, ignore_index=True)


def plot_feature_importance(
    importance: pd.DataFrame, path: Path, top_n: int = FEATURE_IMPORTANCE_TOP_N
) -> None:
    """Plot the top feature importances for both tree models side by side."""
    models = importance["model"].unique().tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7))
    if len(models) == 1:
        axes = [axes]

    colors = {"Decision Tree": "#4C72B0", "Random Forest": "#DD8452"}
    for ax, model_name in zip(axes, models):
        subset = (
            importance[importance["model"] == model_name]
            .sort_values("importance", ascending=False)
            .head(top_n)
            .sort_values("importance")
        )
        ax.barh(
            subset["feature"],
            subset["importance"],
            color=colors.get(model_name, "#55A868"),
            edgecolor="white",
        )
        ax.set_title(f"{model_name}: Top {top_n} Feature Importances")
        ax.set_xlabel("Impurity-based importance")
        ax.set_ylabel("Feature")
        max_value = max(float(subset["importance"].max()), 1e-9)
        for index, value in enumerate(subset["importance"]):
            ax.text(
                value + max_value * 0.012,
                index,
                f"{value:.4f}",
                va="center",
                fontsize=8,
                color="#333333",
            )
        ax.set_xlim(0, max_value * 1.22)
    fig.suptitle(
        "Feature Importance Comparison - Pre-Call Classification Models",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# Decision tree interpretation
# ---------------------------------------------------------------------------

def _human_condition(
    feature_name: str,
    threshold: float,
    goes_left: bool,
    metadata: pd.DataFrame,
    scaler_stats: dict[str, tuple[float, float]],
) -> str:
    """Convert a fitted-tree split into a readable customer condition."""
    if feature_name not in metadata.index:
        return feature_name

    row = metadata.loc[feature_name]
    source = str(row["source_variable"])
    feature_type = str(row["feature_type"])

    if feature_type == "numeric":
        mean, scale = scaler_stats[source]
        raw_threshold = threshold * scale + mean
        operator = "<=" if goes_left else ">"
        return f"{source} {operator} {raw_threshold:.2f}"

    if feature_type == "binary":
        value = 0 if goes_left else 1
        return f"{source} = {value}"

    if feature_type == "one-hot":
        prefix = f"cat__{source}_"
        category = feature_name[len(prefix):] if feature_name.startswith(prefix) else "?"
        operator = "!=" if goes_left else "="
        return f"{source} {operator} {category}"

    return feature_name


_NUMERIC_CONDITION = re.compile(
    r"^(?P<feature>[A-Za-z0-9_]+) (?P<operator><=|>) "
    r"(?P<value>-?\d+(?:\.\d+)?)$"
)


def _simplify_conditions(conditions: list[str]) -> list[str]:
    """
    Remove redundant numeric bounds from a root-to-leaf path.

    A tree path can contain repeated splits on the same feature (for example
    ``pdays <= 177.5`` followed by ``pdays <= 98.5``). For presentation, the
    tightest lower and upper bound is sufficient; the exact path remains in
    the exported text tree.
    """
    seen: set[str] = set()
    output: list[str | None] = []
    numeric_bounds: dict[str, dict[str, float | int | None]] = {}

    for condition in conditions:
        if condition in seen:
            continue
        seen.add(condition)

        match = _NUMERIC_CONDITION.match(condition)
        if not match:
            output.append(condition)
            continue

        feature = match.group("feature")
        operator = match.group("operator")
        value = float(match.group("value"))

        if feature not in numeric_bounds:
            output.append(None)
            numeric_bounds[feature] = {
                "index": len(output) - 1,
                "lower": None,
                "upper": None,
            }
        bounds = numeric_bounds[feature]

        if operator == "<=":
            current = bounds["upper"]
            bounds["upper"] = value if current is None else min(float(current), value)
        else:
            current = bounds["lower"]
            bounds["lower"] = value if current is None else max(float(current), value)

    for feature, bounds in numeric_bounds.items():
        parts = []
        if bounds["lower"] is not None:
            parts.append(f"{feature} > {float(bounds['lower']):.2f}")
        if bounds["upper"] is not None:
            parts.append(f"{feature} <= {float(bounds['upper']):.2f}")
        output[int(bounds["index"])] = " AND ".join(parts)

    return [condition for condition in output if condition]


def _human_feature_label(feature_name: str, metadata: pd.DataFrame) -> str:
    """Short readable label for tree diagrams."""
    if feature_name not in metadata.index:
        return feature_name

    row = metadata.loc[feature_name]
    source = str(row["source_variable"])
    feature_type = str(row["feature_type"])

    if feature_type == "one-hot":
        prefix = f"cat__{source}_"
        category = feature_name[len(prefix):] if feature_name.startswith(prefix) else "?"
        return f"{source}={category}"
    return source


def extract_decision_rules(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    metadata: pd.DataFrame,
    scaler_stats: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Traverse the fitted tree and export readable leaf-level decision rules.

    Counts are computed from the actual training rows routed to each leaf.
    This avoids relying on ``tree_.value``, which stores class-weighted
    proportions when ``class_weight='balanced'`` is used.
    """
    tree = model.tree_
    total_positive = int(y_train.sum()) if int(y_train.sum()) else 1

    leaf_ids = model.apply(X_train)
    assignment = pd.DataFrame(
        {"leaf": leaf_ids, "target": y_train.to_numpy()}
    )
    counts_by_leaf = assignment.groupby("leaf")["target"].agg(
        samples="size", subscribers="sum"
    )
    counts_by_leaf["non_subscribers"] = (
        counts_by_leaf["samples"] - counts_by_leaf["subscribers"]
    )

    rules: list[dict[str, object]] = []

    def recurse(node: int, conditions: list[str]) -> None:
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])

        if left == right:  # leaf
            if node in counts_by_leaf.index:
                row = counts_by_leaf.loc[node]
                samples = int(row["samples"])
                subscribers = int(row["subscribers"])
                non_subscribers = int(row["non_subscribers"])
            else:
                samples = subscribers = non_subscribers = 0

            precision = pct(subscribers, samples)
            recall = pct(subscribers, total_positive)
            simplified = _simplify_conditions(conditions)
            rules.append(
                {
                    "rule": " AND ".join(simplified) if simplified else "All customers",
                    "samples": samples,
                    "non_subscribers": non_subscribers,
                    "subscribers": subscribers,
                    "precision_pct": precision,
                    "recall_of_all_subscribers_pct": recall,
                }
            )
            return

        feature = feature_names[int(tree.feature[node])]
        threshold = float(tree.threshold[node])

        left_condition = _human_condition(
            feature, threshold, True, metadata, scaler_stats
        )
        right_condition = _human_condition(
            feature, threshold, False, metadata, scaler_stats
        )
        recurse(left, conditions + [left_condition])
        recurse(right, conditions + [right_condition])

    recurse(0, [])

    frame = pd.DataFrame(rules)
    frame = frame.sort_values(
        ["subscribers", "precision_pct"], ascending=[False, False]
    ).reset_index(drop=True)
    frame[["precision_pct", "recall_of_all_subscribers_pct"]] = frame[
        ["precision_pct", "recall_of_all_subscribers_pct"]
    ].round(2)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    metadata: pd.DataFrame,
    path: Path,
    display_depth: int = 3,
) -> None:
    """
    Visualise the top levels of the fitted tree at a readable size.

    The selected tree can have >100 leaves; plotting all of them produces an
    unreadable image. The top ``display_depth`` levels are shown here, while
    the complete tree is exported to decision_tree_report.txt and all leaf
    rules to decision_rules.csv.
    """
    depth = int(model.get_depth())
    leaves = int(model.get_n_leaves())
    shown_depth = min(display_depth, depth)
    labels = [_human_feature_label(name, metadata) for name in feature_names]
    fig, ax = plt.subplots(figsize=(34, 14))
    plot_tree(
        model,
        feature_names=labels,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        fontsize=11,
        max_depth=shown_depth,
        ax=ax,
    )
    ax.set_title(
        f"Decision Tree Structure - Top {shown_depth} of {depth} Levels "
        f"({leaves} leaves in full tree, class_weight='balanced')",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.01,
        "Colours reflect the majority class per node. Because class weights are "
        "balanced, tree 'value' shows weighted class proportions rather than raw "
        "counts; full rules with actual counts are in decision_rules.csv and "
        "decision_tree_report.txt.",
        fontsize=8,
        color="#666666",
    )
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# K-Means segmentation
# ---------------------------------------------------------------------------

def build_cluster_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create customer-segmentation features; y and duration are never included."""
    frame = raw.copy()

    for column in ["default", "housing", "loan"]:
        frame[column] = frame[column].map({"yes": 1, "no": 0}).astype("int8")
        if frame[column].isna().any():
            raise ValueError(f"Unexpected binary value in '{column}'.")

    frame["has_previous_contact"] = (frame["pdays"] != -1).astype("int8")
    frame["previous_campaign_success"] = (
        frame["poutcome"] == "success"
    ).astype("int8")

    features = frame[CLUSTER_FEATURES].copy()
    if features.isna().any().any():
        raise ValueError("Cluster feature frame contains missing values.")
    if TARGET in features.columns or "duration" in features.columns:
        raise AssertionError("Target/duration leaked into clustering features.")
    return features


def build_cluster_preprocessor() -> ColumnTransformer:
    """Scaling and one-hot encoding for the independent clustering task."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CLUSTER_NUMERIC),
            ("bin", "passthrough", CLUSTER_BINARY),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CLUSTER_NOMINAL,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def evaluate_cluster_counts(
    Z_train: np.ndarray,
    sample_index: np.ndarray,
) -> tuple[pd.DataFrame, KMeans]:
    """Compare K-Means solutions from 2 to 10 clusters."""
    rows = []
    fitted_models: dict[int, KMeans] = {}

    for k in K_RANGE:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(Z_train)
        counts = np.bincount(labels, minlength=k)
        silhouette = float(
            silhouette_score(
                Z_train[sample_index],
                labels[sample_index],
                random_state=RANDOM_STATE,
            )
        )
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": silhouette,
                "min_cluster_count": int(counts.min()),
                "max_cluster_count": int(counts.max()),
                "min_cluster_share_pct": pct(int(counts.min()), len(labels)),
                "cluster_sizes": ", ".join(str(int(value)) for value in counts),
            }
        )
        fitted_models[k] = model

    metrics = pd.DataFrame(rows)

    valid = metrics[
        metrics["min_cluster_share_pct"] >= MIN_CLUSTER_SHARE_PCT
    ]
    if valid.empty:
        valid = metrics
    best_k = int(
        valid.sort_values(["silhouette_score", "k"], ascending=[False, True])
        .iloc[0]["k"]
    )
    return metrics, fitted_models[best_k]


def plot_elbow(metrics: pd.DataFrame, best_k: int, path: Path) -> None:
    """Elbow chart of inertia against cluster count."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        metrics["k"],
        metrics["inertia"],
        marker="o",
        linewidth=2,
        markersize=7,
        color="#4C72B0",
    )
    ax.axvline(
        best_k,
        color="#C44E52",
        linestyle="--",
        linewidth=1.4,
        label=f"Selected k = {best_k}",
    )
    ax.set_title("K-Means Elbow Method (Training Data)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster sum of squares (inertia)")
    ax.set_xticks(list(metrics["k"]))
    ax.legend()
    save_figure(fig, path)


def plot_silhouette(metrics: pd.DataFrame, best_k: int, path: Path) -> None:
    """Silhouette score comparison across candidate cluster counts."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        "#C44E52" if k == best_k else "#4C72B0" for k in metrics["k"]
    ]
    bars = ax.bar(
        metrics["k"].astype(str),
        metrics["silhouette_score"],
        color=colors,
        width=0.65,
    )
    for bar, score in zip(bars, metrics["silhouette_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Silhouette Score by Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score (training sample)")
    ax.set_ylim(0, max(float(metrics["silhouette_score"].max()) * 1.2, 0.05))
    ax.text(
        0.98,
        0.95,
        f"Selected k = {best_k} (highlighted)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#666666",
    )
    save_figure(fig, path)


def plot_clusters_pca(
    Z_train: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    test_Z: np.ndarray,
    test_labels: np.ndarray,
    path: Path,
) -> None:
    """Two-dimensional PCA view of training clusters and test assignments."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    train_pca = pca.fit_transform(Z_train)
    centroids_pca = pca.transform(centroids)
    test_pca = pca.transform(test_Z)

    rng = np.random.RandomState(RANDOM_STATE)
    sample_size = min(PCA_SAMPLE_SIZE, len(train_pca))
    sample_index = rng.choice(len(train_pca), size=sample_size, replace=False)
    test_sample_size = min(2_000, len(test_pca))
    test_index = rng.choice(len(test_pca), size=test_sample_size, replace=False)

    palette = sns.color_palette("tab10", n_colors=len(centroids))
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for cluster_id in range(len(centroids)):
        mask = labels[sample_index] == cluster_id
        ax.scatter(
            train_pca[sample_index][mask, 0],
            train_pca[sample_index][mask, 1],
            s=10,
            alpha=0.35,
            color=palette[cluster_id],
            label=f"Cluster {cluster_id} (n={int((labels == cluster_id).sum()):,})",
            linewidths=0,
        )

    ax.scatter(
        test_pca[test_index, 0],
        test_pca[test_index, 1],
        s=12,
        marker=".",
        color="#888888",
        alpha=0.25,
        label="Test assignments",
        linewidths=0,
    )
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        s=220,
        marker="X",
        color="black",
        edgecolors="white",
        linewidths=1.5,
        label="Centroids",
        zorder=5,
    )
    for cluster_id, (x_value, y_value) in enumerate(centroids_pca):
        ax.annotate(
            str(cluster_id),
            (x_value, y_value),
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    explained = pca.explained_variance_ratio_ * 100
    ax.set_title("K-Means Customer Clusters - PCA Projection")
    ax.set_xlabel(f"Principal component 1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"Principal component 2 ({explained[1]:.1f}% variance)")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.text(
        0.01,
        -0.02,
        "Clusters fitted on training data only; grey points are test-set "
        "assignments. PCA is used for visualisation, not clustering.",
        fontsize=8,
        color="#666666",
    )
    save_figure(fig, path)


def _top_category(series: pd.Series) -> tuple[str, float]:
    """Return the most frequent category and its within-cluster percentage."""
    counts = series.value_counts(normalize=True)
    if counts.empty:
        return "", 0.0
    return str(counts.index[0]), float(counts.iloc[0] * 100)


def build_cluster_profiles(
    raw_train: pd.DataFrame,
    labels: np.ndarray,
    y_train_raw: pd.Series,
) -> pd.DataFrame:
    """
    Summarise each cluster using original customer attributes.

    The target is only used AFTER clustering as a post-hoc business validation
    metric; it is never an input to feature scaling, PCA, or K-Means.
    """
    frame = raw_train.copy()
    frame["cluster"] = labels
    overall_age = float(frame["age"].mean())
    overall_balance = float(frame["balance"].median())

    rows = []
    for cluster_id in sorted(frame["cluster"].unique()):
        subset = frame[frame["cluster"] == cluster_id]
        y_subset = y_train_raw.loc[subset.index]

        top_job, top_job_pct = _top_category(subset["job"])
        top_marital, top_marital_pct = _top_category(subset["marital"])
        top_education, top_education_pct = _top_category(subset["education"])
        top_poutcome, top_poutcome_pct = _top_category(subset["poutcome"])

        age_mean = float(subset["age"].mean())
        balance_median = float(subset["balance"].median())
        previous_mean = float(subset["previous"].mean())
        housing_yes = pct(int((subset["housing"] == "yes").sum()), len(subset))
        loan_yes = pct(int((subset["loan"] == "yes").sum()), len(subset))
        default_yes = pct(int((subset["default"] == "yes").sum()), len(subset))
        previous_contact = pct(
            int((subset["pdays"] != -1).sum()), len(subset)
        )
        previous_success = pct(
            int((subset["poutcome"] == "success").sum()), len(subset)
        )
        observed_rate = pct(int((y_subset == "yes").sum()), len(y_subset))

        age_relation = (
            "older" if age_mean > overall_age + 2
            else "younger" if age_mean < overall_age - 2
            else "average-age"
        )
        balance_relation = (
            "high-balance" if balance_median > overall_balance * 1.2
            else "low-balance" if balance_median < overall_balance * 0.8
            else "mid-balance"
        )
        description = (
            f"{age_relation}, {balance_relation} customers; most common job is "
            f"{top_job} ({top_job_pct:.0f}%), {housing_yes:.0f}% have a housing "
            f"loan, and {previous_contact:.0f}% had previous campaign contact."
        )

        rows.append(
            {
                "cluster": int(cluster_id),
                "n_customers": len(subset),
                "share_pct": pct(len(subset), len(frame)),
                "age_mean": age_mean,
                "balance_median": balance_median,
                "balance_mean": float(subset["balance"].mean()),
                "previous_mean": previous_mean,
                "pct_previous_contact": previous_contact,
                "pct_previous_success": previous_success,
                "pct_housing_loan": housing_yes,
                "pct_personal_loan": loan_yes,
                "pct_credit_default": default_yes,
                "top_job": top_job,
                "top_job_pct": top_job_pct,
                "top_marital": top_marital,
                "top_marital_pct": top_marital_pct,
                "top_education": top_education,
                "top_education_pct": top_education_pct,
                "top_poutcome": top_poutcome,
                "top_poutcome_pct": top_poutcome_pct,
                "observed_subscription_rate_pct": observed_rate,
                "description": description,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_classification_report(
    path: Path,
    model_name: str,
    search: GridSearchCV,
    evaluation: dict[str, object],
    tree_text: str | None,
    top_rules: pd.DataFrame | None,
    top_features: pd.DataFrame,
) -> None:
    """Write a detailed text report for one classifier."""
    metrics = evaluation["metrics"]
    lines = [
        "=" * 78,
        f"{model_name.upper()} - PHASE 3 REPORT",
        "=" * 78,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Random state: 42",
        "Positive class: 1 = subscribed (yes)",
        f"CV: StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)",
        "",
        "MODEL SELECTION (training data only)",
        "-" * 78,
        f"Best parameters: {search.best_params_}",
        f"Best mean CV F1: {search.best_score_:.4f}",
        "Selection metric: F1 (positive class) on 5 stratified folds",
        "",
        "TEST-SET EVALUATION (evaluated once, not used for tuning)",
        "-" * 78,
        f"Accuracy : {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}  (positive class)",
        f"Recall   : {metrics['recall']:.4f}  (positive class)",
        f"F1-score : {metrics['f1']:.4f}",
        f"ROC-AUC  : {metrics['roc_auc']:.4f}",
        f"Predicted positives: {metrics['predicted_positive']:,} "
        f"({metrics['predicted_positive_pct']:.2f}% of test rows; "
        f"base rate is 11.70%)",
        "",
        "Confusion matrix [[TN, FP], [FN, TP]]:",
        f"  TN={metrics['true_negative']:,}  FP={metrics['false_positive']:,}",
        f"  FN={metrics['false_negative']:,}  TP={metrics['true_positive']:,}",
        "",
        "CLASSIFICATION REPORT",
        "-" * 78,
        evaluation["classification_report"],
        "",
        "TOP FEATURE IMPORTANCES",
        "-" * 78,
        top_features.to_string(index=False),
        "",
    ]

    if top_rules is not None:
        high_precision = (
            top_rules[top_rules["samples"] >= 50]
            .sort_values(
                ["precision_pct", "subscribers"], ascending=[False, False]
            )
            .head(8)
        )
        lines.extend(
            [
                "MOST IMPORTANT DECISION RULES (ranked by training subscribers)",
                "-" * 78,
                top_rules.head(15).to_string(index=False),
                "",
                "HIGHEST-PRECISION RULES (minimum 50 training samples)",
                "-" * 78,
                high_precision.to_string(index=False),
                "",
                "Each rule is a root-to-leaf path. 'precision_pct' is the leaf's "
                "subscriber share; 'recall_of_all_subscribers_pct' is the share of "
                "all training subscribers captured by that leaf. Repeated numeric "
                "bounds are simplified for readability; the exact path is in the "
                "full tree below.",
                "",
            ]
        )

    if tree_text is not None:
        lines.extend(
            [
                "FULL TREE STRUCTURE (export_text, transformed feature space)",
                "-" * 78,
                tree_text,
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_model_comparison(
    metrics_frame: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Generate the comparison narrative from actual test metrics."""
    indexed = metrics_frame.set_index("model")
    dt = indexed.loc["Decision Tree"]
    rf = indexed.loc["Random Forest"]

    positive_support = float(dt["true_positive"] + dt["false_negative"])
    negative_support = float(dt["true_negative"] + dt["false_positive"])
    base_rate = pct(positive_support, positive_support + negative_support)

    best_f1 = indexed["f1"].idxmax()
    best_auc = indexed["roc_auc"].idxmax()
    best_recall = indexed["recall"].idxmax()
    best_precision = indexed["precision"].idxmax()
    best_accuracy = indexed["accuracy"].idxmax()

    lines = [
        "## 4. Test-set evaluation and model comparison",
        "",
        "All values below are from the held-out test set. The test set was not "
        "used for hyperparameter selection or threshold tuning.",
        "",
        md_table(
            metrics_frame[
                [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "true_positive",
                    "false_positive",
                    "false_negative",
                ]
            ].round(4)
        ),
        "",
        "### Which model performs better overall?",
        "",
        f"**{best_f1}** has the higher F1-score "
        f"({indexed.loc[best_f1, 'f1']:.4f}) and **{best_auc}** has the higher "
        f"ROC-AUC ({indexed.loc[best_auc, 'roc_auc']:.4f}). "
        f"{best_accuracy} also has the higher raw accuracy "
        f"({indexed.loc[best_accuracy, 'accuracy']:.4f}), but accuracy is not the "
        "primary criterion because 88.30% of customers do not subscribe. "
        f"{best_f1} is the stronger ranked classifier on this test set.",
        "",
        "### Which model identifies more actual subscribers?",
        "",
        f"**{best_recall}** has the higher recall "
        f"({indexed.loc[best_recall, 'recall']:.4f}), capturing "
        f"{int(indexed.loc[best_recall, 'true_positive']):,} of the "
        f"{int(indexed.loc[best_recall, 'true_positive'] + indexed.loc[best_recall, 'false_negative']):,} "
        "actual subscribers in the test set. "
        f"The Decision Tree captured {int(dt['true_positive']):,}; the Random "
        f"Forest captured {int(rf['true_positive']):,}. The model with higher "
        "recall is preferable when the bank wants to avoid missing interested "
        "customers and can absorb more calls.",
        "",
        f"The Decision Tree finds {int(dt['true_positive'] - rf['true_positive']):,} "
        "more subscribers than the Random Forest, but it also predicts "
        f"{int(dt['predicted_positive'] - rf['predicted_positive']):,} more "
        "positives, i.e. extra calls. The Random Forest finds "
        f"{pct(float(rf['true_positive']), float(dt['true_positive'])):.1f}% of "
        "the Decision Tree's subscribers while making "
        f"{pct(float(dt['predicted_positive'] - rf['predicted_positive']), float(dt['predicted_positive'])):.1f}% "
        "fewer predicted contacts. That is usually the better cost trade-off "
        "unless call capacity is effectively unconstrained.",
        "",
        "### Which model has better precision?",
        "",
        f"**{best_precision}** has the higher precision "
        f"({indexed.loc[best_precision, 'precision']:.4f}). Precision here means "
        "the share of predicted subscribers who actually subscribed: "
        f"Decision Tree {dt['precision']:.4f} vs Random Forest {rf['precision']:.4f}. "
        "Higher precision reduces wasted call-centre effort; higher recall reduces "
        "missed revenue opportunities. The choice depends on campaign economics.",
        "",
        f"Relative to the {base_rate:.2f}% positive base rate, precision lift is "
        f"**{float(dt['precision']) * 100 / base_rate:.2f}x** for the Decision Tree "
        f"and **{float(rf['precision']) * 100 / base_rate:.2f}x** for the Random "
        "Forest. A value above 1 means the model enriches the contact list "
        "compared with calling customers at random.",
        "",
        "### Which is more suitable for the bank's marketing objectives?",
        "",
        f"For a broad, budget-constrained campaign, **{best_f1}** is the more "
        "balanced choice because it has the best F1 and ROC-AUC, which indicates "
        "better ranking of likely subscribers across thresholds. "
        f"**{best_recall}** is preferable if the bank prioritises contacting as "
        "many potential subscribers as possible and has capacity for extra calls. "
        "If call costs dominate, the higher-precision model should be preferred, "
        "possibly with a threshold selected to match the contact budget. "
        "Accuracy should not decide the campaign: a constant 'no' model would "
        "score 88.30% and identify zero subscribers, which is worse for marketing "
        "than either evaluated model.",
        "",
        "### Trade-offs: complexity vs interpretability",
        "",
        f"- **Decision Tree:** selected with {dt['best_params']}. It can be read "
        "as explicit if/then rules, is easy to explain to marketing and "
        "compliance stakeholders, and exposes clear customer conditions. It is "
        "less stable and generally weaker in predictive performance because a "
        "single tree overfits or underfits more easily.",
        f"- **Random Forest:** selected with {rf['best_params']}. It averages many trees, giving "
        "stronger discrimination and more stable probabilities, but its internal "
        "logic is much harder to explain. It provides feature importance for "
        "global interpretation, while local explanations such as SHAP would be "
        "needed for individual decisions.",
        "- The Random Forest is the better candidate for a production ranking "
        "model; the Decision Tree is the better candidate for a transparent "
        "policy or a first explanation of which customer characteristics matter. "
        "This is a business trade-off, not only a metric trade-off.",
        "",
        "### Important imbalance note",
        "",
        "Both models use `class_weight='balanced'`. At the default 0.5 "
        f"threshold, the Decision Tree predicts {dt['predicted_positive']:,} "
        f"subscribers ({dt['predicted_positive_pct']:.2f}% of test rows) and the "
        f"Random Forest predicts {rf['predicted_positive']:,} "
        f"({rf['predicted_positive_pct']:.2f}%), compared with the 11.70% base "
        "rate. This deliberately trades precision for recall. The relevant "
        "business question is how many real subscribers are found at an "
        "acceptable cost per contact; accuracy alone hides this trade-off.",
        "",
    ]
    return "\n".join(lines), metrics_frame


def build_phase3_report(
    metrics_frame: pd.DataFrame,
    comparison_text: str,
    cv_table: pd.DataFrame,
    top_rules: pd.DataFrame,
    rf_importance: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    profiles: pd.DataFrame,
    best_k: int,
    tree_depth: int,
    tree_leaves: int,
    raw_test_cluster_distribution: pd.DataFrame,
) -> str:
    """Compose the Phase 3 overview report."""
    lines: list[str] = []
    add = lines.append

    add("# Phase 3 - Data Mining Model Development")
    add("")
    add("**Project:** Bank Marketing Intelligence System  ")
    add("**Course context:** Information Systems Management  ")
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    add("**Models:** Decision Tree, Random Forest, K-Means  ")
    add("**Scope:** Phase 3 only - no Phase 4/5 work and no test-set tuning.")
    add("")
    add("---")
    add("")
    add("## 1. Executive summary")
    add("")
    dt = metrics_frame.set_index("model").loc["Decision Tree"]
    rf = metrics_frame.set_index("model").loc["Random Forest"]
    add(f"- **Decision Tree** (depth={tree_depth}, leaves={tree_leaves}): "
        f"F1={dt['f1']:.4f}, recall={dt['recall']:.4f}, precision={dt['precision']:.4f}, "
        f"ROC-AUC={dt['roc_auc']:.4f}.")
    add(f"- **Random Forest** (best params in report): F1={rf['f1']:.4f}, "
        f"recall={rf['recall']:.4f}, precision={rf['precision']:.4f}, "
        f"ROC-AUC={rf['roc_auc']:.4f}.")
    add(f"- Both classifiers reuse the exact Phase 2 stratified split "
        "(36,168 train / 9,043 test) and exclude `duration`.")
    add(f"- Hyperparameters were selected with 5-fold stratified CV on training "
        f"data only; the test set was evaluated once per model.")
    add(f"- **K-Means** selected **k={best_k}** from k=2-10 using silhouette "
        f"score with a minimum cluster-share guardrail "
        f"({MIN_CLUSTER_SHARE_PCT:.0f}% of training customers). The target `y` "
        "and `duration` were not clustering inputs.")
    add("- The processed features are reused from Phase 2; the saved "
        "preprocessing pipeline is copied to `models/preprocessing_pipeline.joblib`.")
    add("")
    add("---")
    add("")
    add("## 2. Data and leakage protection")
    add("")
    add("- Inputs: `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, "
        "`y_test.csv` from Phase 2.")
    add("- The Phase 2 fitted pipeline was trained on the training split only. "
        "It was loaded and reused for classification, not refitted on test data.")
    add("- Model selection used `StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=42)` on the training set.")
    add("- `duration` is absent from the 59 model-ready Phase 2 features, so the "
        "pre-call models cannot exploit post-call leakage.")
    add("- K-Means has its own `StandardScaler` + `OneHotEncoder` pipeline fitted "
        "on training customer rows only. `y` is used solely after clustering to "
        "describe observed conversion, never to form clusters.")
    add("- The raw Phase 2 split was reproduced and verified against the saved "
        "`y_train.csv` before clustering, ensuring the same customers are used.")
    add("")
    add("## 3. Model selection")
    add("")
    add("Hyperparameters were selected by 5-fold stratified cross-validation on "
        "the training set, using F1 for the positive class as the selection "
        "metric. The test set is not involved in this table.")
    add("")
    add(md_table(cv_table.round(4)))
    add("")
    add("---")
    add("")
    add(comparison_text)
    add("---")
    add("")
    add("## 5. Decision Tree interpretation")
    add("")
    add(f"The selected tree has **{tree_depth} levels** and **{tree_leaves} leaves**. "
        "The chart shows the top 3 levels at readable scale; the complete tree is "
        "in `decision_tree_report.txt`, and all "
        f"{tree_leaves} leaf rules are in `decision_rules.csv`.")
    add("")
    add("### Key decision rules (ranked by subscribers captured in training)")
    add("")
    display_rules = top_rules[
        ["rank", "rule", "samples", "subscribers", "precision_pct",
         "recall_of_all_subscribers_pct"]
    ].head(10).copy()
    add(md_table(display_rules.round(2)))
    add("")
    high_precision = (
        top_rules[top_rules["samples"] >= 50]
        .sort_values(["precision_pct", "subscribers"], ascending=[False, False])
        .head(5)
    )
    add("### Highest-precision rules (minimum 50 training samples)")
    add("")
    add(md_table(
        high_precision[
            ["rule", "samples", "subscribers", "precision_pct"]
        ].round(2)
    ))
    add("")
    add(f"Example: the top-precision leaf captures "
        f"**{high_precision.iloc[0]['subscribers']:.0f} subscribers from "
        f"{high_precision.iloc[0]['samples']:.0f} customers "
        f"({high_precision.iloc[0]['precision_pct']:.1f}% precision)**. "
        "High-precision rules are better for small, targeted campaigns; "
        "high-volume rules are better for broad outreach."
        if not high_precision.empty else
        "No leaf had at least 50 training samples in this tree.")
    add("")
    add("Rules use original units where possible: numeric thresholds were "
        "converted back from the standardized scale using the Phase 2 scaler, and "
        "one-hot splits are expressed as category membership. Repeated numeric "
        "bounds are simplified for readability; exact paths remain in the full "
        "text tree.")
    add("")
    add("### Top features")
    add("")
    add("Feature importances for both models are in `feature_importance.csv` and "
        "`feature_importance.png`.")
    add("")
    add("Random Forest top features:")
    add("")
    add(md_table(
        rf_importance[["feature", "importance"]].head(10).round(4)
    ))
    add("")
    add("---")
    add("")
    add("## 6. K-Means customer segmentation")
    add("")
    add(f"- Candidate k values: {K_RANGE[0]}-{K_RANGE[-1]}.")
    add(f"- Selected **k={best_k}** by highest silhouette among solutions where "
        f"every cluster contains at least {MIN_CLUSTER_SHARE_PCT:.0f}% of training "
        "customers. This guardrail prevents technically good but degenerate "
        "one-customer clusters.")
    add("- Full metrics and segment profiles are in `clustering_report.md`, "
        "`clustering_metrics.csv`, and `cluster_profiles.csv`.")
    add("")
    add(md_table(
        cluster_metrics[
            ["k", "inertia", "silhouette_score", "min_cluster_share_pct"]
        ].round(4)
    ))
    add("")
    add("Cluster sizes on the training set:")
    add("")
    add(md_table(
        profiles[["cluster", "n_customers", "share_pct",
                  "observed_subscription_rate_pct"]].round(2)
    ))
    add("")
    add("Held-out test assignment distribution (descriptive check only):")
    add("")
    add(md_table(raw_test_cluster_distribution.round(2)))
    add("")
    add("The observed subscription rate is **post-hoc** information and was not "
        "an input to K-Means. It shows whether segments differ in business value.")
    add("")
    add("---")
    add("")
    add("## 7. Verification, issues, and limitations")
    add("")
    add("- Both classifiers loaded from joblib reproduce identical test "
        "predictions; the preprocessing pipeline is also reloadable.")
    add("- Zero missing/non-finite values were present in the Phase 2 matrices.")
    add("- K-Means input feature list is "
        + ", ".join(f"`{name}`" for name in CLUSTER_FEATURES)
        + "; neither `y` nor `duration` is included.")
    add("- The test set remains untouched for final evaluation and is preserved "
        "for Phase 5.")
    add("")
    add("### Issues found during implementation")
    add("")
    add("- **Class-weighted tree values:** scikit-learn stores normalized, "
        "class-weighted proportions in `tree_.value` when "
        "`class_weight='balanced'` is used. Rule counts are therefore computed "
        "from actual training rows routed to each leaf via `model.apply()`, not "
        "from `tree_.value`.")
    add("- **Tree size:** the selected tree has 121 leaves, so plotting it in full "
        "is unreadable. The chart shows the top 3 levels, while the full structure "
        "and all leaf rules are exported as text/CSV.")
    add("- **Degenerate clustering solutions:** k=7 had the highest raw silhouette "
        "(0.171), but it produced a one-customer cluster and a 170-customer "
        "cluster. The minimum 2% cluster-share guardrail excludes it; k=4 has the "
        "best valid silhouette (0.167) and interpretable segment sizes.")
    add("- **CV ranking:** an initial results-export line sorted on a non-existent "
        "`rank` column; it was corrected to `rank_test_score`. This affected only "
        "the exported CV CSV, not model selection.")
    add("")
    add("### Limitations")
    add("")
    add("- Class imbalance remains: test positives are 1,058 of 9,043 (11.70%). "
        "Precision/recall trade-offs must be tied to campaign cost and capacity.")
    add("- Hyperparameters were selected on F1 at the default 0.5 decision "
        "threshold. A tuned threshold or precision-at-budget objective may better "
        "match the bank's economics.")
    add("- Phase 2 scaling/encoding was fitted on the whole training set before "
        "CV; this is a minor optimistic bias for CV estimates but does not touch "
        "the test set. In production retraining, keep preprocessing inside each "
        "fold or refit on the final training set.")
    add("- Tree feature importance is impurity-based and can favor continuous "
        "variables; it is not causal evidence.")
    add("- K-Means assumes roughly spherical clusters and treats one-hot "
        "dimensions as Euclidean distances. Segments are descriptive, not "
        "predictive, and silhouette scores are modest.")
    add("- No campaign cost/revenue data are available, so model selection cannot "
        "be expressed directly in profit.")
    add("")
    add("---")
    add("")
    add("## 8. Reproduce")
    add("")
    add("```bash")
    add("python src/phase3_models.py")
    add("```")
    add("")
    add("Optional:")
    add("")
    add("```bash")
    add("python src/phase3_models.py \\")
    add("    --processed-dir data/processed \\")
    add("    --models-dir models \\")
    add("    --reports-dir reports/phase3")
    add("```")
    add("")
    add("The script validates inputs, trains all three models, verifies reloading, "
        "and writes the reports, charts, and model artefacts listed below.")
    add("")
    add("### Artefacts")
    add("")
    for name in [
        "models/decision_tree.joblib",
        "models/random_forest.joblib",
        "models/kmeans.joblib",
        "models/preprocessing_pipeline.joblib",
        "reports/phase3/classification_comparison.csv",
        "reports/phase3/decision_tree_report.txt",
        "reports/phase3/random_forest_report.txt",
        "reports/phase3/clustering_report.md",
        "reports/phase3/feature_importance.csv",
        "reports/phase3/cluster_profiles.csv",
        "reports/phase3/decision_rules.csv",
        "reports/phase3/clustering_metrics.csv",
        "reports/phase3/phase3_report.md",
        "reports/phase3/*.png (8 charts)",
    ]:
        add(f"- `{name}`")
    add("")

    return "\n".join(lines)


def build_clustering_report(
    metrics: pd.DataFrame,
    best_k: int,
    profiles: pd.DataFrame,
    test_distribution: pd.DataFrame,
    train_silhouette: float,
    test_silhouette: float,
    n_features: int,
) -> str:
    """Compose the clustering report with metrics and segment narratives."""
    lines: list[str] = []
    add = lines.append

    add("# K-Means Customer Segmentation Report")
    add("")
    add("**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "  ")
    add("**Method:** StandardScaler + OneHotEncoder + K-Means  ")
    add(f"**Selected clusters:** {best_k}  ")
    add("**Training customers:** 36,168  ")
    add("**Input features:** " + ", ".join(f"`{name}`" for name in CLUSTER_FEATURES))
    add("")
    add("> **Independence statement:** the target `y` and `duration` were not "
        "used as clustering inputs. `y` appears only after clustering, as a "
        "post-hoc observed subscription rate to help interpret the segments.")
    add("")
    add("---")
    add("")
    add("## 1. Approach")
    add("")
    add("K-Means is an unsupervised task, independent of the supervised "
        "classifiers. It uses stable customer profile attributes (age, balance, "
        "occupation, marital status, education, loans, default) and prior "
        "campaign history. Current-campaign execution fields (`contact`, `day`, "
        "`month`, `campaign`) and `duration` are excluded so that segments "
        "describe customers rather than campaign tactics.")
    add("")
    add(f"- Numerical features are standardized and nominal features are "
        f"one-hot encoded inside a training-only pipeline.")
    add(f"- The resulting training matrix has {n_features} features.")
    add(f"- The same 80/20 customer split as Phase 2 is reproduced; the saved "
        f"`y_train.csv` is used only to verify row identity.")
    add("")
    add("## 2. Choosing the number of clusters")
    add("")
    add("The Elbow Method examines inertia, while silhouette scores measure "
        "cluster cohesion and separation. The selection rule is: choose the "
        f"highest silhouette among solutions where every cluster has at least "
        f"{MIN_CLUSTER_SHARE_PCT:.0f}% of training customers.")
    add("")
    display = metrics[
        ["k", "inertia", "silhouette_score", "min_cluster_count",
         "min_cluster_share_pct", "cluster_sizes"]
    ].copy()
    add(md_table(display.round(4)))
    add("")
    add(f"**Selected k = {best_k}.** Silhouette scores are modest, which is "
        "normal for mixed banking data with one-hot dimensions; the value is in "
        "interpretability and business differentiation, not in a high absolute "
        "score. Very large k values produced tiny or singleton clusters and were "
        "rejected by the minimum-share guardrail.")
    add("")
    add(f"- Training silhouette (5,000-row sample): **{train_silhouette:.4f}**")
    add(f"- Held-out test silhouette (assignment generalisation check): "
        f"**{test_silhouette:.4f}**")
    add("")
    add("![Elbow method](elbow_method.png)")
    add("")
    add("![Silhouette scores](silhouette_scores.png)")
    add("")
    add("## 3. Cluster visualisation")
    add("")
    add("![Customer clusters in PCA space](customer_clusters.png)")
    add("")
    add("PCA is used only for two-dimensional display; K-Means uses the full "
        "scaled/encoded feature space.")
    add("")
    add("## 4. Cluster profiles")
    add("")
    add(md_table(
        profiles[
            ["cluster", "n_customers", "share_pct", "age_mean",
             "balance_median", "top_job", "top_job_pct", "top_education",
             "pct_housing_loan", "pct_previous_contact",
             "observed_subscription_rate_pct"]
        ].round(2)
    ))
    add("")
    add("Full profiles, including marital status, loans, default, previous "
        "success, and top poutcome, are in `cluster_profiles.csv`.")
    add("")
    add("## 5. Segment interpretation")
    add("")
    overall_rate = (
        profiles["observed_subscription_rate_pct"] * profiles["n_customers"]
    ).sum() / profiles["n_customers"].sum()
    for _, row in profiles.iterrows():
        rate_gap = row["observed_subscription_rate_pct"] - overall_rate
        add(f"### Cluster {int(row['cluster'])} - {int(row['n_customers']):,} "
            f"customers ({row['share_pct']:.1f}%)")
        add("")
        add(f"- Mean age **{row['age_mean']:.1f} years**; median balance "
            f"**{row['balance_median']:,.0f} EUR**; mean previous contacts "
            f"**{row['previous_mean']:.2f}**.")
        add(f"- Most common job **{row['top_job']}** ({row['top_job_pct']:.1f}%), "
            f"education **{row['top_education']}** "
            f"({row['top_education_pct']:.1f}%), marital status "
            f"**{row['top_marital']}** ({row['top_marital_pct']:.1f}%).")
        add(f"- Housing loan **{row['pct_housing_loan']:.1f}%**, personal loan "
            f"**{row['pct_personal_loan']:.1f}%**, credit default "
            f"**{row['pct_credit_default']:.1f}%**.")
        add(f"- Previous campaign contact **{row['pct_previous_contact']:.1f}%**; "
            f"previous success **{row['pct_previous_success']:.1f}%**.")
        add(f"- Observed subscription rate (post-hoc, not used for clustering): "
            f"**{row['observed_subscription_rate_pct']:.2f}%** "
            f"({rate_gap:+.2f} pp vs overall {overall_rate:.2f}%).")
        add("")
        add(f"**Interpretation:** {row['description']}")
        add("")
        if rate_gap > 2:
            add("This segment is above-average in observed conversion and is a "
                "candidate for priority outreach or a targeted term-deposit offer.")
        elif rate_gap < -2:
            add("This segment is below-average in observed conversion; use lower-"
                "cost channels or a different product proposition rather than "
                "high-contact campaigns.")
        else:
            add("This segment is near the overall conversion rate; it can support "
                "volume campaigns but is less differentiated than the extremes.")
        add("")

    add("## 6. Held-out test assignment check")
    add("")
    add(md_table(test_distribution.round(2)))
    add("")
    add("The test distribution is a descriptive stability check only. Test "
        "customers did not influence the scaler, encoder, or centroids.")
    add("")
    add("## 7. Business use and limitations")
    add("")
    add("- Segments can guide differentiated offers, channel choice, and contact "
        "frequency; they do not replace the supervised conversion model.")
    add("- K-Means assumes roughly spherical, equally sized clusters and uses "
        "Euclidean distance on one-hot dimensions.")
    add("- The silhouette score is modest; cluster boundaries are not sharp, and "
        "membership should be treated as a marketing heuristic.")
    add("- The observed subscription rate is descriptive and not causal. It was "
        "not used to fit the clusters.")
    add("- Cluster labels are arbitrary numeric IDs; business names should be "
        "agreed with marketing stakeholders.")
    add("")
    add("## 8. Reproduce")
    add("")
    add("```bash")
    add("python src/phase3_models.py")
    add("```")
    add("")
    add("The clustering step is deterministic with `random_state=42` and "
        "`n_init=10`; it is fitted on the same training rows as the classifiers.")
    add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact verification
# ---------------------------------------------------------------------------

def verify_saved_models(
    models_dir: Path,
    models: dict[str, object],
    X_test: pd.DataFrame,
    kmeans_features_test: pd.DataFrame,
) -> None:
    """Reload every saved model and confirm predictions are reproducible."""
    checks = {
        "decision_tree.joblib": models["Decision Tree"],
        "random_forest.joblib": models["Random Forest"],
    }
    for filename, original in checks.items():
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model artefact missing: {path}")
        loaded = joblib.load(path)
        original_pred = original.predict(X_test)
        loaded_pred = loaded.predict(X_test)
        if not np.array_equal(original_pred, loaded_pred):
            raise RuntimeError(f"Reloaded model {filename} predictions differ.")
        print(f"  verified reload: {filename} ({len(loaded_pred):,} predictions identical)")

    kmeans_path = models_dir / "kmeans.joblib"
    if not kmeans_path.exists():
        raise FileNotFoundError(f"Model artefact missing: {kmeans_path}")
    loaded_kmeans = joblib.load(kmeans_path)
    labels = loaded_kmeans.predict(kmeans_features_test)
    if len(labels) != len(kmeans_features_test):
        raise RuntimeError("Reloaded K-Means produced an invalid assignment count.")
    print(
        f"  verified reload: kmeans.joblib "
        f"({len(labels):,} test assignments, clusters={sorted(set(labels))})"
    )

    preprocessing_path = models_dir / "preprocessing_pipeline.joblib"
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Pipeline artefact missing: {preprocessing_path}")
    loaded_pipeline = joblib.load(preprocessing_path)
    if not hasattr(loaded_pipeline, "transform"):
        raise RuntimeError("Reloaded preprocessing pipeline is not fitted.")
    print("  verified reload: preprocessing_pipeline.joblib")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 3 model development for the Bank Marketing Intelligence System."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Phase 2 output directory (default: data/processed).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Model output directory (default: models).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Report output directory (default: reports/phase3).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate, and persist all Phase 3 models."""
    args = parse_args(argv)
    processed_dir = Path(args.processed_dir).expanduser()
    models_dir = Path(args.models_dir).expanduser()
    reports_dir = Path(args.reports_dir).expanduser()

    if not processed_dir.is_absolute():
        processed_dir = (Path.cwd() / processed_dir).resolve()
    if not models_dir.is_absolute():
        models_dir = (Path.cwd() / models_dir).resolve()
    if not reports_dir.is_absolute():
        reports_dir = (Path.cwd() / reports_dir).resolve()

    print("=" * 78)
    print("BANK MARKETING INTELLIGENCE SYSTEM - PHASE 3")
    print("Decision Tree, Random Forest, and K-Means segmentation")
    print("=" * 78)

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            feature_dictionary,
            preprocessing_pipeline,
        ) = load_phase2_inputs(processed_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("INPUTS AND LEAKAGE CHECKS")
    print("=" * 78)
    print(f"X_train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    print(f"X_test : {X_test.shape[0]:,} x {X_test.shape[1]}")
    print(f"Train positive rate: {pct(int(y_train.sum()), len(y_train)):.2f}%")
    print(f"Test positive rate : {pct(int(y_test.sum()), len(y_test)):.2f}%")
    print("Missing values: 0 | Non-finite values: 0")
    print("Reusing Phase 2 fitted preprocessing pipeline (not refitted).")

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    # Copy the Phase 2 pipeline for Phase 5 reuse ----------------------------
    preprocessing_copy = models_dir / "preprocessing_pipeline.joblib"
    joblib.dump(preprocessing_pipeline, preprocessing_copy)
    print(f"  saved: {preprocessing_copy.name} (copy of Phase 2 pipeline)")

    metadata = feature_dictionary.set_index("feature_name")
    scaler_stats: dict[str, tuple[float, float]] = {}
    try:
        numeric_pipeline = preprocessing_pipeline.named_steps[
            "preprocessor"
        ].named_transformers_["num"]
        scaler = numeric_pipeline.named_steps["scaler"]
        for name, mean, scale in zip(
            numeric_pipeline.feature_names_in_,
            scaler.mean_,
            scaler.scale_,
        ):
            scaler_stats[str(name)] = (float(mean), float(scale))
    except (AttributeError, KeyError) as exc:
        print(
            f"WARNING: could not extract scaler statistics for rule conversion: {exc}",
            file=sys.stderr,
        )

    # Classification ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("MODEL 1 - DECISION TREE (CV model selection on training data)")
    print("=" * 78)
    try:
        dt_search = tune_decision_tree(X_train, y_train)
        dt_model = dt_search.best_estimator_
        dt_evaluation = evaluate_classifier(
            "Decision Tree",
            dt_model,
            X_test,
            y_test,
            dt_search.best_score_,
            dt_search.best_params_,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"\nERROR training Decision Tree: {exc}", file=sys.stderr)
        return 1
    print(
        f"  test F1={dt_evaluation['metrics']['f1']:.4f} | "
        f"recall={dt_evaluation['metrics']['recall']:.4f} | "
        f"precision={dt_evaluation['metrics']['precision']:.4f} | "
        f"ROC-AUC={dt_evaluation['metrics']['roc_auc']:.4f}"
    )
    joblib.dump(dt_model, models_dir / "decision_tree.joblib")
    print("  saved: decision_tree.joblib")

    print("\n" + "=" * 78)
    print("MODEL 2 - RANDOM FOREST (CV model selection on training data)")
    print("=" * 78)
    try:
        rf_search = tune_random_forest(X_train, y_train)
        rf_model = rf_search.best_estimator_
        rf_evaluation = evaluate_classifier(
            "Random Forest",
            rf_model,
            X_test,
            y_test,
            rf_search.best_score_,
            rf_search.best_params_,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"\nERROR training Random Forest: {exc}", file=sys.stderr)
        return 1
    print(
        f"  test F1={rf_evaluation['metrics']['f1']:.4f} | "
        f"recall={rf_evaluation['metrics']['recall']:.4f} | "
        f"precision={rf_evaluation['metrics']['precision']:.4f} | "
        f"ROC-AUC={rf_evaluation['metrics']['roc_auc']:.4f}"
    )
    joblib.dump(rf_model, models_dir / "random_forest.joblib")
    print("  saved: random_forest.joblib")

    # Classification evaluation artefacts ------------------------------------
    print("\n" + "=" * 78)
    print("MODEL EVALUATION AND CHARTS")
    print("=" * 78)
    eval_map = {
        "Decision Tree": dt_evaluation,
        "Random Forest": rf_evaluation,
    }
    metrics_frame = pd.DataFrame(
        [dt_evaluation["metrics"], rf_evaluation["metrics"]]
    )
    comparison_csv = reports_dir / "classification_comparison.csv"
    comparison_output = metrics_frame.copy()
    rounded_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "predicted_positive_pct",
        "cv_f1_mean",
    ]
    comparison_output[rounded_columns] = comparison_output[rounded_columns].round(4)
    comparison_output.to_csv(comparison_csv, index=False)
    print(f"  saved: {comparison_csv.name}")

    comparison_text, _ = build_model_comparison(metrics_frame)

    plot_confusion_matrix(
        dt_evaluation["confusion_matrix"],
        "Decision Tree",
        reports_dir / "confusion_matrix_decision_tree.png",
    )
    plot_confusion_matrix(
        rf_evaluation["confusion_matrix"],
        "Random Forest",
        reports_dir / "confusion_matrix_random_forest.png",
    )
    plot_roc_curves(eval_map, y_test, reports_dir / "roc_curve.png")

    feature_importance = build_feature_importance_frame(
        {"Decision Tree": dt_model, "Random Forest": rf_model},
        list(X_train.columns),
    )
    feature_importance.to_csv(
        reports_dir / "feature_importance.csv", index=False
    )
    print("  saved: feature_importance.csv")
    plot_feature_importance(
        feature_importance, reports_dir / "feature_importance.png"
    )

    # Decision tree visualisation and rules -----------------------------------
    try:
        dt_rules = extract_decision_rules(
            dt_model,
            X_train,
            y_train,
            list(X_train.columns),
            metadata,
            scaler_stats,
        )
        dt_rules.to_csv(reports_dir / "decision_rules.csv", index=False)
        print("  saved: decision_rules.csv")
    except (KeyError, ValueError) as exc:
        print(f"\nERROR extracting decision rules: {exc}", file=sys.stderr)
        return 1

    plot_decision_tree(
        dt_model,
        list(X_train.columns),
        metadata,
        reports_dir / "decision_tree_visualization.png",
    )

    dt_report_text = export_text(
        dt_model, feature_names=list(X_train.columns), decimals=3
    )
    top_dt_features = feature_importance[
        feature_importance["model"] == "Decision Tree"
    ][["feature", "importance"]].head(20)
    top_rf_features = feature_importance[
        feature_importance["model"] == "Random Forest"
    ][["feature", "importance"]].head(20)
    write_classification_report(
        reports_dir / "decision_tree_report.txt",
        "Decision Tree",
        dt_search,
        dt_evaluation,
        dt_report_text,
        dt_rules,
        top_dt_features,
    )
    print("  saved: decision_tree_report.txt")
    write_classification_report(
        reports_dir / "random_forest_report.txt",
        "Random Forest",
        rf_search,
        rf_evaluation,
        None,
        None,
        top_rf_features,
    )
    print("  saved: random_forest_report.txt")

    # K-Means ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MODEL 3 - K-MEANS CUSTOMER SEGMENTATION (unsupervised)")
    print("=" * 78)
    try:
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = load_clustering_raw_data(
            X_train, X_test, y_train
        )
        cluster_train_features = build_cluster_features(X_train_raw)
        cluster_test_features = build_cluster_features(X_test_raw)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR preparing clustering data: {exc}", file=sys.stderr)
        return 1

    if TARGET in cluster_train_features.columns or "duration" in cluster_train_features.columns:
        print("ERROR: y or duration leaked into clustering features.", file=sys.stderr)
        return 1
    print("Clustering inputs (y and duration excluded):")
    print("  " + ", ".join(CLUSTER_FEATURES))

    cluster_preprocessor = build_cluster_preprocessor()
    try:
        Z_train = cluster_preprocessor.fit_transform(cluster_train_features)
        Z_test = cluster_preprocessor.transform(cluster_test_features)
    except (ValueError, TypeError) as exc:
        print(f"\nERROR fitting clustering preprocessing: {exc}", file=sys.stderr)
        return 1

    rng = np.random.RandomState(RANDOM_STATE)
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(Z_train))
    silhouette_index = rng.choice(len(Z_train), size=sample_size, replace=False)

    print(f"Clustering matrix: {Z_train.shape[0]:,} x {Z_train.shape[1]}")
    print("Evaluating k=2..10 (inertia + silhouette)...")
    cluster_metrics, _ = evaluate_cluster_counts(Z_train, silhouette_index)

    valid = cluster_metrics[
        cluster_metrics["min_cluster_share_pct"] >= MIN_CLUSTER_SHARE_PCT
    ]
    if valid.empty:
        valid = cluster_metrics
    best_k = int(
        valid.sort_values(["silhouette_score", "k"], ascending=[False, True])
        .iloc[0]["k"]
    )
    cluster_metrics.to_csv(
        reports_dir / "clustering_metrics.csv", index=False
    )
    print(cluster_metrics.round(4).to_string(index=False))
    print(f"Selected k={best_k} (highest silhouette with minimum cluster share "
          f">= {MIN_CLUSTER_SHARE_PCT:.0f}%).")

    # Final pipeline: the preprocessor is fitted on train, then the selected
    # K-Means is fitted on its transformed output.
    final_kmeans = KMeans(
        n_clusters=best_k, random_state=RANDOM_STATE, n_init=10
    )
    final_kmeans.fit(Z_train)
    cluster_pipeline = Pipeline(
        steps=[
            ("preprocessor", cluster_preprocessor),
            ("kmeans", final_kmeans),
        ]
    )
    # Verify the fitted pipeline assigns the same training labels.
    pipeline_labels = cluster_pipeline.predict(cluster_train_features)
    if not np.array_equal(pipeline_labels, final_kmeans.labels_):
        print(
            "WARNING: pipeline labels differ from direct K-Means labels; "
            "check preprocessing consistency.",
            file=sys.stderr,
        )

    labels_train = final_kmeans.labels_.astype(int)
    labels_test = cluster_pipeline.predict(cluster_test_features).astype(int)

    train_silhouette = float(
        silhouette_score(
            Z_train[silhouette_index],
            labels_train[silhouette_index],
            random_state=RANDOM_STATE,
        )
    )
    test_sample_size = min(3_000, len(Z_test))
    test_index = rng.choice(len(Z_test), size=test_sample_size, replace=False)
    test_silhouette = float(
        silhouette_score(
            Z_test[test_index],
            labels_test[test_index],
            random_state=RANDOM_STATE,
        )
    )
    print(f"Silhouette (train sample): {train_silhouette:.4f}")
    print(f"Silhouette (test sample) : {test_silhouette:.4f}")

    joblib.dump(cluster_pipeline, models_dir / "kmeans.joblib")
    print("  saved: kmeans.joblib")

    plot_elbow(cluster_metrics, best_k, reports_dir / "elbow_method.png")
    plot_silhouette(cluster_metrics, best_k, reports_dir / "silhouette_scores.png")
    plot_clusters_pca(
        Z_train,
        labels_train,
        final_kmeans.cluster_centers_,
        Z_test,
        labels_test,
        reports_dir / "customer_clusters.png",
    )

    profiles = build_cluster_profiles(
        X_train_raw, labels_train, y_train_raw
    )
    profiles.to_csv(reports_dir / "cluster_profiles.csv", index=False)
    print("  saved: cluster_profiles.csv")

    test_distribution = (
        pd.Series(labels_test, name="cluster")
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n_customers")
    )
    test_distribution["cluster"] = test_distribution["cluster"].astype(int)
    test_distribution["n_customers"] = test_distribution["n_customers"].astype(int)
    test_distribution["share_pct"] = (
        test_distribution["n_customers"] / len(labels_test) * 100
    )

    clustering_report = build_clustering_report(
        cluster_metrics,
        best_k,
        profiles,
        test_distribution,
        train_silhouette,
        test_silhouette,
        Z_train.shape[1],
    )
    (reports_dir / "clustering_report.md").write_text(
        clustering_report, encoding="utf-8"
    )
    print("  saved: clustering_report.md")

    # Phase 3 overview report ------------------------------------------------
    cv_table = pd.DataFrame(
        [
            {
                "model": "Decision Tree",
                "best_params": json.dumps(dt_search.best_params_, sort_keys=True),
                "cv_f1_mean": dt_search.best_score_,
                "cv_f1_std": dt_search.cv_results_["std_test_score"][dt_search.best_index_],
                "test_f1": dt_evaluation["metrics"]["f1"],
                "test_roc_auc": dt_evaluation["metrics"]["roc_auc"],
            },
            {
                "model": "Random Forest",
                "best_params": json.dumps(rf_search.best_params_, sort_keys=True),
                "cv_f1_mean": rf_search.best_score_,
                "cv_f1_std": rf_search.cv_results_["std_test_score"][rf_search.best_index_],
                "test_f1": rf_evaluation["metrics"]["f1"],
                "test_roc_auc": rf_evaluation["metrics"]["roc_auc"],
            },
        ]
    )
    phase3_report = build_phase3_report(
        metrics_frame,
        comparison_text,
        cv_table,
        dt_rules,
        top_rf_features,
        cluster_metrics,
        profiles,
        best_k,
        dt_model.get_depth(),
        dt_model.get_n_leaves(),
        test_distribution,
    )
    (reports_dir / "phase3_report.md").write_text(
        phase3_report, encoding="utf-8"
    )
    print("  saved: phase3_report.md")

    # Save CV results for reproducibility -----------------------------------
    pd.DataFrame(dt_search.cv_results_).sort_values("rank_test_score").to_csv(
        reports_dir / "cv_results_decision_tree.csv", index=False
    )
    pd.DataFrame(rf_search.cv_results_).sort_values("rank_test_score").to_csv(
        reports_dir / "cv_results_random_forest.csv", index=False
    )
    print("  saved: cv_results_decision_tree.csv, cv_results_random_forest.csv")

    # Verification ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    try:
        verify_saved_models(
            models_dir,
            {"Decision Tree": dt_model, "Random Forest": rf_model},
            X_test,
            cluster_test_features,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR verifying saved models: {exc}", file=sys.stderr)
        return 1

    required_charts = [
        "confusion_matrix_decision_tree.png",
        "confusion_matrix_random_forest.png",
        "decision_tree_visualization.png",
        "feature_importance.png",
        "roc_curve.png",
        "elbow_method.png",
        "silhouette_scores.png",
        "customer_clusters.png",
    ]
    missing_charts = [
        name for name in required_charts if not (reports_dir / name).exists()
    ]
    if missing_charts:
        print(f"ERROR missing charts: {missing_charts}", file=sys.stderr)
        return 1
    print(f"  verified charts: {len(required_charts)}")

    print("\n" + "=" * 78)
    print("PHASE 3 COMPLETE")
    print("=" * 78)
    print(f"Models directory : {models_dir}")
    print(f"Reports directory: {reports_dir}")
    print(f"Clusters selected: {best_k}")
    print(f"Decision Tree    : F1={dt_evaluation['metrics']['f1']:.4f}, "
          f"ROC-AUC={dt_evaluation['metrics']['roc_auc']:.4f}")
    print(f"Random Forest    : F1={rf_evaluation['metrics']['f1']:.4f}, "
          f"ROC-AUC={rf_evaluation['metrics']['roc_auc']:.4f}")
    print("\nNo Phase 4/5 work was performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
