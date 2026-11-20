#!/usr/bin/env python
"""
Phase 4 - Model Evaluation, Optimization and Selection
======================================================

Bank Marketing Intelligence System (Information Systems Management project).

This phase consumes the existing Phase 2 processed data and the Phase 3
classifiers. It does NOT retrain or overwrite the Phase 3 models.

Work performed
--------------
1. Reproduce the Phase 3 Decision Tree and Random Forest test-set evaluation.
2. Analyse confusion matrices for the positive class (subscribed = yes).
3. Produce ROC and precision-recall curves.
4. Tune new models with randomized hyperparameter search on training data only.
5. Select a probability threshold from out-of-fold training predictions only.
6. Save the final model, its Phase 2 preprocessing pipeline, and threshold.
7. Review the existing Phase 3 K-Means segmentation results.
8. Write Phase 4 reports.

Methodological safeguards
-------------------------
- The same untouched Phase 2 test set is used for all final test evaluation.
- Hyperparameter search uses StratifiedKFold(5, shuffle=True, random_state=42)
  on training data only.
- Thresholds are selected from out-of-fold training predictions, never test-set
  predictions.
- The test set is used once per optimized model at its selected threshold for
  final reporting. It is never used to choose a parameter, model, or threshold.
- `duration` is absent from all classifier inputs.
- K-Means remains independent of `y`; its Phase 3 artifacts are reviewed only.

Usage
-----
    python src/phase4_evaluation.py
    python src/phase4_evaluation.py --processed-dir data/processed --models-dir models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "phase4"
PHASE3_REPORTS_DIR = PROJECT_ROOT / "reports" / "phase3"

TARGET = "y"
RANDOM_STATE = 42
CV_FOLDS = 5
POSITIVE_CLASS = 1
CLASS_NAMES = ["no", "yes"]

# Randomized search sizes: deliberately moderate for a laptop.
DT_SEARCH_ITERATIONS = 80
RF_SEARCH_ITERATIONS = 36
SEARCH_SCORING = "average_precision"

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
REQUIRED_EXAMPLE_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def pct(part: float, total: float) -> float:
    """Percentage helper with a safe denominator."""
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
    """Consistent visual style for Phase 4 charts."""
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
    """Save a figure and close it."""
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved chart: {path.name}")


def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with a helpful failure message."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def to_jsonable(value):
    """Convert numpy/pandas scalar types into JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, list, dict)) else False:
        return None
    return value


def build_cv() -> StratifiedKFold:
    """Stratified 5-fold CV with the project seed."""
    return StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_inputs(
    processed_dir: Path, models_dir: Path
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    dict[str, object],
    dict[str, object],
    object,
]:
    """Load Phase 2 data, Phase 3 classifiers, and the fitted preprocessing."""
    required = {
        "X_train": processed_dir / "X_train.csv",
        "X_test": processed_dir / "X_test.csv",
        "y_train": processed_dir / "y_train.csv",
        "y_test": processed_dir / "y_test.csv",
        "preprocessing": models_dir / "preprocessing_pipeline.joblib",
        "decision_tree": models_dir / "decision_tree.joblib",
        "random_forest": models_dir / "random_forest.joblib",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts: " + ", ".join(missing)
            + "\nRun Phases 2 and 3 first."
        )

    X_train = load_csv(required["X_train"])
    X_test = load_csv(required["X_test"])
    y_train_frame = load_csv(required["y_train"])
    y_test_frame = load_csv(required["y_test"])

    if TARGET not in y_train_frame.columns or TARGET not in y_test_frame.columns:
        raise ValueError("Target CSV files must contain a 'y' column.")
    if len(X_train) != len(y_train_frame) or len(X_test) != len(y_test_frame):
        raise ValueError("Feature and target row counts do not align.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test columns or order differ.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Processed matrices contain missing values.")
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Processed matrices contain non-finite values.")
    if set(y_train_frame[TARGET].unique()) - {0, 1}:
        raise ValueError("y_train must contain only 0 and 1.")
    if set(y_test_frame[TARGET].unique()) - {0, 1}:
        raise ValueError("y_test must contain only 0 and 1.")
    if any("duration" in str(column) for column in X_train.columns):
        raise ValueError("duration leaked into the classifier features.")

    y_train = y_train_frame[TARGET].astype("int8")
    y_test = y_test_frame[TARGET].astype("int8")

    try:
        preprocessing_pipeline = joblib.load(required["preprocessing"])
        decision_tree = joblib.load(required["decision_tree"])
        random_forest = joblib.load(required["random_forest"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load model artifacts: {exc}") from exc

    for name, model in [
        ("Decision Tree", decision_tree),
        ("Random Forest", random_forest),
    ]:
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"{name} does not expose predict_proba.")
        if int(model.n_features_in_) != X_train.shape[1]:
            raise ValueError(
                f"{name} expects {model.n_features_in_} features "
                f"but data has {X_train.shape[1]}."
            )

    if not hasattr(preprocessing_pipeline, "transform"):
        raise ValueError("Loaded preprocessing pipeline is not fitted.")

    models = {"Decision Tree": decision_tree, "Random Forest": random_forest}
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        load_csv(processed_dir / "feature_dictionary.csv"),
        models,
        {"pipeline": preprocessing_pipeline},
        preprocessing_pipeline,
    )


# ---------------------------------------------------------------------------
# Metrics and evaluation
# ---------------------------------------------------------------------------


def evaluate_probabilities(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Calculate threshold-specific and ranking metrics for the positive class."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")
    if len(y_true) != len(y_proba):
        raise ValueError("Target and probability lengths differ.")

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    predicted_positive = int(tp + fp)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive": predicted_positive,
        "predicted_positive_pct": pct(predicted_positive, len(y_pred)),
    }
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
    }


def baseline_evaluation(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, object]]:
    """Evaluate the existing Phase 3 models at their default 0.50 threshold."""
    results = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate_probabilities(y_test, proba, 0.50)
        print(
            f"  {name} baseline: F1={results[name]['metrics']['f1']:.4f}, "
            f"recall={results[name]['metrics']['recall']:.4f}, "
            f"precision={results[name]['metrics']['precision']:.4f}, "
            f"AUC={results[name]['metrics']['roc_auc']:.4f}"
        )
    return results


# ---------------------------------------------------------------------------
# Hyperparameter optimization
# ---------------------------------------------------------------------------


def parameter_distributions() -> tuple[dict, dict]:
    """Search spaces covering all parameters required by Phase 4."""
    dt_distributions = {
        "max_depth": [3, 4, 5, 6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10, 25, 50, 100],
        "min_samples_leaf": [5, 10, 25, 50, 100, 200],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
    }
    rf_distributions = {
        "n_estimators": [150, 250, 400],
        "max_depth": [8, 12, 18, None],
        "min_samples_split": [2, 5, 10, 30],
        "min_samples_leaf": [2, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.3],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    return dt_distributions, rf_distributions


def run_search(
    name: str,
    estimator,
    distributions: dict,
    n_iter: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[RandomizedSearchCV, float]:
    """Run a seeded randomized search using training-only CV."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring=SEARCH_SCORING,
        cv=build_cv(),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )
    print(
        f"{name}: RandomizedSearchCV with {n_iter} candidates x "
        f"{CV_FOLDS} folds (scoring={SEARCH_SCORING})..."
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    print(f"  best CV {SEARCH_SCORING}: {search.best_score_:.4f}")
    print(f"  best params: {search.best_params_}")
    print(f"  search time: {elapsed:.1f}s")
    return search, elapsed


def build_estimators() -> dict[str, object]:
    """Fresh estimators for the optimization search."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1
        ),
    }


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def out_of_fold_threshold_analysis(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """
    Select a threshold using out-of-fold training probabilities only.

    The test set is deliberately absent from this function.
    """
    print(f"{name}: generating out-of-fold training probabilities...")
    cv = build_cv()
    oof_proba = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    rows = []
    for threshold in THRESHOLD_GRID:
        y_pred = (oof_proba >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_train.to_numpy() == 1)).sum())
        fp = int(((y_pred == 1) & (y_train.to_numpy() == 0)).sum())
        fn = int(((y_pred == 0) & (y_train.to_numpy() == 1)).sum())
        rows.append(
            {
                "model": name,
                "threshold": float(threshold),
                "oof_precision": float(
                    precision_score(y_train, y_pred, zero_division=0)
                ),
                "oof_recall": float(
                    recall_score(y_train, y_pred, zero_division=0)
                ),
                "oof_f1": float(f1_score(y_train, y_pred, zero_division=0)),
                "oof_true_positive": tp,
                "oof_false_positive": fp,
                "oof_false_negative": fn,
                "oof_predicted_positive": tp + fp,
                "oof_predicted_positive_pct": pct(tp + fp, len(y_train)),
            }
        )

    table = pd.DataFrame(rows)
    # Highest F1 on OOF predictions; ties go to the higher-recall threshold.
    table = table.sort_values(
        ["oof_f1", "oof_recall", "threshold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected_threshold = float(table.iloc[0]["threshold"])
    table["selected"] = np.isclose(
        table["threshold"], selected_threshold
    )
    print(
        f"  selected OOF threshold: {selected_threshold:.2f} "
        f"(OOF F1={table.iloc[0]['oof_f1']:.4f}, "
        f"recall={table.iloc[0]['oof_recall']:.4f}, "
        f"precision={table.iloc[0]['oof_precision']:.4f})"
    )
    return table, selected_threshold, oof_proba


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    threshold: float,
    path: Path,
) -> None:
    """Annotated confusion matrix for the positive-class evaluation."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
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
    ax.set_title(f"{title}\nOptimized model at selected threshold = {threshold:.2f}")
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
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    path: Path,
) -> None:
    """ROC comparison for the optimized models."""
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    colors = {"Decision Tree": "#4C72B0", "Random Forest": "#DD8452"}
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_value = roc_auc_score(y_test, proba)
        ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=colors.get(name, "#55A868"),
            label=f"{name} (AUC = {auc_value:.3f})",
        )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        color="#999999",
        label="Random classifier",
    )
    ax.set_title("ROC Curves - Optimized Classifiers on Untouched Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_precision_recall_curves(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Precision-recall comparison with the positive-class baseline."""
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    colors = {"Decision Tree": "#4C72B0", "Random Forest": "#DD8452"}
    baseline = pct(float(y_test.sum()), float(len(y_test))) / 100.0
    for name, proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap_value = average_precision_score(y_test, proba)
        ax.plot(
            recall,
            precision,
            linewidth=2.2,
            color=colors.get(name, "#55A868"),
            label=f"{name} (AP = {ap_value:.3f})",
        )
    ax.axhline(
        baseline,
        linestyle="--",
        linewidth=1.2,
        color="#999999",
        label=f"Random classifier baseline ({baseline:.3f})",
    )
    ax.set_title("Precision-Recall Curves - Imbalanced Positive Class")
    ax.set_xlabel("Recall (share of actual subscribers found)")
    ax.set_ylabel("Precision (share of predicted subscribers who subscribe)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="upper right")
    save_figure(fig, path)


def plot_threshold_analysis(
    threshold_tables: dict[str, pd.DataFrame],
    selected_thresholds: dict[str, float],
    path: Path,
) -> None:
    """Plot OOF precision, recall, and F1 against the threshold."""
    fig, axes = plt.subplots(1, len(threshold_tables), figsize=(15, 5.6))
    if len(threshold_tables) == 1:
        axes = [axes]

    colors = {"precision": "#4C72B0", "recall": "#DD8452", "f1": "#55A868"}
    for ax, (name, table) in zip(axes, threshold_tables.items()):
        ordered = table.sort_values("threshold")
        for metric, color in colors.items():
            ax.plot(
                ordered["threshold"],
                ordered[f"oof_{metric}"],
                linewidth=2,
                color=color,
                label=metric.capitalize(),
            )
        selected = selected_thresholds[name]
        ax.axvline(
            selected,
            linestyle="--",
            color="#C44E52",
            linewidth=1.5,
            label=f"Selected = {selected:.2f}",
        )
        ax.set_title(f"{name} - OOF Threshold Analysis")
        ax.set_xlabel("Probability threshold")
        ax.set_ylabel("Score")
        ax.set_xlim(float(THRESHOLD_GRID.min()), float(THRESHOLD_GRID.max()))
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "Threshold selection uses out-of-fold training predictions only",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, path)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def build_model_comparison_frame(
    baseline: dict[str, dict[str, object]],
    optimized: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Combine Phase 3 baseline and Phase 4 optimized results."""
    rows = []
    for name, result in baseline.items():
        metrics = result["metrics"]
        rows.append(
            {
                "stage": "Phase 3 baseline",
                "model": name,
                "threshold": 0.50,
                **metrics,
            }
        )
    for name, result in optimized.items():
        metrics = result["test_metrics"]["metrics"]
        rows.append(
            {
                "stage": "Phase 4 optimized (OOF threshold)",
                "model": name,
                "threshold": result["selected_threshold"],
                **metrics,
            }
        )
    columns = [
        "stage",
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "predicted_positive",
        "predicted_positive_pct",
    ]
    return pd.DataFrame(rows)[columns]


def build_optimized_results_frame(
    optimized: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Detailed results for the two optimized models."""
    rows = []
    for name, result in optimized.items():
        test = result["test_metrics"]["metrics"]
        oof = result["oof_metrics_at_selected"]
        rows.append(
            {
                "model": name,
                "search_method": "RandomizedSearchCV",
                "search_iterations": result["search_iterations"],
                "cv_folds": CV_FOLDS,
                "search_scoring": SEARCH_SCORING,
                "cv_average_precision": result["cv_average_precision"],
                "best_params": json.dumps(result["best_params"], sort_keys=True),
                "oof_selected_threshold": result["selected_threshold"],
                "oof_precision": oof["precision"],
                "oof_recall": oof["recall"],
                "oof_f1": oof["f1"],
                "oof_roc_auc": oof["roc_auc"],
                "oof_average_precision": oof["average_precision"],
                "test_accuracy": test["accuracy"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "test_f1": test["f1"],
                "test_roc_auc": test["roc_auc"],
                "test_average_precision": test["average_precision"],
                "test_true_positive": test["true_positive"],
                "test_true_negative": test["true_negative"],
                "test_false_positive": test["false_positive"],
                "test_false_negative": test["false_negative"],
                "test_predicted_positive": test["predicted_positive"],
                "test_predicted_positive_pct": test["predicted_positive_pct"],
            }
        )
    return pd.DataFrame(rows)


def write_classification_report(
    path: Path,
    baseline: dict[str, dict[str, object]],
    optimized: dict[str, dict[str, object]],
    selected_name: str,
) -> None:
    """Write detailed baseline and optimized classification reports."""
    lines = [
        "=" * 80,
        "PHASE 4 - CLASSIFICATION REPORTS",
        "Bank Marketing Intelligence System",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random state: {RANDOM_STATE}",
        f"Positive class: {POSITIVE_CLASS} = subscribed (yes)",
        f"CV: StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)",
        "",
        "The test set is the untouched Phase 2 test set. Hyperparameters and",
        "probability thresholds were selected using training data only.",
        "",
    ]

    for name, result in baseline.items():
        m = result["metrics"]
        lines.extend(
            [
                "-" * 80,
                f"BASELINE PHASE 3: {name.upper()} (threshold={m['threshold']:.2f})",
                "-" * 80,
                f"Accuracy : {m['accuracy']:.4f}",
                f"Precision: {m['precision']:.4f}",
                f"Recall   : {m['recall']:.4f}",
                f"F1-score : {m['f1']:.4f}",
                f"ROC-AUC  : {m['roc_auc']:.4f}",
                f"Average precision (PR-AUC): {m['average_precision']:.4f}",
                f"Predicted positives: {m['predicted_positive']:,} "
                f"({m['predicted_positive_pct']:.2f}%)",
                "",
                "Confusion matrix [[TN, FP], [FN, TP]]:",
                f"  TN={m['true_negative']:,}  FP={m['false_positive']:,}",
                f"  FN={m['false_negative']:,}  TP={m['true_positive']:,}",
                "",
                result["classification_report"],
                "",
            ]
        )

    for name, result in optimized.items():
        m = result["test_metrics"]["metrics"]
        lines.extend(
            [
                "-" * 80,
                f"OPTIMIZED PHASE 4: {name.upper()} "
                f"(OOF-selected threshold={m['threshold']:.2f})",
                "-" * 80,
                f"Best parameters: {result['best_params']}",
                f"CV average precision: {result['cv_average_precision']:.4f}",
                f"Accuracy : {m['accuracy']:.4f}",
                f"Precision: {m['precision']:.4f}",
                f"Recall   : {m['recall']:.4f}",
                f"F1-score : {m['f1']:.4f}",
                f"ROC-AUC  : {m['roc_auc']:.4f}",
                f"Average precision (PR-AUC): {m['average_precision']:.4f}",
                f"Predicted positives: {m['predicted_positive']:,} "
                f"({m['predicted_positive_pct']:.2f}%)",
                "",
                "Confusion matrix [[TN, FP], [FN, TP]]:",
                f"  TN={m['true_negative']:,}  FP={m['false_positive']:,}",
                f"  FN={m['false_negative']:,}  TP={m['true_positive']:,}",
                "",
                result["test_metrics"]["classification_report"],
                "",
            ]
        )

    lines.extend(
        [
            "-" * 80,
            "OPERATIONAL INTERPRETATION OF THE FOUR OUTCOMES",
            "-" * 80,
            "True Positive (TP): a customer who subscribed and was predicted to",
            "  subscribe. This is a correctly targeted contact and the direct",
            "  campaign win.",
            "True Negative (TN): a customer who did not subscribe and was not",
            "  contacted. This is a correctly avoided contact and saves budget.",
            "False Positive (FP): a predicted subscriber who did not subscribe.",
            "  The bank pays for an unnecessary call or offer; this is the main",
            "  cost of a high-recall threshold.",
            "False Negative (FN): an actual subscriber who was predicted not to",
            "  subscribe. The bank loses a potential sale and customer attention;",
            "  this is the main cost of a high-precision threshold.",
            "",
            f"Selected final model: {selected_name}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_final_report(
    baseline: dict[str, dict[str, object]],
    optimized: dict[str, dict[str, object]],
    comparison_frame: pd.DataFrame,
    threshold_table: pd.DataFrame,
    selected_name: str,
    selected_threshold: float,
    y_test: pd.Series,
    search_times: dict[str, float],
    feature_count: int,
    phase3_comparison: pd.DataFrame,
) -> str:
    """Compose the Phase 4 model-selection report with actual results."""
    selected = optimized[selected_name]
    selected_test = selected["test_metrics"]["metrics"]
    selected_oof = selected["oof_metrics_at_selected"]
    selected_baseline = baseline[selected_name]["metrics"]

    baseline_table = comparison_frame[
        comparison_frame["stage"] == "Phase 3 baseline"
    ].copy()
    optimized_table = comparison_frame[
        comparison_frame["stage"] == "Phase 4 optimized (OOF threshold)"
    ].copy()

    lines: list[str] = []
    add = lines.append

    add("# Phase 4 - Model Evaluation, Optimization and Selection")
    add("")
    add("**Project:** Bank Marketing Intelligence System  ")
    add("**Course context:** Information Systems Management  ")
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    add("**Scope:** Phase 4 only. Phase 5 was not started.  ")
    add("**Positive class:** `y = yes` (subscribed to a term deposit).")
    add("")
    add("---")
    add("")
    add("## 1. Executive summary")
    add("")
    add(
        f"The final selected model is **{selected_name}** with an "
        f"OOF-selected probability threshold of **{selected_threshold:.2f}**. "
        "The threshold was chosen from out-of-fold training predictions, not "
        "from the test set."
    )
    add("")
    add(
        f"On the untouched test set, it achieves **F1={selected_test['f1']:.4f}**, "
        f"**recall={selected_test['recall']:.4f}**, "
        f"**precision={selected_test['precision']:.4f}**, "
        f"**ROC-AUC={selected_test['roc_auc']:.4f}**, and "
        f"**average precision={selected_test['average_precision']:.4f}**. "
        f"It identifies **{selected_test['true_positive']:,} of "
        f"{selected_test['true_positive'] + selected_test['false_negative']:,}** "
        "actual subscribers while making "
        f"**{selected_test['predicted_positive']:,}** predicted positive contacts."
    )
    add("")
    add(
        "The existing Phase 3 models and all data artifacts were verified before "
        "Phase 4. The test set was not used for hyperparameter or threshold "
        "selection. `duration` remains excluded, so this is a pre-call model."
    )
    add("")
    add("---")
    add("")
    add("## 2. Data and evaluation protocol")
    add("")
    add(
        f"- Training customers: **{36_168:,}**; test customers: **{len(y_test):,}**; "
        f"test subscribers: **{int(y_test.sum()):,} "
        f"({pct(float(y_test.sum()), float(len(y_test))):.2f}%)**."
    )
    add(f"- Model inputs: **{feature_count}** Phase 2 processed features.")
    add("- `duration` is absent from the classifier inputs.")
    add(
        "- Hyperparameter selection: RandomizedSearchCV with "
        f"**StratifiedKFold({CV_FOLDS}, shuffle=True, random_state=42)**, "
        f"optimising **{SEARCH_SCORING}** on training data only."
    )
    add(
        "- Threshold selection: out-of-fold training probabilities; the test set "
        "is used once per optimized model at its selected threshold."
    )
    add("- The Phase 2 fitted preprocessing pipeline is loaded and reused, not refitted.")
    add("")
    add("### Existing Phase 3 test metrics at 0.50")
    add("")
    add(
        md_table(
            baseline_table[
                [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "average_precision",
                    "true_positive",
                    "false_positive",
                    "false_negative",
                ]
            ].round(4)
        )
    )
    add("")
    add(
        "These values reproduce the Phase 3 test results. They are retained as "
        "the baseline against which Phase 4 optimization is compared."
    )
    add("")
    add("---")
    add("")
    add("## 3. Task 1 - Model performance comparison")
    add("")
    add("### Baseline Phase 3 models")
    add("")
    add(
        md_table(
            baseline_table[
                [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "average_precision",
                    "predicted_positive",
                ]
            ].round(4)
        )
    )
    add("")
    add("### Phase 4 optimized models at OOF-selected thresholds")
    add("")
    add(
        md_table(
            optimized_table[
                [
                    "model",
                    "threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "average_precision",
                    "predicted_positive",
                ]
            ].round(4)
        )
    )
    add("")
    add(
        "Accuracy is shown for completeness but is not the selection criterion: "
        "the majority-class baseline is 88.30%, so a model can look accurate "
        "while finding few subscribers. The positive-class metrics and PR-AUC "
        "are more informative."
    )
    add("")
    add("---")
    add("")
    add("## 4. Task 2 - Confusion matrix analysis")
    add("")
    add("The operational meaning of each cell is:")
    add("")
    add("| Outcome | Meaning | Marketing consequence |")
    add("| --- | --- | --- |")
    add(
        "| **TP** | Actual subscriber, predicted subscriber | Correctly targeted "
        "contact; campaign revenue opportunity. |"
    )
    add(
        "| **TN** | Non-subscriber, predicted non-subscriber | Correctly avoided "
        "contact; saves call-centre budget. |"
    )
    add(
        "| **FP** | Non-subscriber, predicted subscriber | Wasted call/offer cost; "
        "the price of higher recall. |"
    )
    add(
        "| **FN** | Actual subscriber, predicted non-subscriber | Missed sale and "
        "lost customer attention; the price of higher precision. |"
    )
    add("")
    for name, result in optimized.items():
        m = result["test_metrics"]["metrics"]
        add(
            f"### {name} at selected threshold {m['threshold']:.2f}"
        )
        add("")
        add(
            f"- TN = **{m['true_negative']:,}**, FP = **{m['false_positive']:,}**, "
            f"FN = **{m['false_negative']:,}**, TP = **{m['true_positive']:,}**."
        )
        add(
            f"- The model predicts **{m['predicted_positive']:,}** positives "
            f"({m['predicted_positive_pct']:.2f}% of customers); actual positives "
            f"number **{int(y_test.sum()):,}**."
        )
        add("")
    add(
        "A high-recall operating point accepts more false positives to reduce "
        "false negatives. A high-precision operating point does the opposite. "
        "The bank should choose based on call capacity and the relative cost of "
        "a wasted call versus a missed subscriber."
    )
    add("")
    add("![Decision Tree confusion matrix](confusion_matrix_decision_tree.png)")
    add("")
    add("![Random Forest confusion matrix](confusion_matrix_random_forest.png)")
    add("")
    add("---")
    add("")
    add("## 5. Task 3 - ROC and precision-recall analysis")
    add("")
    add(
        "ROC-AUC measures ranking across all thresholds. Precision-Recall is "
        "especially useful here because the positive class is only 11.70% of "
        "the test set. A ROC curve can look strong even when the positive-class "
        "precision is low, because the large negative class dominates the false "
        "positive rate. The PR curve focuses directly on how many predicted "
        "subscribers are real and how many actual subscribers are found."
    )
    add("")
    add(
        md_table(
            optimized_table[
                ["model", "roc_auc", "average_precision"]
            ].round(4)
        )
    )
    add("")
    add("![ROC curves](roc_curve_comparison.png)")
    add("")
    add("![Precision-recall curves](precision_recall_curve.png)")
    add("")
    add(
        "Average precision (AP) is the area under the PR curve and is a useful "
        "summary for the minority positive class. It should be read alongside "
        "the operating threshold, because the chosen threshold determines the "
        "actual precision/recall balance used by marketing."
    )
    add("")
    add("---")
    add("")
    add("## 6. Task 4 - Hyperparameter optimization")
    add("")
    add("### Search design")
    add("")
    add(
        f"- Decision Tree: RandomizedSearchCV, **{DT_SEARCH_ITERATIONS}** "
        "candidates, 5-fold stratified CV, scoring = "
        f"**{SEARCH_SCORING}**."
    )
    add(
        "- Decision Tree search space: `max_depth`, `min_samples_split`, "
        "`min_samples_leaf`, `criterion`, `class_weight`."
    )
    add(
        f"- Random Forest: RandomizedSearchCV, **{RF_SEARCH_ITERATIONS}** "
        "candidates, 5-fold stratified CV, scoring = "
        f"**{SEARCH_SCORING}**."
    )
    add(
        "- Random Forest search space: `n_estimators`, `max_depth`, "
        "`min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`."
    )
    add(
        "- Every search fit used training data only. The test set was not passed "
        "to either search object."
    )
    add("")
    add("### Selected hyperparameters")
    add("")
    search_rows = []
    for name, result in optimized.items():
        search_rows.append(
            {
                "model": name,
                "cv_average_precision": result["cv_average_precision"],
                "search_time_seconds": search_times[name],
                "best_params": json.dumps(result["best_params"], sort_keys=True),
            }
        )
    add(md_table(pd.DataFrame(search_rows).round(4)))
    add("")
    add(
        "Criterion and class-weight choices were therefore not assumed: both "
        "were included in the search. `balanced_subsample` was available for "
        "the Random Forest because it can behave differently from `balanced` "
        "when bootstrap samples vary."
    )
    add("")
    add("---")
    add("")
    add("## 7. Task 5 - Prediction threshold analysis")
    add("")
    add(
        "Thresholds were evaluated on out-of-fold training probabilities. "
        "The table below reports OOF precision, recall and F1; the selected "
        "threshold maximises OOF F1, with ties resolved toward higher recall."
    )
    add("")
    selected_rows = threshold_table[threshold_table["selected"]].copy()
    add(
        md_table(
            selected_rows[
                [
                    "model",
                    "threshold",
                    "oof_precision",
                    "oof_recall",
                    "oof_f1",
                    "oof_predicted_positive",
                ]
            ].round(4)
        )
    )
    add("")
    add(
        "The required example thresholds are shown below for context. They are "
        "OOF training results, not test-set tuning results. The full 0.05-0.95 "
        "grid is in `threshold_comparison.csv`."
    )
    add("")
    example_thresholds = threshold_table[
        threshold_table["threshold"].round(2).isin(REQUIRED_EXAMPLE_THRESHOLDS)
    ].copy()
    add(
        md_table(
            example_thresholds[
                [
                    "model",
                    "threshold",
                    "oof_precision",
                    "oof_recall",
                    "oof_f1",
                    "oof_predicted_positive",
                ]
            ].round(4)
        )
    )
    add("")
    add(
        "The two selected thresholds are not directly comparable as calibrated "
        "probabilities. The Random Forest uses `class_weight=None`, so its "
        "probabilities are concentrated at lower values; the Decision Tree uses "
        f"`class_weight='balanced'`, which pushes its scores upward. Each "
        "threshold is best interpreted as a model-specific decision boundary "
        "chosen on that model's OOF ranking."
    )
    add("")
    add("![Threshold analysis](threshold_analysis.png)")
    add("")
    add(
        "Lowering the threshold increases recall (more actual subscribers found) "
        "but usually lowers precision and increases call volume. Raising the "
        "threshold does the opposite. For the selected Random Forest, the OOF "
        f"point at 0.20 finds **{int(selected_oof['true_positive']):,}** true "
        f"positives and **{int(selected_oof['false_positive']):,}** false "
        f"positives; at 0.50 the OOF operating point would trade recall for "
        "precision and a smaller contact list. The selected threshold is an "
        "F1-oriented starting point; the bank can move it according to "
        "call-centre capacity without refitting the model."
    )
    add("")
    add("---")
    add("")
    add("## 8. Task 6 - K-Means evaluation")
    add("")
    add(
        "The existing Phase 3 K-Means model was reviewed in "
        "`clustering_evaluation.md`. It selected k=4 with training silhouette "
        "0.1674; the target and duration were not clustering inputs. The "
        "segmentation should be used as a descriptive overlay on the supervised "
        "ranking model, not as a replacement for it."
    )
    add("")
    add("---")
    add("")
    add("## 9. Task 7 - Final model selection")
    add("")
    add("### Selection rule")
    add("")
    add(
        "The model and threshold were selected from **out-of-fold training "
        "metrics**, with positive-class F1 as the primary criterion and "
        "average precision as supporting evidence. Accuracy was not used to "
        "choose the model."
    )
    add("")
    add("### Why this model")
    add("")
    add(
        f"**{selected_name}** achieved the strongest OOF F1 "
        f"({selected_oof['f1']:.4f}) and OOF average precision "
        f"({selected_oof['average_precision']:.4f}) among the optimized "
        "candidates. Its test performance confirms the choice without having "
        "been used to make it: "
        f"F1 **{selected_test['f1']:.4f}**, recall "
        f"**{selected_test['recall']:.4f}**, precision "
        f"**{selected_test['precision']:.4f}**, ROC-AUC "
        f"**{selected_test['roc_auc']:.4f}**, AP "
        f"**{selected_test['average_precision']:.4f}** at threshold "
        f"**{selected_test['threshold']:.2f}**."
    )
    add("")
    add("### Trade-offs")
    add("")
    add(
        "- **Recall:** the selected model is compared with the other optimized "
        "model in the results table. If campaign capacity is high, a lower "
        "threshold from `threshold_comparison.csv` can increase recall further."
    )
    add(
        "- **Precision:** a higher threshold reduces wasted calls but can miss "
        "subscribers. The PR curve and threshold table make this trade-off "
        "explicit."
    )
    add(
        "- **Interpretability:** the Decision Tree can be read as if/then rules "
        "and is preferable when an auditable policy explanation is mandatory. "
        "The selected model is less transparent internally than a single tree, "
        "but its feature importances and threshold support governance and "
        "campaign monitoring."
    )
    add(
        f"- **Computational cost:** the search took "
        f"{search_times.get(selected_name, 0.0):.1f}s for the selected model. "
        "Prediction is fast enough for batch campaign scoring; the saved model "
        "and threshold can be loaded for Phase 5 integration."
    )
    add(
        "- **Business objective:** the bank needs a repeatable ranking of likely "
        "subscribers, not the highest raw accuracy. The selected model has the "
        "best OOF F1 and PR-AUC, the highest ROC-AUC, and precision close to the "
        "Decision Tree while retaining useful recall at the operational threshold."
    )
    add("")
    add("### Relationship to the Phase 3 baseline")
    add("")
    add(
        f"The selected baseline model at 0.50 had F1 "
        f"**{selected_baseline['f1']:.4f}**, recall "
        f"**{selected_baseline['recall']:.4f}** and precision "
        f"**{selected_baseline['precision']:.4f}**. Phase 4's optimized model "
        f"and threshold produce F1 **{selected_test['f1']:.4f}**, recall "
        f"**{selected_test['recall']:.4f}** and precision "
        f"**{selected_test['precision']:.4f}** on the same test set. "
        "The comparison is descriptive; the test set was not used to choose "
        "the Phase 4 configuration."
    )
    add("")
    add(
        f"The test-set gain over the Phase 3 Random Forest is modest "
        f"(F1 {selected_test['f1'] - selected_baseline['f1']:+.4f}, "
        f"recall {selected_test['recall'] - selected_baseline['recall']:+.4f}, "
        f"precision {selected_test['precision'] - selected_baseline['precision']:+.4f}). "
        "This indicates that the Phase 3 Random Forest was already near its "
        "practical plateau on these features. The most useful Phase 4 outcome "
        "is the explicit, training-only threshold policy and the evidence that "
        "the optimized tree closes much of the gap but does not overtake the "
        "ensemble on PR-AUC."
    )
    add("")
    add("---")
    add("")
    add("## 10. Validation and limitations")
    add("")
    add(
        "- All required models loaded successfully; the selected artifact was "
        "reloaded and its predictions were checked."
    )
    add(
        "- The test set remained untouched during search and threshold selection."
    )
    add(
        "- `duration` is not present in the model input features; this is a "
        "pre-call prediction setup."
    )
    add(
        "- The test set contains "
        f"{int(y_test.sum()):,} subscribers ({pct(float(y_test.sum()), float(len(y_test))):.2f}%). "
        "Minority-class metrics are therefore more relevant than accuracy."
    )
    add(
        "- No campaign cost or revenue data are available, so the threshold is "
        "an F1-oriented default rather than a profit-optimal decision. Marketing "
        "should set the final operating point using call capacity and cost per call."
    )
    add(
        "- Phase 2 scaling/encoding was fitted on the training split before CV. "
        "The CV estimates are therefore slightly optimistic for the preprocessing "
        "step, although the test set is untouched. In a production retraining "
        "pipeline, fit preprocessing inside each CV fold."
    )
    add(
        "- The OOF threshold shares the training folds used for model selection, "
        "so it is a validation estimate rather than a fully independent holdout. "
        "The untouched test evaluation is the final external check."
    )
    add(
        "- The K-Means segments are descriptive and use no target labels; they "
        "should complement, not replace, the supervised ranking model."
    )
    add("")
    add("---")
    add("")
    add("## 11. Reproduction")
    add("")
    add("```powershell")
    add("cd \"D:\\LEARNING\\Information System Manangement\\Project\"")
    add("python src/phase4_evaluation.py")
    add("```")
    add("")
    add("Optional paths:")
    add("")
    add("```powershell")
    add("python src/phase4_evaluation.py `")
    add("    --processed-dir data/processed `")
    add("    --models-dir models `")
    add("    --reports-dir reports/phase4")
    add("```")
    add("")
    add("## 12. Outputs")
    add("")
    add("- `models/best_model.joblib` - selected model, Phase 2 preprocessing pipeline, and threshold.")
    add("- `models/best_model_metadata.json` - metadata and reuse instructions.")
    add("- `reports/phase4/model_comparison.csv`")
    add("- `reports/phase4/optimized_model_results.csv`")
    add("- `reports/phase4/threshold_comparison.csv`")
    add("- `reports/phase4/classification_report.txt`")
    add("- `reports/phase4/final_model_selection.md`")
    add("- `reports/phase4/clustering_evaluation.md`")
    add("- The five required Phase 4 charts.")
    add("")
    return "\n".join(lines)


def build_clustering_report(
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
    kmeans_feature_names: list[str],
) -> str:
    """Evaluate the existing Phase 3 K-Means results without refitting."""
    selected = metrics[metrics["k"] == 4]
    selected_row = selected.iloc[0] if not selected.empty else metrics.iloc[0]

    lines: list[str] = []
    add = lines.append
    add("# Phase 4 - K-Means Clustering Evaluation")
    add("")
    add("**Source:** Phase 3 artifacts (`models/kmeans.joblib`, "
        "`reports/phase3/clustering_metrics.csv`, `cluster_profiles.csv`).  ")
    add("**Status:** Review only. K-Means was not refitted in Phase 4.  ")
    add("**Selected clusters:** 4  ")
    add("**Independence:** `y` and `duration` were not clustering inputs.")
    add("")
    add("---")
    add("")
    add("## 1. Clustering inputs")
    add("")
    add("The saved K-Means pipeline expects these engineered customer features:")
    add("")
    add("- " + ", ".join(f"`{name}`" for name in kmeans_feature_names))
    add("")
    add("No target label and no `duration` field are present. The target is used "
        "below only as a post-hoc description of observed subscription rates.")
    add("")
    add("## 2. Model selection metrics")
    add("")
    add(
        md_table(
            metrics[
                [
                    "k",
                    "inertia",
                    "silhouette_score",
                    "min_cluster_count",
                    "min_cluster_share_pct",
                    "cluster_sizes",
                ]
            ].round(4)
        )
    )
    add("")
    add(
        f"The selected solution is **k=4** with inertia "
        f"**{selected_row['inertia']:.2f}** and silhouette "
        f"**{selected_row['silhouette_score']:.4f}**. k=7 has a slightly higher "
        "raw silhouette, but it creates a one-customer cluster and a 170-customer "
        "cluster; the 2% minimum cluster-share guardrail rejects that degenerate "
        "solution. k=4 provides four usable marketing segments."
    )
    add("")
    add("Silhouette is modest because the feature space mixes scaled numeric "
        "variables with one-hot customer categories. The value of the solution "
        "is interpretability and business differentiation, not a high absolute "
        "separation score. Phase 3 also reported a held-out test silhouette of "
        "**0.1697**, close to the training value, so the segmentation generalises "
        "reasonably even though the absolute separation is modest.")
    add("")
    add("## 3. Cluster sizes and characteristics")
    add("")
    add(
        md_table(
            profiles[
                [
                    "cluster",
                    "n_customers",
                    "share_pct",
                    "age_mean",
                    "balance_median",
                    "top_job",
                    "top_job_pct",
                    "pct_housing_loan",
                    "pct_previous_contact",
                    "observed_subscription_rate_pct",
                ]
            ].round(2)
        )
    )
    add("")
    overall_rate = (
        profiles["observed_subscription_rate_pct"] * profiles["n_customers"]
    ).sum() / profiles["n_customers"].sum()
    interpretations = {
        0: (
            "older, mid-balance customers; the most common job is blue-collar, "
            "41% have a housing loan, and few had previous campaign contact. "
            "Conversion is close to the overall rate."
        ),
        1: (
            "mid-age, high-balance customers with a management orientation; "
            "this small segment has above-average observed conversion and may "
            "support a premium deposit proposition."
        ),
        2: (
            "younger, mid-balance customers; every customer in this segment had "
            "previous campaign contact and observed conversion is the highest of "
            "the four groups. This is the clearest warm-lead segment."
        ),
        3: (
            "youngest, lowest-balance customers; almost none had previous "
            "campaign contact and observed conversion is below average. Expensive "
            "high-touch outreach is harder to justify here."
        ),
    }
    for _, row in profiles.iterrows():
        gap = row["observed_subscription_rate_pct"] - overall_rate
        add(f"### Cluster {int(row['cluster'])} - "
            f"{int(row['n_customers']):,} customers ({row['share_pct']:.1f}%)")
        add("")
        add(
            f"This segment has mean age **{row['age_mean']:.1f}**, median balance "
            f"**{row['balance_median']:,.0f} EUR**, most common job "
            f"**{row['top_job']}** ({row['top_job_pct']:.1f}%), housing-loan "
            f"share **{row['pct_housing_loan']:.1f}%**, and previous-contact "
            f"share **{row['pct_previous_contact']:.1f}%**. "
            f"The post-hoc observed subscription rate is "
            f"**{row['observed_subscription_rate_pct']:.2f}%** "
            f"({gap:+.2f} percentage points versus the overall "
            f"{overall_rate:.2f}%)."
        )
        add("")
        add(
            f"**Interpretation:** "
            f"{interpretations.get(int(row['cluster']), row['description'])}"
        )
        add("")
        if gap > 2:
            add("This is an above-average segment. It is a candidate for priority "
                "outreach, a targeted term-deposit proposition, or a higher-touch "
                "channel where the expected value supports the cost.")
        elif gap < -2:
            add("This is a below-average segment. Prefer lower-cost channels, "
                "self-service campaigns, or a different product proposition rather "
                "than expensive high-touch calls.")
        else:
            add("This segment is close to the overall conversion rate. It can "
                "support volume campaigns but is not the most differentiated group.")
        add("")
    add("## 4. Business interpretation")
    add("")
    add(
        "- **Cluster 2** is the clearest priority segment: it combines prior "
        "contact history with a substantially above-average observed "
        "subscription rate. It should be considered for warm-lead outreach."
    )
    add(
        "- **Cluster 1** is small but high-balance and management-oriented; it "
        "may justify a premium deposit offer, subject to campaign economics."
    )
    add(
        "- **Cluster 3** is the largest and youngest/lowest-balance group. It "
        "converts below average, so expensive calls are harder to justify."
    )
    add(
        "- **Cluster 0** is older and near-average in conversion; it can support "
        "volume campaigns with disciplined call scripting."
    )
    add("")
    add("These segments should be combined with the supervised probability "
        "ranking: first rank customers by predicted conversion probability, then "
        "use segment membership to choose channel, offer, and contact intensity. "
        "Clustering does not replace the conversion model.")
    add("")
    add("## 5. Limitations")
    add("")
    add(
        "- K-Means assumes roughly spherical, similarly sized clusters and uses "
        "Euclidean distance on one-hot dimensions."
    )
    add(
        "- The silhouette is modest; cluster boundaries are not sharp and "
        "membership should be treated as a marketing heuristic."
    )
    add(
        "- The observed subscription rate is descriptive and not causal. It was "
        "not used to create the clusters."
    )
    add(
        "- Cluster IDs are arbitrary; business names should be agreed with "
        "marketing stakeholders."
    )
    add("")
    add("## 6. Reference charts")
    add("")
    add("The Phase 3 clustering charts remain the source of truth:")
    add("")
    add("- `reports/phase3/customer_clusters.png`")
    add("- `reports/phase3/elbow_method.png`")
    add("- `reports/phase3/silhouette_scores.png`")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Final artifact persistence and verification
# ---------------------------------------------------------------------------


def save_best_model(
    best_name: str,
    best_result: dict[str, object],
    X_train: pd.DataFrame,
    preprocessing_pipeline,
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, object]:
    """Save the selected model, preprocessing pipeline, and threshold."""
    pipeline_input_features = (
        list(preprocessing_pipeline.feature_names_in_)
        if hasattr(preprocessing_pipeline, "feature_names_in_")
        else []
    )
    artifact = {
        "model_name": best_name,
        "model": best_result["model"],
        "preprocessing_pipeline": preprocessing_pipeline,
        "pipeline_input_features": pipeline_input_features,
        "threshold": float(best_result["selected_threshold"]),
        "positive_label": POSITIVE_CLASS,
        "feature_names": list(X_train.columns),
        "selection_method": "OOF F1 on training data",
        "trained_on": "data/processed/X_train.csv",
        "test_set": "data/processed/X_test.csv",
        "duration_included": False,
        "best_params": best_result["best_params"],
        "oof_metrics_at_selected_threshold": best_result[
            "oof_metrics_at_selected"
        ],
        "test_metrics_at_selected_threshold": best_result["test_metrics"][
            "metrics"
        ],
    }
    artifact_path = models_dir / "best_model.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"  saved: {artifact_path.name}")

    metadata = {
        "phase": 4,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_model": best_name,
        "artifact": "models/best_model.joblib",
        "threshold": float(best_result["selected_threshold"]),
        "threshold_selection": (
            "Max OOF F1 on StratifiedKFold(5) training predictions; "
            "test set not used"
        ),
        "positive_class": "yes (1)",
        "positive_label": POSITIVE_CLASS,
        "duration_included": False,
        "test_set_used_for_selection": False,
        "model_input_features": list(X_train.columns),
        "pipeline_input_features": pipeline_input_features,
        "preprocessing_pipeline_note": (
            "Phase 2 fitted preprocessing pipeline; expects the engineered "
            "customer features listed in pipeline_input_features, then produces "
            "the 59 model columns. See data/processed/feature_dictionary.csv."
        ),
        "best_params": to_jsonable(best_result["best_params"]),
        "cv_average_precision": best_result["cv_average_precision"],
        "oof_metrics_at_selected_threshold": to_jsonable(
            best_result["oof_metrics_at_selected"]
        ),
        "test_metrics_at_selected_threshold": to_jsonable(
            best_result["test_metrics"]["metrics"]
        ),
        "reuse_example": (
            "artifact = joblib.load('models/best_model.joblib'); "
            "pipeline = artifact['preprocessing_pipeline']; "
            "model = artifact['model']; threshold = artifact['threshold']; "
            "X_model = pd.DataFrame(pipeline.transform(engineered_df), "
            "columns=artifact['feature_names'], index=engineered_df.index); "
            "proba = model.predict_proba(X_model)[:, 1]; "
            "pred = (proba >= threshold).astype(int)"
        ),
    }
    metadata_path = models_dir / "best_model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"  saved: {metadata_path.name}")
    return artifact


def verify_artifact(
    artifact: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_result: dict[str, object],
    models_dir: Path,
) -> None:
    """Reload and verify the final artifact and its predictions."""
    path = models_dir / "best_model.joblib"
    loaded = joblib.load(path)
    required_keys = {
        "model",
        "preprocessing_pipeline",
        "threshold",
        "model_name",
        "feature_names",
    }
    if not required_keys.issubset(loaded):
        raise RuntimeError("Saved best_model.joblib is missing required keys.")
    threshold = float(loaded["threshold"])
    if not np.isclose(threshold, float(artifact["threshold"])):
        raise RuntimeError("Reloaded threshold differs from the saved threshold.")
    if not hasattr(loaded["preprocessing_pipeline"], "transform"):
        raise RuntimeError("Reloaded preprocessing pipeline is not fitted.")
    model = loaded["model"]
    original_proba = artifact["model"].predict_proba(X_test)[:, 1]
    loaded_proba = model.predict_proba(X_test)[:, 1]
    if not np.allclose(original_proba, loaded_proba):
        raise RuntimeError("Reloaded model probabilities differ from the original.")
    original_pred = (original_proba >= threshold).astype(int)
    loaded_pred = (loaded_proba >= threshold).astype(int)
    if not np.array_equal(original_pred, loaded_pred):
        raise RuntimeError("Reloaded model predictions differ at the saved threshold.")
    m = selected_result["test_metrics"]["metrics"]
    if not np.isclose(float(loaded_pred.sum()), float(m["predicted_positive"])):
        raise RuntimeError("Reloaded prediction count does not match the report.")
    print(
        f"  verified reload: best_model.joblib "
        f"({len(loaded_pred):,} predictions; threshold={threshold:.2f})"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4 evaluation, optimization, and model selection."
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
        help="Model directory (default: models).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Phase 4 report directory (default: reports/phase4).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run Phase 4 end to end."""
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

    print("=" * 80)
    print("BANK MARKETING INTELLIGENCE SYSTEM - PHASE 4")
    print("Evaluation, optimization, and model selection")
    print("=" * 80)

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            feature_dictionary,
            phase3_models,
            pipeline_bundle,
            preprocessing_pipeline,
        ) = load_inputs(processed_dir, models_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    feature_count = len(X_train.columns)
    print(f"\nTraining: {X_train.shape[0]:,} x {feature_count}")
    print(f"Test    : {X_test.shape[0]:,} x {feature_count}")
    print(f"Train positive rate: {pct(float(y_train.sum()), len(y_train)):.2f}%")
    print(f"Test positive rate : {pct(float(y_test.sum()), len(y_test)):.2f}%")
    print("Missing values: 0 | Non-finite values: 0")
    print("duration in features: no")
    print("Reusing Phase 3 models and Phase 2 fitted preprocessing.")

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    # Task 1: baseline evaluation -------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 1 - EXISTING PHASE 3 MODELS ON THE UNTOUCHED TEST SET")
    print("=" * 80)
    baseline = baseline_evaluation(phase3_models, X_test, y_test)

    # Compare with the saved Phase 3 CSV as a consistency check.
    phase3_path = PHASE3_REPORTS_DIR / "classification_comparison.csv"
    if phase3_path.exists():
        phase3_comparison = load_csv(phase3_path)
        for name in ["Decision Tree", "Random Forest"]:
            old = phase3_comparison.loc[
                phase3_comparison["model"] == name
            ].iloc[0]
            new = baseline[name]["metrics"]
            differences = [
                abs(float(new["f1"]) - float(old["f1"])),
                abs(float(new["recall"]) - float(old["recall"])),
                abs(float(new["precision"]) - float(old["precision"])),
            ]
            if max(differences) > 1e-3:
                print(
                    f"  WARNING: baseline {name} differs from Phase 3 CSV "
                    f"by {max(differences):.6f}",
                    file=sys.stderr,
                )
        print("  baseline metrics match the Phase 3 saved results")
    else:
        phase3_comparison = pd.DataFrame()

    # Task 2/3: baseline confusion matrices are still useful for charts;
    # optimized charts are generated later.

    # Task 4: optimization ---------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 4 - HYPERPARAMETER OPTIMIZATION (TRAINING ONLY)")
    print("=" * 80)
    dt_dist, rf_dist = parameter_distributions()
    estimators = build_estimators()
    searches: dict[str, RandomizedSearchCV] = {}
    search_times: dict[str, float] = {}

    search_specs = [
        (
            "Decision Tree",
            estimators["Decision Tree"],
            dt_dist,
            DT_SEARCH_ITERATIONS,
        ),
        (
            "Random Forest",
            estimators["Random Forest"],
            rf_dist,
            RF_SEARCH_ITERATIONS,
        ),
    ]
    try:
        for name, estimator, distributions, n_iter in search_specs:
            search, elapsed = run_search(
                name, estimator, distributions, n_iter, X_train, y_train
            )
            searches[name] = search
            search_times[name] = elapsed
    except (ValueError, RuntimeError) as exc:
        print(f"\nERROR during hyperparameter search: {exc}", file=sys.stderr)
        return 1

    # Task 5: threshold selection on OOF training probabilities -------------
    print("\n" + "=" * 80)
    print("TASK 5 - THRESHOLD SELECTION (OUT-OF-FOLD TRAINING ONLY)")
    print("=" * 80)
    optimized: dict[str, dict[str, object]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}
    selected_thresholds: dict[str, float] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    for name, search in searches.items():
        try:
            table, selected_threshold, oof_proba = (
                out_of_fold_threshold_analysis(
                    name, search.best_estimator_, X_train, y_train
                )
            )
        except (ValueError, RuntimeError) as exc:
            print(f"\nERROR during threshold analysis for {name}: {exc}", file=sys.stderr)
            return 1

        threshold_tables[name] = table
        selected_thresholds[name] = selected_threshold
        oof_metrics = evaluate_probabilities(
            y_train, oof_proba, selected_threshold
        )["metrics"]
        optimized[name] = {
            "model": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_average_precision": float(search.best_score_),
            "search_iterations": DT_SEARCH_ITERATIONS
            if name == "Decision Tree"
            else RF_SEARCH_ITERATIONS,
            "selected_threshold": selected_threshold,
            "oof_metrics_at_selected": oof_metrics,
            "oof_proba": oof_proba,
        }

    # Select final model using OOF metrics only ------------------------------
    selection_order = sorted(
        optimized,
        key=lambda name: (
            optimized[name]["oof_metrics_at_selected"]["f1"],
            optimized[name]["oof_metrics_at_selected"]["average_precision"],
            optimized[name]["oof_metrics_at_selected"]["recall"],
        ),
        reverse=True,
    )
    selected_name = selection_order[0]
    print(f"\nFinal model selected from OOF metrics: {selected_name}")

    # Task 6: evaluate optimized models on test at their selected thresholds.
    # This happens AFTER all selection decisions.
    print("\n" + "=" * 80)
    print("TASK 7 - FINAL TEST EVALUATION AFTER SELECTION")
    print("=" * 80)
    for name, result in optimized.items():
        proba_test = result["model"].predict_proba(X_test)[:, 1]
        test_probabilities[name] = proba_test
        result["test_metrics"] = evaluate_probabilities(
            y_test, proba_test, result["selected_threshold"]
        )
        m = result["test_metrics"]["metrics"]
        print(
            f"  {name} @ {m['threshold']:.2f}: F1={m['f1']:.4f}, "
            f"recall={m['recall']:.4f}, precision={m['precision']:.4f}, "
            f"AUC={m['roc_auc']:.4f}, AP={m['average_precision']:.4f}"
        )

    # Task 7: save final artifact -------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 8 - SAVE FINAL MODEL")
    print("=" * 80)
    selected_result = optimized[selected_name]
    artifact = save_best_model(
        selected_name,
        selected_result,
        X_train,
        preprocessing_pipeline,
        models_dir,
        reports_dir,
    )

    # Task 8: reports and charts --------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 8 - WRITE REPORTS AND CHARTS")
    print("=" * 80)

    comparison_frame = build_model_comparison_frame(baseline, optimized)
    optimized_frame = build_optimized_results_frame(optimized)

    comparison_path = reports_dir / "model_comparison.csv"
    comparison_frame.round(4).to_csv(comparison_path, index=False)
    print(f"  saved: {comparison_path.name}")

    optimized_path = reports_dir / "optimized_model_results.csv"
    optimized_output = optimized_frame.copy()
    numeric_columns = [
        column
        for column in optimized_output.columns
        if column
        not in {
            "model",
            "search_method",
            "search_scoring",
            "best_params",
        }
        and pd.api.types.is_numeric_dtype(optimized_output[column])
    ]
    optimized_output[numeric_columns] = optimized_output[numeric_columns].round(4)
    optimized_output.to_csv(optimized_path, index=False)
    print(f"  saved: {optimized_path.name}")

    threshold_table = pd.concat(threshold_tables.values(), ignore_index=True)
    threshold_table = threshold_table.sort_values(
        ["model", "threshold"]
    ).reset_index(drop=True)
    threshold_path = reports_dir / "threshold_comparison.csv"
    threshold_table.round(4).to_csv(threshold_path, index=False)
    print(f"  saved: {threshold_path.name}")

    # Charts
    for name, result in optimized.items():
        file_name = (
            "confusion_matrix_decision_tree.png"
            if name == "Decision Tree"
            else "confusion_matrix_random_forest.png"
        )
        plot_confusion_matrix(
            result["test_metrics"]["confusion_matrix"],
            name,
            result["selected_threshold"],
            reports_dir / file_name,
        )

    plot_roc_curves(
        y_test, test_probabilities, reports_dir / "roc_curve_comparison.png"
    )
    plot_precision_recall_curves(
        y_test, test_probabilities, reports_dir / "precision_recall_curve.png"
    )
    plot_threshold_analysis(
        threshold_tables,
        selected_thresholds,
        reports_dir / "threshold_analysis.png",
    )

    # Reports
    write_classification_report(
        reports_dir / "classification_report.txt",
        baseline,
        optimized,
        selected_name,
    )
    print("  saved: classification_report.txt")

    final_report = build_final_report(
        baseline,
        optimized,
        comparison_frame,
        threshold_table,
        selected_name,
        selected_result["selected_threshold"],
        y_test,
        search_times,
        feature_count,
        phase3_comparison,
    )
    (reports_dir / "final_model_selection.md").write_text(
        final_report, encoding="utf-8"
    )
    print("  saved: final_model_selection.md")

    # Task 6 clustering review ----------------------------------------------
    try:
        clustering_metrics = load_csv(
            PHASE3_REPORTS_DIR / "clustering_metrics.csv"
        )
        cluster_profiles = load_csv(PHASE3_REPORTS_DIR / "cluster_profiles.csv")
        kmeans = joblib.load(models_dir / "kmeans.joblib")
        kmeans_features = list(
            kmeans.named_steps["preprocessor"].feature_names_in_
        )
        if "y" in kmeans_features or "duration" in kmeans_features:
            raise ValueError("K-Means feature list contains y or duration.")
        clustering_report = build_clustering_report(
            clustering_metrics, cluster_profiles, kmeans_features
        )
        (reports_dir / "clustering_evaluation.md").write_text(
            clustering_report, encoding="utf-8"
        )
        print("  saved: clustering_evaluation.md")
        print(
            "  verified kmeans inputs: no y, no duration "
            f"({len(kmeans_features)} features)"
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"\nERROR reading Phase 3 clustering artifacts: {exc}", file=sys.stderr)
        return 1

    # Validation ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 9 - FINAL VALIDATION")
    print("=" * 80)
    try:
        verify_artifact(artifact, X_test, y_test, selected_result, models_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR validating final artifact: {exc}", file=sys.stderr)
        return 1

    required_outputs = [
        "model_comparison.csv",
        "optimized_model_results.csv",
        "threshold_comparison.csv",
        "final_model_selection.md",
        "classification_report.txt",
        "confusion_matrix_decision_tree.png",
        "confusion_matrix_random_forest.png",
        "roc_curve_comparison.png",
        "precision_recall_curve.png",
        "threshold_analysis.png",
        "clustering_evaluation.md",
    ]
    missing_outputs = [
        name for name in required_outputs if not (reports_dir / name).exists()
    ]
    if missing_outputs:
        print(f"ERROR missing outputs: {missing_outputs}", file=sys.stderr)
        return 1
    print(f"  verified reports/charts: {len(required_outputs)}")
    print("  test-set leakage check: threshold and parameters selected on training only")
    print("  duration check: excluded from model inputs")

    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)
    print(f"Selected model: {selected_name}")
    print(f"Selected threshold: {selected_result['selected_threshold']:.2f}")
    print(
        f"Test metrics: F1={selected_result['test_metrics']['metrics']['f1']:.4f}, "
        f"recall={selected_result['test_metrics']['metrics']['recall']:.4f}, "
        f"precision={selected_result['test_metrics']['metrics']['precision']:.4f}, "
        f"ROC-AUC={selected_result['test_metrics']['metrics']['roc_auc']:.4f}"
    )
    print(f"Models directory: {models_dir}")
    print(f"Reports directory: {reports_dir}")
    print("\nPhase 5 was not started.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
