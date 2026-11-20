#!/usr/bin/env python
"""
Phase 1 - Dataset Understanding and Exploratory Data Analysis
=============================================================

Bank Marketing Intelligence System (Information Systems Management project).

This script loads the UCI Bank Marketing dataset (bank-full.csv), validates it,
profiles data quality, produces descriptive statistics, analyses customer
segments, generates professional charts, and writes a reproducible Markdown
report. No machine-learning models are trained in this phase.

Usage
-----
    python src/phase1_exploration.py
    python src/phase1_exploration.py --data data/raw/bank-full.csv
    python src/phase1_exploration.py --output-dir reports/phase1 --dpi 200

Outputs (default: reports/phase1/)
---------------------------------
    dataset_summary.csv
    missing_values.csv
    unknown_values.csv
    categorical_summary.csv
    numerical_summary.csv
    target_distribution.csv
    subscription_rates_by_segment.csv   (supporting table)
    phase1_report.md
    01_... 11_... .png                  (11 charts)

The original raw CSV is never modified.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

# Use a non-interactive backend when running as a script. In Jupyter the
# notebook backend is left untouched so charts can render inline.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase1"

TARGET = "y"
UNKNOWN_TOKEN = "unknown"
EXPECTED_ROWS = 45_211

NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_COLUMNS = [
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

AGE_BINS = [17, 24, 34, 44, 54, 64, 120]
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]

# Target colours used consistently across charts.
TARGET_COLORS = {"no": "#4C72B0", "yes": "#DD8452"}
ACCENT = "#4C72B0"
BASELINE_COLOR = "#C44E52"

SEGMENT_SPEC = [
    ("job", "Occupation"),
    ("age_group", "Age group"),
    ("marital", "Marital status"),
    ("education", "Education"),
    ("poutcome", "Previous campaign outcome"),
    ("contact", "Contact method"),
    ("housing", "Housing loan"),
    ("loan", "Personal loan"),
    ("default", "Credit default"),
    ("month", "Contact month"),
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def pct(part: float, total: float) -> float:
    """Return part as a percentage of total, safely handling zero."""
    return 100.0 * part / total if total else 0.0


def overall_subscription_rate(df: pd.DataFrame) -> float:
    """Overall subscription rate in percent."""
    return pct(int((df[TARGET] == "yes").sum()), len(df))


def set_plot_style(dpi: int = 150) -> None:
    """Apply a clean, consistent Matplotlib/Seaborn theme."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="deep",
        rc={
            "figure.dpi": 110,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#888888",
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        },
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure to disk and release its memory."""
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved chart: {path.name}")


def md_table(frame: pd.DataFrame, float_fmt: str = "{:.2f}") -> str:
    """Render a small DataFrame as GitHub-flavoured Markdown (no extra deps)."""
    columns = [str(c) for c in frame.columns]
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


def top_and_bottom(table: pd.DataFrame, min_n: int = 200) -> tuple[pd.Series, pd.Series]:
    """Return the highest- and lowest-converting segment with sample >= min_n."""
    eligible = table[table["n"] >= min_n].sort_values(
        "subscription_rate_pct", ascending=False
    )
    return eligible.iloc[0], eligible.iloc[-1]


# ---------------------------------------------------------------------------
# Task A - Dataset loading and validation
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load bank-full.csv (semicolon separated) and validate its structure.

    Raises
    ------
    FileNotFoundError, ValueError
        With actionable messages for missing paths, parse errors, or schema
        mismatches.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Place bank-full.csv in data/raw/ or pass --data <path>."
        )
    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    try:
        # The UCI file uses ';' - never assume comma separation.
        df = pd.read_csv(path, sep=";")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Could not parse {path} with sep=';'. "
            f"Check delimiter and file integrity. Details: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Encoding error while reading {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read dataset {path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset contains no rows: {path}")

    # Normalise headers and string values without touching the source file.
    df.columns = [str(c).strip() for c in df.columns]

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_columns}\n"
            f"Columns found ({len(df.columns)}): {list(df.columns)}"
        )

    extra_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_columns:
        warnings.warn(
            f"Ignoring unexpected extra columns: {extra_columns}", stacklevel=2
        )

    df = df[EXPECTED_COLUMNS].copy()

    for column in CATEGORICAL_COLUMNS + [TARGET]:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    invalid_targets = sorted(set(df[TARGET].dropna().unique()) - {"yes", "no"})
    if invalid_targets:
        raise ValueError(
            f"Target column '{TARGET}' contains unexpected values: {invalid_targets}. "
            "Expected only 'yes' / 'no'."
        )

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' was expected to be numeric but has dtype "
                f"{df[column].dtype}. Check the delimiter or data integrity."
            )

    if len(df) != EXPECTED_ROWS:
        warnings.warn(
            f"Expected {EXPECTED_ROWS:,} records for bank-full.csv but found "
            f"{len(df):,}. Continuing with the loaded data.",
            stacklevel=2,
        )

    return df


def print_dataset_overview(df: pd.DataFrame, path: Path) -> None:
    """Task A console output: shape, head, tail, names and dtypes."""
    print("\n" + "=" * 78)
    print("TASK A - DATASET LOADING")
    print("=" * 78)
    print(f"File              : {path}")
    print(f"Shape (rows, cols): {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory usage      : {df.memory_usage(deep=True).sum() / 1024:,.1f} KB")

    print("\nFirst 5 records:")
    print(df.head(5).to_string())

    print("\nLast 5 records:")
    print(df.tail(5).to_string())

    print("\nColumns and data types:")
    dtype_table = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).to_numpy(),
            "non_null": df.notna().sum().to_numpy(),
            "unique": df.nunique(dropna=True).to_numpy(),
        }
    )
    print(dtype_table.to_string(index=False))


# ---------------------------------------------------------------------------
# Task B - Data quality analysis
# ---------------------------------------------------------------------------

def analyse_quality(df: pd.DataFrame) -> dict[str, object]:
    """Profile missing values, duplicates, 'unknown' categories and pdays=-1."""
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).to_numpy(),
            "non_null_count": df.notna().sum().to_numpy(),
            "missing_count": df.isna().sum().to_numpy(),
        }
    )
    missing["missing_pct"] = (
        missing["missing_count"] / len(df) * 100
    ).round(4)

    duplicate_rows = int(df.duplicated().sum())

    unknown_rows = []
    for column in CATEGORICAL_COLUMNS:
        unknown_count = int(df[column].eq(UNKNOWN_TOKEN).sum())
        unknown_rows.append(
            {
                "column": column,
                "unknown_count": unknown_count,
                "unknown_pct": round(pct(unknown_count, len(df)), 4),
                "unique_categories": int(df[column].nunique(dropna=True)),
                "has_unknown": unknown_count > 0,
            }
        )
    unknown = pd.DataFrame(unknown_rows).sort_values(
        "unknown_pct", ascending=False
    ).reset_index(drop=True)

    pdays_minus_one = int((df["pdays"] == -1).sum())
    previous_zero = int((df["previous"] == 0).sum())
    inconsistent = int(((df["pdays"] == -1) != (df["previous"] == 0)).sum())

    quality = {
        "missing": missing,
        "unknown": unknown,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": pct(duplicate_rows, len(df)),
        "total_missing_cells": int(df.isna().sum().sum()),
        "unknown_cells": int(
            sum(df[c].eq(UNKNOWN_TOKEN).sum() for c in CATEGORICAL_COLUMNS)
        ),
        "pdays_minus_one": pdays_minus_one,
        "pdays_minus_one_pct": pct(pdays_minus_one, len(df)),
        "previous_zero": previous_zero,
        "pdays_previous_inconsistent": inconsistent,
    }

    print("\n" + "=" * 78)
    print("TASK B - DATA QUALITY ANALYSIS")
    print("=" * 78)
    print(f"Missing cells (all columns): {quality['total_missing_cells']:,}")
    print(f"Duplicate rows             : {duplicate_rows:,} "
          f"({quality['duplicate_pct']:.2f}%)")
    print(f"'unknown' cells (categorical): {quality['unknown_cells']:,}")

    print("\n'unknown' values by column:")
    print(unknown.to_string(index=False))

    print(
        f"\npdays = -1 (never contacted before): {pdays_minus_one:,} "
        f"({quality['pdays_minus_one_pct']:.2f}%)"
    )
    print(f"previous = 0                     : {previous_zero:,}")
    print(f"pdays=-1 / previous=0 mismatches : {inconsistent:,}")
    print("\nNOTE: 'unknown' categories and pdays=-1 are retained in this phase.")
    print("They are treated as informative categories, not automatically deleted.")

    return quality


# ---------------------------------------------------------------------------
# Task C - Statistical analysis
# ---------------------------------------------------------------------------

def analyse_statistics(df: pd.DataFrame) -> dict[str, object]:
    """Descriptive statistics for numerical variables, categories and target."""
    numeric = df[NUMERIC_COLUMNS].describe().T
    numeric = numeric.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})
    numeric.index.name = "variable"
    numeric = numeric.reset_index()
    numeric["iqr"] = numeric["q75"] - numeric["q25"]
    numeric["skewness"] = df[NUMERIC_COLUMNS].skew().to_numpy()
    numeric["kurtosis"] = df[NUMERIC_COLUMNS].kurtosis().to_numpy()
    numeric["cv_pct"] = (
        numeric["std"] / numeric["mean"] * 100
    ).replace([np.inf, -np.inf], np.nan)
    numeric["negative_count"] = (df[NUMERIC_COLUMNS] < 0).sum().to_numpy()
    numeric["zero_count"] = (df[NUMERIC_COLUMNS] == 0).sum().to_numpy()
    numeric["count"] = numeric["count"].astype(int)
    numeric = numeric[
        [
            "variable",
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "iqr",
            "skewness",
            "kurtosis",
            "cv_pct",
            "negative_count",
            "zero_count",
        ]
    ]
    float_columns = [
        "mean", "std", "min", "q25", "median", "q75", "max",
        "iqr", "skewness", "kurtosis", "cv_pct",
    ]
    numeric[float_columns] = numeric[float_columns].round(4)

    categorical_rows = []
    for column in CATEGORICAL_COLUMNS:
        counts = df[column].value_counts(dropna=False)
        for value, count in counts.items():
            categorical_rows.append(
                {
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "pct_of_column": round(pct(count, len(df)), 4),
                    "is_unknown": str(value).lower() == UNKNOWN_TOKEN,
                }
            )
    categorical = pd.DataFrame(categorical_rows)

    target_counts = df[TARGET].value_counts().reindex(["no", "yes"])
    target = pd.DataFrame(
        {
            "y": target_counts.index,
            "count": target_counts.to_numpy(),
            "percentage": [
                round(pct(value, len(df)), 4) for value in target_counts.to_numpy()
            ],
        }
    )

    dtypes = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).to_numpy(),
            "role": [
                "target"
                if c == TARGET
                else ("numeric" if c in NUMERIC_COLUMNS else "categorical")
                for c in df.columns
            ],
            "unique_values": df.nunique(dropna=True).to_numpy(),
        }
    )

    statistics = {
        "numeric": numeric,
        "categorical": categorical,
        "target": target,
        "dtypes": dtypes,
    }

    print("\n" + "=" * 78)
    print("TASK C - STATISTICAL ANALYSIS")
    print("=" * 78)
    print("\nNumerical variables (rounded):")
    print(numeric.round(2).to_string(index=False))

    print("\nTarget distribution:")
    print(target.to_string(index=False))

    yes_pct = float(target.loc[target["y"] == "yes", "percentage"].iloc[0])
    no_pct = float(target.loc[target["y"] == "no", "percentage"].iloc[0])
    print(f"\nNon-subscribers: {no_pct:.2f}% | Subscribers: {yes_pct:.2f}%")
    print(
        f"Class imbalance: {no_pct / yes_pct:.2f} non-subscribers per subscriber. "
        "Accuracy alone is therefore misleading."
    )

    return statistics


# ---------------------------------------------------------------------------
# Task D - Customer analysis
# ---------------------------------------------------------------------------

def add_customer_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a readable age_group column added."""
    enriched = df.copy()
    enriched["age_group"] = pd.cut(
        enriched["age"],
        bins=AGE_BINS,
        labels=AGE_ORDER,
        right=True,
        include_lowest=True,
    )
    return enriched


def subscription_rate_table(
    df: pd.DataFrame,
    column: str,
    *,
    order: list[str] | None = None,
    sort_by_rate: bool = True,
) -> pd.DataFrame:
    """Counts and subscription rate (%) for each category of one column."""
    grouped = (
        df[[column, TARGET]]
        .dropna()
        .groupby(column, observed=True)[TARGET]
        .agg(
            n="size",
            subscriptions=lambda series: int((series == "yes").sum()),
        )
        .reset_index()
        .rename(columns={column: "segment"})
    )
    grouped["non_subscriptions"] = grouped["n"] - grouped["subscriptions"]
    grouped["subscription_rate_pct"] = (
        grouped["subscriptions"] / grouped["n"] * 100
    )
    grouped = grouped[
        [
            "segment",
            "n",
            "subscriptions",
            "non_subscriptions",
            "subscription_rate_pct",
        ]
    ]

    if order is not None:
        rank = {value: index for index, value in enumerate(order)}
        grouped = (
            grouped.assign(_rank=grouped["segment"].astype(str).map(rank))
            .sort_values("_rank")
            .drop(columns="_rank")
        )
    elif sort_by_rate:
        grouped = grouped.sort_values("subscription_rate_pct", ascending=False)

    return grouped.reset_index(drop=True)


def build_rate_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build subscription-rate tables for every customer segment."""
    tables: dict[str, pd.DataFrame] = {}
    for column, _ in SEGMENT_SPEC:
        if column == "age_group":
            tables[column] = subscription_rate_table(df, column, order=AGE_ORDER)
        elif column == "month":
            tables[column] = subscription_rate_table(df, column, order=MONTH_ORDER)
        else:
            tables[column] = subscription_rate_table(df, column)
    return tables


def print_customer_analysis(
    df: pd.DataFrame, tables: dict[str, pd.DataFrame]
) -> None:
    """Task D console output."""
    print("\n" + "=" * 78)
    print("TASK D - CUSTOMER ANALYSIS")
    print("=" * 78)

    print("\nAge distribution (numerical):")
    age_stats = df["age"].describe()
    print(
        f"  min={age_stats['min']:.0f}, q25={age_stats['25%']:.0f}, "
        f"median={age_stats['50%']:.0f}, mean={age_stats['mean']:.1f}, "
        f"q75={age_stats['75%']:.0f}, max={age_stats['max']:.0f}"
    )

    for column, label in SEGMENT_SPEC:
        print(f"\nSubscription rate by {label.lower()}:")
        table = tables[column].copy()
        table["subscription_rate_pct"] = table["subscription_rate_pct"].round(2)
        print(table.to_string(index=False))

    print("\nAccount balance:")
    balance = df["balance"]
    print(
        f"  mean={balance.mean():,.1f} EUR, median={balance.median():,.1f} EUR, "
        f"min={balance.min():,} EUR, max={balance.max():,} EUR"
    )
    print(
        f"  negative balances: {(balance < 0).sum():,} "
        f"({pct((balance < 0).sum(), len(df)):.2f}%)"
    )


# ---------------------------------------------------------------------------
# Task E - Visualisations
# ---------------------------------------------------------------------------

def _rate_colors(rates: pd.Series) -> list[tuple[float, float, float, float]]:
    """Map higher subscription rates to darker blue shades."""
    maximum = max(float(rates.max()), 1e-9)
    colormap = plt.get_cmap("Blues")
    return [colormap(0.35 + 0.50 * float(rate) / maximum) for rate in rates]


def _rate_bar_labels(
    ax: plt.Axes,
    rates: pd.Series,
    counts: pd.Series,
    *,
    horizontal: bool,
) -> None:
    """Add 'rate % (n=...)' labels to a subscription-rate bar chart."""
    maximum = max(float(rates.max()), 1e-9)
    for index, (rate, count) in enumerate(zip(rates, counts)):
        label = f"{rate:.1f}%  (n={int(count):,})"
        if horizontal:
            ax.text(
                rate + maximum * 0.02,
                index,
                label,
                va="center",
                ha="left",
                fontsize=9,
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 1.0, "pad": 0.8},
                zorder=5,
            )
        else:
            ax.text(
                index,
                rate + maximum * 0.02,
                f"{rate:.1f}%\nn={int(count):,}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 1.0, "pad": 0.8},
                zorder=5,
            )


def plot_target_distribution(
    df: pd.DataFrame, target: pd.DataFrame, path: Path
) -> None:
    """Chart 1 - target class distribution."""
    counts = target.set_index("y").reindex(["no", "yes"])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(
        ["No", "Yes"],
        counts["count"],
        color=[TARGET_COLORS["no"], TARGET_COLORS["yes"]],
        width=0.58,
    )
    for bar, (_, row) in zip(bars, counts.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts["count"].max() * 0.015,
            f"{int(row['count']):,}\n({row['percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )
    ax.set_title("Term-Deposit Subscription Distribution (Target Variable)")
    ax.set_xlabel("Subscribed to a term deposit (y)")
    ax.set_ylabel("Number of customers")
    ax.set_ylim(0, counts["count"].max() * 1.18)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ratio = counts.loc["no", "count"] / counts.loc["yes", "count"]
    ax.text(
        0.98,
        0.55,
        f"Class imbalance: {ratio:.2f} non-subscribers per subscriber",
        transform=ax.transAxes,
        fontsize=9,
        color="#666666",
        va="center",
        ha="right",
    )
    save_figure(fig, path)


def plot_age_distribution(df: pd.DataFrame, path: Path) -> None:
    """Chart 2 - customer age distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        df["age"],
        bins=np.arange(17, 101, 2),
        kde=True,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    mean_age = df["age"].mean()
    median_age = df["age"].median()
    ax.axvline(
        mean_age,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.4,
        label=f"Mean age: {mean_age:.1f}",
    )
    ax.axvline(
        median_age,
        color="#55A868",
        linestyle=":",
        linewidth=1.6,
        label=f"Median age: {median_age:.1f}",
    )
    ax.set_title("Customer Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of customers")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.55,
        f"n = {len(df):,} | range {df['age'].min():.0f}-{df['age'].max():.0f} years",
        transform=ax.transAxes,
        fontsize=9,
        color="#666666",
        va="center",
        ha="right",
    )
    save_figure(fig, path)


def plot_job_distribution(df: pd.DataFrame, path: Path) -> None:
    """Chart 3 - customer occupation distribution."""
    counts = df["job"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    colors = sns.color_palette("crest", n_colors=len(counts))
    bars = ax.barh(counts.index, counts.to_numpy(), color=colors)
    for bar, value in zip(bars, counts.to_numpy()):
        ax.text(
            value + counts.max() * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,} ({pct(value, len(df)):.1f}%)",
            va="center",
            fontsize=9,
            color="#333333",
        )
    ax.set_title("Customer Occupation Distribution")
    ax.set_xlabel("Number of customers")
    ax.set_ylabel("Occupation (job)")
    ax.set_xlim(0, counts.max() * 1.22)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    save_figure(fig, path)


def plot_balance_distribution(df: pd.DataFrame, path: Path) -> None:
    """Chart 4 - account balance distribution (clipped for readability)."""
    balance = df["balance"]
    lower, upper = balance.quantile([0.01, 0.99])
    displayed = balance[(balance >= lower) & (balance <= upper)]
    outliers = int(len(balance) - len(displayed))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        displayed,
        bins=60,
        color="#55A868",
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    mean_balance = balance.mean()
    median_balance = balance.median()
    ax.axvline(
        mean_balance,
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=1.4,
        label=f"Mean: {mean_balance:,.0f} EUR",
    )
    ax.axvline(
        median_balance,
        color="#4C72B0",
        linestyle=":",
        linewidth=1.6,
        label=f"Median: {median_balance:,.0f} EUR",
    )
    ax.set_title("Customer Account Balance Distribution")
    ax.set_xlabel("Average yearly balance (EUR)")
    ax.set_ylabel("Number of customers")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    ax.legend(loc="upper right")
    fig.text(
        0.01,
        -0.02,
        f"n = {len(df):,} | displayed range: 1st-99th percentile "
        f"({lower:,.0f} to {upper:,.0f} EUR); {outliers:,} tail observations are "
        "excluded from the chart and summarised in numerical_summary.csv.",
        fontsize=8,
        color="#666666",
    )
    save_figure(fig, path)


def plot_rate_bars(
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    path: Path,
    *,
    baseline: float,
    horizontal: bool = False,
) -> None:
    """Generic subscription-rate bar chart with labels and baseline."""
    labels = table["segment"].astype(str)
    rates = table["subscription_rate_pct"].astype(float)
    counts = table["n"].astype(int)
    colors = _rate_colors(rates)
    maximum = max(float(rates.max()), 1e-9)

    if horizontal:
        fig, ax = plt.subplots(figsize=(9.2, max(4.6, 0.46 * len(table) + 2.0)))
        bars = ax.barh(labels, rates, color=colors, edgecolor="white")
        ax.set_xlim(0, maximum * 1.38)
        _rate_bar_labels(ax, rates, counts, horizontal=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title.replace("Subscription Rate by ", ""))
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0f}%")
        )
        ax.axvline(
            baseline,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"Overall rate: {baseline:.1f}%",
        )
        ax.legend(loc="lower right")
        for bar in bars:
            bar.set_height(0.72)
    else:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        positions = np.arange(len(table))
        ax.bar(
            positions,
            rates,
            color=colors,
            edgecolor="white",
            width=0.68,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0 if len(table) <= 6 else 30,
                           ha="right" if len(table) > 6 else "center")
        ax.set_xlim(-0.6, len(table) - 0.0)
        ax.set_ylim(0, maximum * 1.30)
        _rate_bar_labels(ax, rates, counts, horizontal=False)
        ax.set_xlabel(title.replace("Subscription Rate by ", ""))
        ax.set_ylabel(xlabel)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0f}%")
        )
        ax.axhline(
            baseline,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"Overall rate: {baseline:.1f}%",
        )
        ax.legend(loc="upper right")

    ax.set_title(title)
    save_figure(fig, path)


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Chart 11 - correlation heatmap for numerical variables."""
    numeric_frame = df[NUMERIC_COLUMNS].copy()
    numeric_frame["y (encoded)"] = (df[TARGET] == "yes").astype(int)
    correlation = numeric_frame.corr()

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    sns.heatmap(
        correlation,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of Numerical Variables (with Encoded Target)")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
    ax.grid(False)
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    save_figure(fig, path)


def create_visualizations(
    df: pd.DataFrame,
    statistics: dict[str, object],
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> list[Path]:
    """Generate all 11 required charts and return their paths."""
    print("\n" + "=" * 78)
    print("TASK E - EXPLORATORY VISUALIZATIONS")
    print("=" * 78)

    baseline = overall_subscription_rate(df)
    charts = [
        (
            "01_target_distribution.png",
            lambda path: plot_target_distribution(df, statistics["target"], path),
        ),
        (
            "02_age_distribution.png",
            lambda path: plot_age_distribution(df, path),
        ),
        (
            "03_job_distribution.png",
            lambda path: plot_job_distribution(df, path),
        ),
        (
            "04_balance_distribution.png",
            lambda path: plot_balance_distribution(df, path),
        ),
        (
            "05_subscription_rate_by_job.png",
            lambda path: plot_rate_bars(
                tables["job"],
                "Subscription Rate by Occupation",
                "Subscription rate (%)",
                path,
                baseline=baseline,
                horizontal=True,
            ),
        ),
        (
            "06_subscription_rate_by_age_group.png",
            lambda path: plot_rate_bars(
                tables["age_group"],
                "Subscription Rate by Age Group",
                "Subscription rate (%)",
                path,
                baseline=baseline,
            ),
        ),
        (
            "07_subscription_rate_by_marital_status.png",
            lambda path: plot_rate_bars(
                tables["marital"],
                "Subscription Rate by Marital Status",
                "Subscription rate (%)",
                path,
                baseline=baseline,
            ),
        ),
        (
            "08_subscription_rate_by_education.png",
            lambda path: plot_rate_bars(
                tables["education"],
                "Subscription Rate by Education Level",
                "Subscription rate (%)",
                path,
                baseline=baseline,
            ),
        ),
        (
            "09_subscription_rate_by_poutcome.png",
            lambda path: plot_rate_bars(
                tables["poutcome"],
                "Subscription Rate by Previous Campaign Outcome",
                "Subscription rate (%)",
                path,
                baseline=baseline,
            ),
        ),
        (
            "10_subscription_rate_by_contact.png",
            lambda path: plot_rate_bars(
                tables["contact"],
                "Subscription Rate by Contact Method",
                "Subscription rate (%)",
                path,
                baseline=baseline,
            ),
        ),
        (
            "11_correlation_heatmap.png",
            lambda path: plot_correlation_heatmap(df, path),
        ),
    ]

    written: list[Path] = []
    for filename, plotter in charts:
        path = output_dir / filename
        plotter(path)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(
    df: pd.DataFrame,
    path: Path,
    quality: dict[str, object],
    statistics: dict[str, object],
    tables: dict[str, pd.DataFrame],
    chart_paths: list[Path],
) -> str:
    """Compose the Phase 1 Markdown report from computed results."""
    n_rows = len(df)
    target = statistics["target"].set_index("y")
    yes_count = int(target.loc["yes", "count"])
    no_count = int(target.loc["no", "count"])
    yes_pct = float(target.loc["yes", "percentage"])
    no_pct = float(target.loc["no", "percentage"])
    imbalance_ratio = no_count / yes_count if yes_count else float("nan")
    baseline = yes_pct

    job_top, job_bottom = top_and_bottom(tables["job"])
    age_top, age_bottom = top_and_bottom(tables["age_group"])
    education_top, education_bottom = top_and_bottom(tables["education"])
    marital_top, marital_bottom = top_and_bottom(tables["marital"])
    poutcome_top, poutcome_bottom = top_and_bottom(tables["poutcome"])
    contact_top, contact_bottom = top_and_bottom(tables["contact"])

    housing_no = tables["housing"].loc[tables["housing"]["segment"] == "no"].iloc[0]
    housing_yes = tables["housing"].loc[tables["housing"]["segment"] == "yes"].iloc[0]
    loan_no = tables["loan"].loc[tables["loan"]["segment"] == "no"].iloc[0]
    loan_yes = tables["loan"].loc[tables["loan"]["segment"] == "yes"].iloc[0]

    duration_corr = float(
        np.corrcoef(df["duration"], (df[TARGET] == "yes").astype(int))[0, 1]
    )
    pdays_corr = float(
        np.corrcoef(df["pdays"], (df[TARGET] == "yes").astype(int))[0, 1]
    )

    balance = df["balance"]
    negative_balance = int((balance < 0).sum())
    pdays_pct = float(quality["pdays_minus_one_pct"])
    unknown = quality["unknown"]

    try:
        display_path = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(path)

    poutcome_unknown = int(
        unknown.loc[unknown["column"] == "poutcome", "unknown_count"].iloc[0]
    )
    poutcome_gap = poutcome_unknown - int(quality["pdays_minus_one"])

    def job_rate(segment: str) -> tuple[float, int]:
        row = tables["job"].loc[tables["job"]["segment"] == segment].iloc[0]
        return float(row["subscription_rate_pct"]), int(row["n"])

    student_rate, student_n = job_rate("student")
    retired_rate, retired_n = job_rate("retired")
    blue_rate, blue_n = job_rate("blue-collar")

    best_month = tables["month"].sort_values(
        "subscription_rate_pct", ascending=False
    ).iloc[0]
    busiest_month = tables["month"].sort_values("n", ascending=False).iloc[0]

    lines: list[str] = []
    add = lines.append

    add("# Phase 1 - Dataset Understanding & Exploratory Data Analysis")
    add("")
    add("**Project:** Bank Marketing Intelligence System  ")
    add("**Course context:** Information Systems Management  ")
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    add(f"**Dataset:** `{display_path}`  ")
    add(f"**Records / columns:** {n_rows:,} / {df.shape[1]}  ")
    add("**Tools:** Python 3, Pandas, NumPy, Matplotlib, Seaborn  ")
    add("**Scope:** Phase 1 only - no machine-learning models were trained.")
    add("")
    add("---")
    add("")
    add("## 1. Executive summary")
    add("")
    add(f"- The dataset contains **{n_rows:,} customer contacts** and **{df.shape[1]} attributes** "
        "describing client profile, current-campaign contact details, and previous-campaign history.")
    add(f"- Only **{yes_count:,} customers ({yes_pct:.2f}%) subscribed** to a term deposit; "
        f"**{no_count:,} ({no_pct:.2f}%) did not**. The majority-to-minority ratio is "
        f"**{imbalance_ratio:.2f}:1**, a clear class-imbalance problem.")
    add(f"- A trivial 'always no' classifier would already score **{no_pct:.2f}% accuracy**, "
        "which shows why accuracy alone cannot evaluate marketing models.")
    add(f"- There are **no NULL cells** and **{quality['duplicate_rows']:,} duplicate rows**. "
        f"Data completeness is high, but **{quality['unknown_cells']:,} 'unknown' category "
        "values** require interpretation rather than deletion.")
    add(f"- **{quality['pdays_minus_one']:,} records ({pdays_pct:.2f}%) have `pdays = -1`**, "
        "meaning the customer had never been contacted before. This is a sentinel value, "
        "not a real elapsed-day count.")
    add(f"- Previous-campaign success is the strongest behavioural signal: "
        f"**{float(poutcome_top['subscription_rate_pct']):.2f}%** of customers with "
        f"`poutcome = {poutcome_top['segment']}` subscribed, versus "
        f"**{float(poutcome_bottom['subscription_rate_pct']):.2f}%** for "
        f"`{poutcome_bottom['segment']}`.")
    add(f"- **{blue_n:,} blue-collar customers** form the largest occupational group, but "
        f"convert at only **{blue_rate:.2f}%** - well below the {baseline:.2f}% baseline. "
        f"Smaller groups such as students ({student_rate:.2f}%, n={student_n:,}) and retirees "
        f"({retired_rate:.2f}%, n={retired_n:,}) convert far better.")
    add("")
    add("---")
    add("")
    add("## 2. Dataset overview (Task A)")
    add("")
    add(f"The file was loaded with `pd.read_csv(path, sep=';')`, validated against the "
        f"documented 17-column schema, and inspected without modification. The raw CSV is "
        f"**{path.stat().st_size / 1024:,.1f} KB** and uses the UCI Bank Marketing schema "
        "(17 attributes, target `y`).")
    add("")
    add(md_table(statistics["dtypes"]))
    add("")
    add("### Dataset-level summary")
    add("")
    add(md_table(pd.DataFrame({
        "metric": [
            "rows",
            "columns",
            "numeric variables",
            "categorical variables",
            "missing cells",
            "duplicate rows",
            "unknown categorical cells",
            "subscribers (yes)",
            "non-subscribers (no)",
            "imbalance ratio (no:yes)",
            "majority-class baseline accuracy (%)",
        ],
        "value": [
            f"{n_rows:,}",
            f"{df.shape[1]}",
            f"{len(NUMERIC_COLUMNS)}",
            f"{len(CATEGORICAL_COLUMNS)}",
            f"{quality['total_missing_cells']:,}",
            f"{quality['duplicate_rows']:,}",
            f"{quality['unknown_cells']:,}",
            f"{yes_count:,}",
            f"{no_count:,}",
            f"{imbalance_ratio:.2f}",
            f"{no_pct:.2f}",
        ],
    })))
    add("")
    add("---")
    add("")
    add("## 3. Data quality (Task B)")
    add("")
    add("### 3.1 Missing values")
    add("")
    add(f"Across all {n_rows:,} rows and {df.shape[1]} columns there are "
        f"**{quality['total_missing_cells']:,} NULL/NaN cells**. No imputation is required "
        "from a technical standpoint; the more important issue is the explicit "
        "'unknown' category described below.")
    add("")
    add("### 3.2 Duplicate records")
    add("")
    add(f"Exact duplicate rows: **{quality['duplicate_rows']:,}** "
        f"({quality['duplicate_pct']:.2f}% of the dataset). No duplicates were removed.")
    add("")
    add("### 3.3 'unknown' categorical values")
    add("")
    add("'unknown' is a *category*, not a standard missing value. It was counted per column "
        "and kept intact. Counts below are of all records.")
    add("")
    add(md_table(unknown[["column", "unknown_count", "unknown_pct", "unique_categories"]]))
    add("")
    add(f"The highest unknown rate is **`contact` ({float(unknown.loc[unknown['column'] == 'contact', 'unknown_pct'].iloc[0]):.2f}%)** "
        f"and **`poutcome` ({float(unknown.loc[unknown['column'] == 'poutcome', 'unknown_pct'].iloc[0]):.2f}%)**. "
        "`poutcome = unknown` largely overlaps with never having been contacted before, "
        "so it carries information rather than being random missingness. "
        f"There are {poutcome_unknown:,} `poutcome = unknown` records but only "
        f"{int(quality['pdays_minus_one']):,} never-contacted records, so {poutcome_gap} "
        "customers have previous contact history without a recorded outcome. "
        "`job` and `education` have small unknown shares.")
    add("")
    add("### 3.4 Special value pdays = -1")
    add("")
    add(f"- `pdays = -1`: **{quality['pdays_minus_one']:,}** records "
        f"(**{pdays_pct:.2f}%**).")
    add(f"- `previous = 0`: **{quality['previous_zero']:,}** records.")
    add(f"- Records where `pdays` / `previous` disagree: "
        f"**{quality['pdays_previous_inconsistent']:,}**.")
    add("")
    add("`-1` is a sentinel meaning **'client not previously contacted'**, not one day before "
        "the campaign. Numeric summaries and correlations that include `-1` describe "
        "contact history, not elapsed time.")
    add("")
    add("---")
    add("")
    add("## 4. Statistical analysis (Task C)")
    add("")
    add("### 4.1 Numerical variables")
    add("")
    numeric_display = statistics["numeric"].copy()
    numeric_display = numeric_display[
        ["variable", "count", "mean", "std", "min", "q25", "median",
         "q75", "max", "skewness"]
    ].round(2)
    add(md_table(numeric_display))
    add("")
    add(f"Account balance is highly right-skewed (skewness "
        f"{float(statistics['numeric'].loc[statistics['numeric']['variable'] == 'balance', 'skewness'].iloc[0]):.2f}), "
        f"ranges from {int(balance.min()):,} to {int(balance.max()):,} EUR, and includes "
        f"**{negative_balance:,} negative balances** "
        f"({pct(negative_balance, n_rows):.2f}%). Duration and campaign counts are likewise "
        "concentrated at low values with long tails.")
    add("")
    add("### 4.2 Categorical variables")
    add("")
    add("`categorical_summary.csv` contains every category with counts and percentages. "
        "The largest groups are married customers, secondary education, and blue-collar "
        "occupations; the target is strongly dominated by 'no'.")
    add("")
    add("Top categories per variable:")
    add("")
    top_categories = (
        statistics["categorical"]
        .sort_values(["column", "count"], ascending=[True, False])
        .groupby("column", sort=False)
        .head(3)
        .loc[:, ["column", "value", "count", "pct_of_column"]]
        .rename(columns={"pct_of_column": "pct_of_column (%)"})
        .round(2)
    )
    add(md_table(top_categories))
    add("")
    add("### 4.3 Target distribution and class imbalance")
    add("")
    add(md_table(target.reset_index()))
    add("")
    add(f"- Non-subscription: **{no_pct:.2f}%** | Subscription: **{yes_pct:.2f}%**.")
    add(f"- Ratio: **{imbalance_ratio:.2f} non-subscribers for every subscriber**.")
    add(f"- A majority-class-only baseline reaches **{no_pct:.2f}% accuracy** while "
        "identifying zero potential customers.")
    add("")
    add("This is a **class-imbalance problem**. Model selection in later phases must use "
        "precision, recall, F1, ROC-AUC / PR-AUC, lift, and campaign-cost-based evaluation - "
        "not accuracy alone. Any resampling or class weighting must be fitted inside "
        "training folds only to avoid validation leakage.")
    add("")
    add("---")
    add("")
    add("## 5. Customer analysis (Task D)")
    add("")
    add(f"The overall subscription rate is **{baseline:.2f}%**. Every segment table below "
        "reports counts and rates; small segments should be interpreted with caution.")
    add("")

    segment_sections = [
        ("job", "Occupation"),
        ("age_group", "Age group"),
        ("marital", "Marital status"),
        ("education", "Education"),
        ("poutcome", "Previous campaign outcome"),
        ("contact", "Contact method"),
        ("housing", "Housing loan"),
        ("loan", "Personal loan"),
        ("default", "Credit default"),
        ("month", "Contact month"),
    ]
    for column, label in segment_sections:
        add(f"### 5.{segment_sections.index((column, label)) + 1} {label}")
        add("")
        table = tables[column].copy()
        table["subscription_rate_pct"] = table["subscription_rate_pct"].round(2)
        add(md_table(table))
        add("")

    add("### 5.11 Account balance")
    add("")
    add(f"- Mean balance: **{balance.mean():,.2f} EUR**; median: "
        f"**{balance.median():,.2f} EUR**; standard deviation: **{balance.std():,.2f} EUR**.")
    add(f"- Range: **{int(balance.min()):,}** to **{int(balance.max()):,} EUR**.")
    add(f"- **{negative_balance:,} customers ({pct(negative_balance, n_rows):.2f}%)** "
        "have a negative average yearly balance.")
    add("- Because the distribution is heavy-tailed, the chart shows the "
        "1st-99th percentile and excludes the remaining tail; full statistics are in "
        "`numerical_summary.csv`.")
    add("")
    add("---")
    add("")
    add("## 6. Visualisations (Task E)")
    add("")
    add("All charts are saved in this folder with titles, axis labels, legends where "
        "relevant, value annotations, and a common visual style.")
    add("")
    chart_titles = {
        "01_target_distribution.png": "Target variable distribution",
        "02_age_distribution.png": "Customer age distribution",
        "03_job_distribution.png": "Customer occupation distribution",
        "04_balance_distribution.png": "Account balance distribution",
        "05_subscription_rate_by_job.png": "Subscription rate by occupation",
        "06_subscription_rate_by_age_group.png": "Subscription rate by age group",
        "07_subscription_rate_by_marital_status.png": "Subscription rate by marital status",
        "08_subscription_rate_by_education.png": "Subscription rate by education",
        "09_subscription_rate_by_poutcome.png": "Subscription rate by previous campaign outcome",
        "10_subscription_rate_by_contact.png": "Subscription rate by contact method",
        "11_correlation_heatmap.png": "Correlation heatmap for numerical variables",
    }
    for chart_path in chart_paths:
        title = chart_titles.get(chart_path.name, chart_path.stem)
        add(f"### {title}")
        add("")
        add(f"![{title}]({chart_path.name})")
        add("")

    add("---")
    add("")
    add("## 7. Important data-mining considerations (Task F)")
    add("")
    add("### 7.1 Why the target variable is imbalanced")
    add("")
    add(f"Term deposits are a low-frequency response product: only **{yes_pct:.2f}%** of "
        "contacts converted. Direct-marketing campaigns deliberately contact many customers "
        "to find relatively few interested ones, so imbalance is inherent to the business "
        "process rather than a data-collection error. With a {:.2f}:1 ratio, a model can "
        "achieve high apparent accuracy by always predicting the majority class."
        .format(imbalance_ratio))
    add("")
    add("### 7.2 Why accuracy alone may be misleading")
    add("")
    add(f"A constant 'no' model scores **{no_pct:.2f}% accuracy** but generates no leads and "
        "no revenue. Conversely, a model that misses a likely subscriber loses a sale, while "
        "a false positive costs call-centre time. Accuracy gives both errors the wrong "
        "weight. Later phases should report **recall/sensitivity, precision, F1, ROC-AUC, "
        "PR-AUC, lift at a fixed contact budget, and calibration**, ideally converted into "
        "campaign profit or cost per acquired deposit.")
    add("")
    add("### 7.3 Why 'unknown' values need investigation")
    add("")
    add("'unknown' can mean different things: information not collected, customer refused to "
        "answer, a legacy-system field, or - for `poutcome` - the customer had no previous "
        "campaign. These mechanisms are not missing-at-random. Deleting such rows would remove "
        "large, non-representative groups (contact unknown alone is "
        f"**{float(unknown.loc[unknown['column'] == 'contact', 'unknown_pct'].iloc[0]):.2f}%** "
        "of records) and bias every later model. The recommended treatment is to keep "
        "'unknown' as an explicit category, investigate its relationship with the target, and "
        "decide per field (for example, encode `poutcome = unknown` together with "
        "`previous = 0`). Only well-justified domain rules should be applied, never automatic "
        "row deletion.")
    add("")
    add("### 7.4 Why pdays = -1 has a special meaning")
    add("")
    add(f"`pdays` is the number of days since the previous campaign contact. The value **-1** "
        f"is a flag for **'never previously contacted'** and applies to **{pdays_pct:.2f}%** "
        "of records. Averaging or correlating `pdays` as a numeric duration mixes two "
        "populations: customers with no history and customers contacted 1-871 days ago. "
        "This creates an artificial relationship (the raw correlation with the encoded "
        f"target is {pdays_corr:+.3f}). Better encodings are a binary "
        "`was_previously_contacted` flag plus a separate numeric `days_since_previous` "
        "defined only for contacted customers.")
    add("")
    add("### 7.5 Why duration may cause data leakage")
    add("")
    add("`duration` is the length of the *last contact* and is recorded only after the call "
        "has happened. It has the strongest raw numerical correlation with the target "
        f"(**{duration_corr:+.3f}**), but at the moment a customer is selected for calling, "
        "the duration is unknown. Using it to predict pre-call subscription would leak "
        "post-outcome information and produce optimistic results that cannot be reproduced "
        "in a real campaign. For a pre-call targeting model, `duration` should be excluded "
        "from the feature set. It is legitimate only for a different use case such as "
        "scoring during or after a call.")
    add("")
    add("---")
    add("")
    add("## 8. Key business insights")
    add("")
    add(f"1. **Previous success is the strongest signal.** Customers with "
        f"`poutcome = success` convert at **{float(poutcome_top['subscription_rate_pct']):.2f}%** "
        f"(n={int(poutcome_top['n']):,}), roughly "
        f"**{float(poutcome_top['subscription_rate_pct']) / baseline:.1f}x** the overall rate. "
        "A warm re-contact list should be a campaign priority.")
    add(f"2. **Occupation matters.** Students (**{student_rate:.2f}%**, n={student_n:,}) and "
        f"retirees (**{retired_rate:.2f}%**, n={retired_n:,}) convert well; blue-collar "
        f"workers (**{blue_rate:.2f}%**, n={blue_n:,}) form the largest group but convert "
        "least. High-volume segments are not automatically high-value segments.")
    add(f"3. **Age is non-linear.** Customers aged 65+ convert at "
        f"**{float(age_top['subscription_rate_pct']):.2f}%** and 18-24-year-olds at a "
        f"comparatively high rate, while the large 35-54 core converts around "
        f"**{float(age_bottom['subscription_rate_pct']):.2f}%**. Age should be modelled as "
        "non-linear or binned, not as a simple straight-line effect.")
    add(f"4. **Channel quality differs.** Cellular contact converts at "
        f"**{float(contact_top['subscription_rate_pct']):.2f}%** versus "
        f"**{float(contact_bottom['subscription_rate_pct']):.2f}%** for unknown contact "
        "records. The unknown channel group is also large, so channel data quality and "
        "contactability deserve attention.")
    add(f"5. **Existing debt reduces response.** Customers without a housing loan convert at "
        f"**{float(housing_no['subscription_rate_pct']):.2f}%** versus "
        f"**{float(housing_yes['subscription_rate_pct']):.2f}%** with one; the pattern is "
        f"similar for personal loans (**{float(loan_no['subscription_rate_pct']):.2f}%** vs "
        f"**{float(loan_yes['subscription_rate_pct']):.2f}%**). Affordability should inform "
        "targeting and product framing.")
    add(f"6. **Education and marital status add context.** Tertiary-educated customers convert "
        f"at **{float(education_top['subscription_rate_pct']):.2f}%** and single customers at "
        f"**{float(marital_top['subscription_rate_pct']):.2f}%**, both above the baseline; "
        "these variables are correlated with age and occupation, so their independent effect "
        "must be tested in a multivariate model.")
    add(f"7. **Timing is uneven but confounded.** The highest observed monthly rate is "
        f"**{float(best_month['subscription_rate_pct']):.2f}% in {best_month['segment']}** "
        f"(n={int(best_month['n']):,}), while the busiest month, "
        f"**{busiest_month['segment']}**, has {int(busiest_month['n']):,} contacts but a "
        f"rate of {float(busiest_month['subscription_rate_pct']):.2f}%. Month is entangled "
        "with campaign design, seasonality, and sample size; it is a candidate feature, not a "
        "standalone recommendation.")
    add("")
    add("---")
    add("")
    add("## 9. Limitations and concerns")
    add("")
    add("- **No ground truth for 'unknown'.** 'unknown' categories may hide different real "
        "states; treating them uniformly could distort segment interpretation.")
    add("- **Class imbalance.** With only 11.70% positives, accuracy is misleading and "
        "minority-class performance must be evaluated explicitly.")
    add("- **Potential leakage.** `duration` is post-contact information and must be excluded "
        "from pre-call models; `pdays = -1` is a sentinel, not a number.")
    add("- **Outliers and skew.** Balance and duration are heavily skewed; balance also "
        "contains negative values. Any scaling or capping must be fitted on training data only.")
    add("- **No year field.** The data span May 2008 - November 2010 but only day and month "
        "are recorded, so long-term trends and macroeconomic effects cannot be modelled "
        "directly.")
    add("- **Historical and local context.** This is one Portuguese bank's 2008-2010 campaign "
        "data; customer behaviour, regulation, and channel mix may differ today.")
    add("- **No cost/revenue variables.** The analysis cannot quantify the profit of a "
        "contact or the cost of a false positive; later phases need campaign economics.")
    add("- **Small segments.** Some rates (for example months with fewer than 1,000 contacts) "
        "are based on small samples and should not be over-interpreted.")
    add("- **Correlation is not causation.** Segment differences may reflect confounding "
        "between age, job, education, campaign timing, and contact history.")
    add("- **No modelling performed.** Phase 1 is descriptive only; no train/test split, "
        "feature selection, or model evaluation has been done.")
    add("")
    add("---")
    add("")
    add("## 10. Reproducibility")
    add("")
    add("Run everything from the project root:")
    add("")
    add("```bash")
    add("python src/phase1_exploration.py")
    add("```")
    add("")
    add("Optional arguments:")
    add("")
    add("```bash")
    add("python src/phase1_exploration.py \\")
    add("    --data data/raw/bank-full.csv \\")
    add("    --output-dir reports/phase1 \\")
    add("    --dpi 200")
    add("```")
    add("")
    add("The script validates the input path and schema, never modifies the raw CSV, and "
        "regenerates every CSV, chart, and this report deterministically (apart from the "
        "generation timestamp).")
    add("")
    add("### Generated artefacts")
    add("")
    for name in [
        "dataset_summary.csv",
        "missing_values.csv",
        "unknown_values.csv",
        "categorical_summary.csv",
        "numerical_summary.csv",
        "target_distribution.csv",
        "subscription_rates_by_segment.csv",
        "phase1_report.md",
    ]:
        add(f"- `{name}`")
    for chart_path in chart_paths:
        add(f"- `{chart_path.name}`")
    add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def write_reports(
    df: pd.DataFrame,
    path: Path,
    output_dir: Path,
    quality: dict[str, object],
    statistics: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """Write the required CSV artefacts and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    target = statistics["target"].set_index("y")
    yes_count = int(target.loc["yes", "count"])
    no_count = int(target.loc["no", "count"])
    yes_pct = float(target.loc["yes", "percentage"])
    no_pct = float(target.loc["no", "percentage"])

    summary = pd.DataFrame(
        {
            "metric": [
                "dataset_file",
                "file_size_kb",
                "rows",
                "columns",
                "numeric_columns",
                "categorical_columns",
                "memory_usage_kb",
                "total_missing_cells",
                "duplicate_rows",
                "duplicate_pct",
                "unknown_categorical_cells",
                "target_positive_count",
                "target_positive_pct",
                "target_negative_count",
                "target_negative_pct",
                "class_imbalance_ratio_no_to_yes",
                "majority_baseline_accuracy_pct",
                "pdays_minus_one_count",
                "pdays_minus_one_pct",
                "previous_zero_count",
                "pdays_previous_inconsistent_count",
            ],
            "value": [
                path.name,
                round(path.stat().st_size / 1024, 2),
                len(df),
                df.shape[1],
                len(NUMERIC_COLUMNS),
                len(CATEGORICAL_COLUMNS),
                round(df.memory_usage(deep=True).sum() / 1024, 2),
                quality["total_missing_cells"],
                quality["duplicate_rows"],
                round(quality["duplicate_pct"], 4),
                quality["unknown_cells"],
                yes_count,
                round(yes_pct, 4),
                no_count,
                round(no_pct, 4),
                round(no_count / yes_count, 4),
                round(no_pct, 4),
                quality["pdays_minus_one"],
                round(quality["pdays_minus_one_pct"], 4),
                quality["previous_zero"],
                quality["pdays_previous_inconsistent"],
            ],
        }
    )

    combined_segments = []
    for column, label in SEGMENT_SPEC:
        table = tables[column].copy()
        table.insert(0, "segment_type", label)
        table["subscription_rate_pct"] = table["subscription_rate_pct"].round(4)
        combined_segments.append(table)
    segments = pd.concat(combined_segments, ignore_index=True)

    outputs = {
        "dataset_summary": summary,
        "missing_values": quality["missing"],
        "unknown_values": quality["unknown"],
        "categorical_summary": statistics["categorical"],
        "numerical_summary": statistics["numeric"],
        "target_distribution": statistics["target"],
        "subscription_rates_by_segment": segments,
    }

    written: dict[str, Path] = {}
    for name, frame in outputs.items():
        file_path = output_dir / f"{name}.csv"
        frame.to_csv(file_path, index=False, encoding="utf-8")
        written[name] = file_path
        print(f"  saved table: {file_path.name}")

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1 EDA for the Bank Marketing Intelligence System."
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
        help="Directory for reports and charts (default: reports/phase1).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved charts (default: 150).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the complete Phase 1 pipeline."""
    args = parse_args(argv)
    data_path = Path(args.data).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    print("=" * 78)
    print("BANK MARKETING INTELLIGENCE SYSTEM - PHASE 1")
    print("Dataset understanding and exploratory data analysis")
    print("=" * 78)

    try:
        df = load_dataset(data_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print_dataset_overview(df, data_path)

    quality = analyse_quality(df)
    statistics = analyse_statistics(df)

    enriched = add_customer_segments(df)
    tables = build_rate_tables(enriched)
    print_customer_analysis(enriched, tables)

    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style(dpi=args.dpi)
    chart_paths = create_visualizations(
        enriched, statistics, tables, output_dir
    )

    print("\n" + "=" * 78)
    print("WRITING REPORTS")
    print("=" * 78)
    write_reports(df, data_path, output_dir, quality, statistics, tables)

    report = build_report(
        df, data_path, quality, statistics, tables, chart_paths
    )
    report_path = output_dir / "phase1_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  saved report: {report_path.name}")

    print("\n" + "=" * 78)
    print("PHASE 1 COMPLETE")
    print("=" * 78)
    print(f"Output directory : {output_dir}")
    print(f"CSV tables       : {len(list(output_dir.glob('*.csv')))}")
    print(f"Charts           : {len(chart_paths)}")
    print(f"Markdown report  : {report_path.name}")
    print(f"Overall subscription rate: {overall_subscription_rate(df):.2f}%")
    print("\nNo machine-learning models were trained in this phase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
