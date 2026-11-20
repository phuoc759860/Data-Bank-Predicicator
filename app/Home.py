"""Landing page for the Bank Marketing Intelligence application."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.components.theme import (
    apply_theme,
    configure_page,
    insight_card,
    kpi_row,
    page_header,
    render_footer,
    section_header,
)
from app.utils.formatting import int_text, pct_text
from src.app_utils import (
    check_required_artifacts,
    load_best_model_bundle,
    load_kmeans_bundle,
    load_model_metadata,
    load_raw_data,
    overall_summary,
)

configure_page("Home")
apply_theme()

try:
    check_required_artifacts()
    data = load_raw_data()
    metadata = load_model_metadata()
    best_model = load_best_model_bundle()
    kmeans = load_kmeans_bundle()
except (RuntimeError, FileNotFoundError, ValueError) as exc:
    st.error("The application cannot start with the current artifacts.")
    st.exception(exc)
    st.stop()

summary = overall_summary(data)
test_metrics = metadata.get("test_metrics_at_selected_threshold", {})

page_header(
    "Bank Marketing Intelligence System",
    "Use evidence, not guesswork, to target the next campaign.",
    "A Streamlit decision-support application built on the real UCI Bank "
    "Marketing dataset, the selected Phase 4 Random Forest, and the Phase 3 "
    "K-Means customer segmentation.",
    hero=True,
)

kpi_row(
    [
        {
            "label": "Customers in dataset",
            "value": int_text(summary["total"]),
            "note": "Original UCI Bank Marketing records",
        },
        {
            "label": "Observed subscribers",
            "value": int_text(summary["subscribers"]),
            "note": "Descriptive outcome, not a model prediction",
        },
        {
            "label": "Subscription rate",
            "value": pct_text(summary["subscription_rate"]),
            "note": "Overall observed rate in the dataset",
        },
        {
            "label": "Selected model F1",
            "value": f"{float(test_metrics.get('f1', 0)):.4f}",
            "note": "Phase 4 test set, yes class",
        },
    ]
)

section_header(
    "System status",
    "These values are read directly from the project artifacts at runtime.",
)
status_columns = st.columns(3)
with status_columns[0]:
    insight_card(
        "Dataset ready",
        f"{int_text(summary['total'])} customers loaded from data/raw/bank-full.csv. "
        "The original file is preserved and not modified.",
    )
with status_columns[1]:
    insight_card(
        "Prediction model ready",
        f"{best_model['model_name']} loaded with decision threshold "
        f"{float(best_model['threshold']):.2f}. The saved Phase 2 preprocessing "
        "pipeline is reused; duration is excluded.",
    )
with status_columns[2]:
    insight_card(
        "Segmentation ready",
        f"Phase 3 K-Means loaded with {kmeans.named_steps['kmeans'].n_clusters} "
        "clusters. Segmentation inputs contain no target label and no duration.",
    )

section_header(
    "What you can do",
    "Start with the Executive Dashboard, then move from insight to action.",
)
feature_columns = st.columns(3)
with feature_columns[0]:
    with st.container(border=True):
        st.markdown("#### Explore the campaign")
        st.write(
            "Filter customers by profile and contact channel. Compare subscription "
            "rates across jobs, education, age groups, and campaign history."
        )
with feature_columns[1]:
    with st.container(border=True):
        st.markdown("#### Score a customer")
        st.write(
            "Enter pre-call customer information and get a probability estimate. "
            "The outcome uses the real Phase 4 model and its selected threshold."
        )
with feature_columns[2]:
    with st.container(border=True):
        st.markdown("#### Plan the audience")
        st.write(
            "Review K-Means segments, upload a CSV for batch scoring, and download "
            "a contact list with probabilities for campaign planning."
        )

section_header(
    "Selected model at a glance",
    "Test-set values reported by the Phase 4 evaluation. Accuracy is not the "
    "selection criterion for this imbalanced target.",
)
model_summary = pd.DataFrame(
    [
        {
            "Model": best_model["model_name"],
            "Decision threshold": f"{float(best_model['threshold']):.2f}",
            "Accuracy": f"{float(test_metrics.get('accuracy', 0)):.4f}",
            "Precision (yes)": f"{float(test_metrics.get('precision', 0)):.4f}",
            "Recall (yes)": f"{float(test_metrics.get('recall', 0)):.4f}",
            "F1 (yes)": f"{float(test_metrics.get('f1', 0)):.4f}",
            "ROC-AUC": f"{float(test_metrics.get('roc_auc', 0)):.4f}",
            "PR-AUC": f"{float(test_metrics.get('average_precision', 0)):.4f}",
        }
    ]
)
st.dataframe(model_summary, hide_index=True, width="stretch")

insight_card(
    "How to read the probability",
    "A model probability is an estimated likelihood based on historical patterns, "
    "not a guarantee that a specific customer will subscribe. Use it to prioritise "
    "outreach and combine it with campaign capacity, offer economics, and human judgement.",
    tone="warning",
)
st.caption("Use the sidebar to open the dashboard, prediction, batch, segmentation, and analytics pages.")
render_footer()
