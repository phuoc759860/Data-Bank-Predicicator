"""Batch prediction page with CSV validation and downloadable results."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.components.charts import probability_histogram
from app.components.theme import (
    apply_theme,
    configure_page,
    insight_card,
    kpi_row,
    page_header,
    render_footer,
    render_plot,
    section_header,
)
from app.utils.formatting import int_text, pct_text
from src.app_utils import (
    REQUIRED_RAW_COLUMNS,
    find_unseen_categories,
    load_best_model_bundle,
    predict_batch,
    validate_raw_input,
)

configure_page("Batch Prediction")
apply_theme()

try:
    artifact = load_best_model_bundle()
except (RuntimeError, FileNotFoundError, ValueError) as exc:
    st.error("The batch prediction model could not be loaded.")
    st.exception(exc)
    st.stop()

page_header(
    "Campaign planning",
    "Score a customer list in one upload",
    "Upload a CSV in the raw Bank Marketing input format. The app validates the "
    "required pre-call fields, applies the saved Phase 4 preprocessing pipeline, "
    "and returns probabilities plus an operational yes/no decision.",
)

with st.expander("Expected CSV format", expanded=True):
    st.write(
        "The file must contain these raw columns. Column names are "
        "case-insensitive; values are validated before scoring."
    )
    st.code(", ".join(REQUIRED_RAW_COLUMNS), language="text")
    st.caption(
        "`duration` is not required and is ignored if present. `y` is optional "
        "and is ignored. Binary fields accept yes/no or 1/0. Numeric fields must "
        "be numeric."
    )

uploaded_file = st.file_uploader(
    "Upload customer CSV",
    type=["csv"],
    help="Try a CSV exported from the raw bank-full.csv structure or a subset of it.",
)

if uploaded_file is None:
    insight_card(
        "No file uploaded yet",
        "The batch scorer does not silently invent customers or predictions. "
        "Upload a real CSV file to generate results.",
    )
    render_footer()
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes.strip():
        raise ValueError("The uploaded file is empty.")
    uploaded_df = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python",
    )
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
    st.error("The uploaded file could not be read as a CSV.")
    st.error(str(exc))
    render_footer()
    st.stop()

st.caption(f"Loaded {len(uploaded_df):,} rows and {len(uploaded_df.columns)} columns.")

try:
    validated = validate_raw_input(uploaded_df)
    unseen = find_unseen_categories(validated)
    predictions = predict_batch(validated)
except ValueError as exc:
    st.error("The uploaded data failed validation. No predictions were produced.")
    st.error(str(exc))
    with st.expander("Required fields and expected types"):
        st.markdown(
            "- Required columns: `" + "`, `".join(REQUIRED_RAW_COLUMNS) + "`\n"
            "- Numeric: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`\n"
            "- Binary: `default`, `housing`, `loan` (yes/no or 1/0)\n"
            "- Categorical: `job`, `marital`, `education`, `contact`, `month`, `poutcome`"
        )
    render_footer()
    st.stop()

if unseen:
    st.warning("Some categorical values were not seen in the training data.")
    for column, values in unseen.items():
        st.write(f"- `{column}`: {', '.join(values[:10])}")
    st.caption(
        "The saved OneHotEncoder uses handle_unknown='ignore'. Predictions "
        "are still produced, but these rows may have weaker evidence."
    )

input_for_output = validated.reset_index(drop=True)
prediction_output = predictions.reset_index(drop=True)
results = pd.concat([input_for_output, prediction_output], axis=1)

positive_count = int((results["predicted_label"] == 1).sum())
positive_rate = 100.0 * positive_count / len(results) if len(results) else 0.0
mean_probability = float(results["predicted_probability"].mean()) if len(results) else 0.0

kpi_row(
    [
        {
            "label": "Rows scored",
            "value": int_text(len(results)),
            "note": "Validated customer records",
        },
        {
            "label": "Predicted subscribers",
            "value": int_text(positive_count),
            "note": f"At threshold {float(artifact['threshold']):.2f}",
        },
        {
            "label": "Predicted contact share",
            "value": pct_text(positive_rate),
            "note": "Share of uploaded rows above threshold",
        },
        {
            "label": "Mean probability",
            "value": pct_text(100.0 * mean_probability),
            "note": "Average model score in this batch",
        },
    ]
)

section_header(
    "Batch results",
    "The original input columns are preserved, and model outputs are appended.",
)
st.dataframe(
    results.head(200),
    hide_index=True,
    width="stretch",
)
st.caption(f"Showing 200 of {len(results):,} scored rows in the preview.")

left, right = st.columns([1.35, 1])
with left:
    render_plot(probability_histogram(results["predicted_probability"], float(artifact["threshold"])))
with right:
    st.metric("Model", artifact["model_name"])
    st.metric("Decision threshold", f"{float(artifact['threshold']):.2f}")
    st.metric("Duration used?", "No")
    st.caption(
        "Rows above the threshold are marked 'Yes'. This is a prioritisation "
        "output, not a guarantee of subscription."
    )

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions as CSV",
    data=csv_bytes,
    file_name="bank_marketing_predictions.csv",
    mime="text/csv",
    help="Includes original columns, probability, predicted outcome, threshold, and model name.",
)

insight_card(
    "How to use the batch file",
    "The `predicted_probability` column ranks customers for outreach. The "
    "`predicted_subscription` column applies the Phase 4 threshold. Review the "
    "probability distribution before choosing call capacity, and remember that "
    "the model estimates association rather than causing a customer to subscribe.",
    tone="warning",
)
render_footer()
