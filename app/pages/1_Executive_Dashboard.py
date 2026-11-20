"""Executive dashboard page."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.charts import (
    age_distribution,
    count_bar,
    rate_by_campaign_group,
    subscription_donut,
    subscription_rate_bar,
)
from app.components.filters import render_sidebar_filters
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
from src.app_utils import load_raw_data, overall_summary, subscription_rate_by

configure_page("Executive Dashboard")
apply_theme()

try:
    data = load_raw_data()
except (RuntimeError, FileNotFoundError, ValueError) as exc:
    st.error("The dataset could not be loaded.")
    st.exception(exc)
    st.stop()

filtered = render_sidebar_filters(data, "executive")

page_header(
    "Executive overview",
    "Campaign performance at a glance",
    "Explore the actual customer base, the observed subscription outcome, and "
    "campaign performance by customer and contact dimensions. Use the sidebar "
    "filters to focus the view.",
)

if filtered.empty:
    st.warning("No customers match the current filters. Widen the selection in the sidebar.")
    render_footer()
    st.stop()

summary = overall_summary(filtered)

kpi_row(
    [
        {
            "label": "Filtered customers",
            "value": int_text(summary["total"]),
            "note": "Rows matching the current sidebar filters",
        },
        {
            "label": "Observed subscribers",
            "value": int_text(summary["subscribers"]),
            "note": "Customers with y = yes",
        },
        {
            "label": "Non-subscribers",
            "value": int_text(summary["non_subscribers"]),
            "note": "Customers with y = no",
        },
        {
            "label": "Subscription rate",
            "value": pct_text(summary["subscription_rate"], 2),
            "note": "Observed rate in the filtered view",
        },
    ]
)

section_header(
    "Customer profile",
    "The subscription outcome is shown as an observed distribution, not as a causal effect.",
)
left, right = st.columns([1, 1.35])
with left:
    render_plot(subscription_donut(filtered))
with right:
    render_plot(age_distribution(filtered))

left, right = st.columns(2)
with left:
    render_plot(count_bar(filtered, "job", "Customer distribution by job"))
with right:
    render_plot(count_bar(filtered, "education", "Customer distribution by education"))

section_header(
    "Campaign performance",
    "Rates are observed shares within each group. Group sizes are shown in the labels.",
)
left, right = st.columns(2)
with left:
    render_plot(
        subscription_rate_bar(
            filtered,
            "contact",
            "Subscription rate by contact channel",
        )
    )
with right:
    render_plot(
        subscription_rate_bar(
            filtered,
            "poutcome",
            "Subscription rate by previous campaign outcome",
        )
    )

left, right = st.columns(2)
with left:
    render_plot(
        subscription_rate_bar(
            filtered,
            "month",
            "Subscription rate by contact month",
        )
    )
with right:
    render_plot(rate_by_campaign_group(filtered))

section_header("Signals to investigate")
job_summary = subscription_rate_by(filtered, "job", min_customers=100)
contact_summary = subscription_rate_by(filtered, "contact", min_customers=100)
if not job_summary.empty:
    top_job = job_summary.iloc[0]
    insight_card(
        "Highest observed job-group rate",
        f"{top_job['job']} has an observed subscription rate of "
        f"{top_job['subscription_rate_pct']:.1f}% across "
        f"{int(top_job['n_customers']):,} filtered customers. "
        "This is a descriptive comparison; job is correlated with many other "
        "customer and campaign factors.",
    )
if not contact_summary.empty:
    top_contact = contact_summary.iloc[0]
    insight_card(
        "Contact-channel pattern",
        f"{top_contact['contact']} shows the highest observed rate in the current "
        f"filter at {top_contact['subscription_rate_pct']:.1f}% "
        f"(n={int(top_contact['n_customers']):,}). Do not interpret the channel "
        "as a causal lever without a controlled campaign test.",
        tone="warning",
    )

with st.expander("View filtered customer records"):
    preview_columns = [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "contact",
        "campaign",
        "previous",
        "poutcome",
        "y",
    ]
    st.dataframe(
        filtered[preview_columns].head(200),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Showing 200 of {len(filtered):,} filtered rows.")

render_footer()
