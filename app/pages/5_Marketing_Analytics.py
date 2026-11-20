"""Marketing analytics page with observed, non-causal segment comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.charts import subscription_rate_bar
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

configure_page("Marketing Analytics")
apply_theme()

try:
    data = load_raw_data()
except (RuntimeError, FileNotFoundError, ValueError) as exc:
    st.error("The dataset could not be loaded.")
    st.exception(exc)
    st.stop()

filtered = render_sidebar_filters(data, "analytics")
min_group = st.sidebar.slider(
    "Minimum group size for charts",
    min_value=30,
    max_value=500,
    value=100,
    step=10,
    help="Groups below this size are hidden so small-sample rates do not dominate the view.",
)

page_header(
    "Evidence for campaign decisions",
    "Marketing analytics",
    "Compare observed subscription rates across customer profiles and campaign "
    "history. These are descriptive associations in historical data, not proof "
    "that changing a customer attribute or channel would cause a subscription.",
)

if filtered.empty:
    st.warning("No customers match the current filters.")
    render_footer()
    st.stop()

summary = overall_summary(filtered)
job_summary = subscription_rate_by(filtered, "job", min_customers=min_group)
age_summary = subscription_rate_by(filtered, "age_group", min_customers=min_group)
education_summary = subscription_rate_by(filtered, "education", min_customers=min_group)
marital_summary = subscription_rate_by(filtered, "marital", min_customers=min_group)
poutcome_summary = subscription_rate_by(filtered, "poutcome", min_customers=min_group)

top_job = job_summary.iloc[0] if not job_summary.empty else None
top_age = age_summary.sort_values("subscription_rate_pct", ascending=False).iloc[0] if not age_summary.empty else None

kpi_row(
    [
        {
            "label": "Customers in view",
            "value": int_text(summary["total"]),
            "note": "After sidebar filters",
        },
        {
            "label": "Observed subscription rate",
            "value": pct_text(summary["subscription_rate"], 2),
            "note": "Descriptive historical rate",
        },
        {
            "label": "Highest job rate",
            "value": f"{top_job['subscription_rate_pct']:.1f}%" if top_job is not None else "—",
            "note": f"{top_job['job']} (n={int(top_job['n_customers']):,})" if top_job is not None else "No group passes the size filter",
        },
        {
            "label": "Highest age-band rate",
            "value": f"{top_age['subscription_rate_pct']:.1f}%" if top_age is not None else "—",
            "note": f"{top_age['age_group']} (n={int(top_age['n_customers']):,})" if top_age is not None else "No group passes the size filter",
        },
    ]
)

profile_tab, campaign_tab, evidence_tab = st.tabs(
    ["Customer profile", "Campaign history", "Evidence and caveats"]
)

with profile_tab:
    section_header(
        "Who subscribes?",
        "Each bar is an observed rate within a group. Hover for sample size and "
        "subscriber count.",
    )
    left, right = st.columns(2)
    with left:
        render_plot(
            subscription_rate_bar(
                filtered,
                "job",
                "Subscription rate by job",
                min_customers=min_group,
            )
        )
    with right:
        render_plot(
            subscription_rate_bar(
                filtered,
                "age_group",
                "Subscription rate by age group",
                min_customers=min_group,
            )
        )
    left, right = st.columns(2)
    with left:
        render_plot(
            subscription_rate_bar(
                filtered,
                "education",
                "Subscription rate by education",
                min_customers=min_group,
            )
        )
    with right:
        render_plot(
            subscription_rate_bar(
                filtered,
                "marital",
                "Subscription rate by marital status",
                min_customers=min_group,
            )
        )

with campaign_tab:
    section_header(
        "Campaign history and contact context",
        "Previous outcomes and contact context are correlated with subscription, "
        "but they are not randomised treatments.",
    )
    left, right = st.columns(2)
    with left:
        render_plot(
            subscription_rate_bar(
                filtered,
                "poutcome",
                "Subscription rate by previous campaign outcome",
                min_customers=min_group,
            )
        )
    with right:
        render_plot(
            subscription_rate_bar(
                filtered,
                "contact",
                "Subscription rate by contact channel",
                min_customers=min_group,
            )
        )
    left, right = st.columns(2)
    with left:
        render_plot(
            subscription_rate_bar(
                filtered,
                "month",
                "Subscription rate by contact month",
                min_customers=min_group,
            )
        )
    with right:
        render_plot(
            subscription_rate_bar(
                filtered,
                "campaign_group",
                "Subscription rate by current-campaign contact frequency",
                min_customers=min_group,
                sort_by_rate=False,
            )
        )

with evidence_tab:
    section_header(
        "What the evidence supports",
        "Use these patterns to ask better questions and design tests.",
    )
    if top_job is not None:
        insight_card(
            "Observed profile pattern",
            f"The highest job-group rate in the current view is {top_job['job']} "
            f"at {top_job['subscription_rate_pct']:.1f}% "
            f"(n={int(top_job['n_customers']):,}). Compare it with the overall "
            f"filtered rate of {summary['subscription_rate']:.2f}%. Job is "
            "associated with age, education, balance, and campaign history, so this "
            "is not a standalone lever.",
        )
    if not poutcome_summary.empty:
        best_poutcome = poutcome_summary.iloc[0]
        insight_card(
            "Previous campaign outcome",
            f"Customers with previous outcome '{best_poutcome['poutcome']}' show an "
            f"observed rate of {best_poutcome['subscription_rate_pct']:.1f}% "
            f"(n={int(best_poutcome['n_customers']):,}). This is a strong descriptive "
            "signal for warm-lead prioritisation, but it may also reflect prior "
            "customer engagement and selection effects.",
            tone="warning",
        )
    if not age_summary.empty:
        insight_card(
            "Age-band comparison",
            "Age-band differences are descriptive. Before targeting an age band, "
            "check whether the pattern persists after controlling for campaign "
            "history, contact channel, and offer type.",
        )

    st.warning(
        "Correlation is not causation. A high observed rate for a job, month, "
        "channel, or previous outcome does not mean that changing that factor "
        "would cause a customer to subscribe. Use controlled tests or holdout "
        "campaigns before making causal claims."
    )

    st.markdown("#### Group summaries")
    selected_summary = st.selectbox(
        "Choose a dimension",
        [
            "job",
            "age_group",
            "education",
            "marital",
            "poutcome",
            "contact",
            "month",
            "campaign_group",
        ],
        key="analytics_summary_dimension",
    )
    summary_table = subscription_rate_by(
        filtered, selected_summary, min_customers=min_group
    )
    st.dataframe(
        summary_table.rename(
            columns={
                "n_customers": "Customers",
                "subscribers": "Subscribers",
                "subscription_rate_pct": "Observed subscription rate (%)",
            }
        ),
        hide_index=True,
        width="stretch",
    )

render_footer()
