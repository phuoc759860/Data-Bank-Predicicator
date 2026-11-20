"""Customer segmentation page using the real Phase 3 K-Means artifact."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.components.charts import cluster_donut, cluster_rate_bar, cluster_scatter
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
    build_cluster_projection,
    get_kmeans_features,
    load_cluster_profiles,
    load_clustering_metrics,
    load_training_cluster_data,
)

configure_page("Customer Segmentation")
apply_theme()

try:
    cluster_data = load_training_cluster_data()
    profiles = load_cluster_profiles()
    metric_table = load_clustering_metrics()
    projection, explained = build_cluster_projection()
    kmeans_features = get_kmeans_features()
except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
    st.error("The K-Means segmentation artifacts could not be loaded.")
    st.exception(exc)
    st.stop()

selected_metrics = metric_table[metric_table["k"] == 4]
selected_metrics = selected_metrics.iloc[0] if not selected_metrics.empty else metric_table.iloc[0]
overall_rate = float(profiles["overall_subscription_rate_pct"].iloc[0])

page_header(
    "Unsupervised segmentation",
    "Understand the customer base",
    "The Phase 3 K-Means model groups customers by profile and prior campaign "
    "history. Segments are descriptive; they do not use the subscription target "
    "to form the clusters.",
)

kpi_row(
    [
        {
            "label": "Training customers",
            "value": int_text(len(cluster_data)),
            "note": "Reproduced Phase 2 training split",
        },
        {
            "label": "Segments",
            "value": "4",
            "note": "Selected k in Phase 3",
        },
        {
            "label": "Silhouette score",
            "value": f"{float(selected_metrics['silhouette_score']):.4f}",
            "note": "Training-sample silhouette",
        },
        {
            "label": "Inertia",
            "value": f"{float(selected_metrics['inertia']):,.0f}",
            "note": "Within-cluster sum of squares",
        },
    ]
)

section_header(
    "Segment distribution",
    "Cluster sizes are actual assignments from the saved K-Means pipeline on the training split.",
)
left, right = st.columns([1, 1.35])
with left:
    render_plot(cluster_donut(profiles))
with right:
    render_plot(cluster_rate_bar(profiles))

section_header(
    "Where the segments sit",
    "PCA is used only for a two-dimensional display; K-Means operates in the full "
    "scaled and encoded feature space.",
)
render_plot(cluster_scatter(projection, explained))

section_header(
    "Segment profiles",
    "Counts, shares, and observed subscription rates are computed from the actual "
    "training customers. The observed rate is a post-hoc descriptive check.",
)
display_profiles = profiles[
    [
        "cluster",
        "n_customers",
        "share_pct",
        "age_mean",
        "balance_median",
        "top_job",
        "pct_housing_loan",
        "pct_previous_contact",
        "observed_subscription_rate_pct",
    ]
].copy()
display_profiles.columns = [
    "Cluster",
    "Customers",
    "Share (%)",
    "Mean age",
    "Median balance (EUR)",
    "Top job",
    "Housing loan (%)",
    "Previous contact (%)",
    "Observed subscription (%)",
]
st.dataframe(
    display_profiles,
    hide_index=True,
    width="stretch",
    column_config={
        "Customers": st.column_config.NumberColumn(format="%d"),
        "Share (%)": st.column_config.NumberColumn(format="%.2f"),
        "Mean age": st.column_config.NumberColumn(format="%.1f"),
        "Median balance (EUR)": st.column_config.NumberColumn(format="%.0f"),
        "Housing loan (%)": st.column_config.NumberColumn(format="%.1f"),
        "Previous contact (%)": st.column_config.NumberColumn(format="%.1f"),
        "Observed subscription (%)": st.column_config.NumberColumn(format="%.2f"),
    },
)

for _, row in profiles.sort_values("cluster").iterrows():
    cluster_number = int(row["cluster"])
    gap = float(row["observed_subscription_rate_pct"]) - overall_rate
    st.markdown(f"#### Cluster {cluster_number}")
    st.write(
        f"{int(row['n_customers']):,} customers ({row['share_pct']:.1f}% of the "
        f"training base). Mean age {row['age_mean']:.1f}; median balance "
        f"EUR {row['balance_median']:,.0f}; most common job {row['top_job']} "
        f"({row['top_job_pct']:.1f}% of the segment); housing-loan share "
        f"{row['pct_housing_loan']:.1f}%; previous-contact share "
        f"{row['pct_previous_contact']:.1f}%."
    )
    if gap > 2:
        st.success(
            f"Observed subscription rate: {row['observed_subscription_rate_pct']:.2f}% "
            f"({gap:+.2f} pp versus the overall {overall_rate:.2f}%). "
            "This is an above-average segment and a candidate for priority outreach "
            "or a higher-touch proposition."
        )
    elif gap < -2:
        st.info(
            f"Observed subscription rate: {row['observed_subscription_rate_pct']:.2f}% "
            f"({gap:+.2f} pp versus the overall {overall_rate:.2f}%). "
            "This is a below-average segment. Lower-cost channels or a different "
            "proposition may be a better fit than expensive high-touch calls."
        )
    else:
        st.info(
            f"Observed subscription rate: {row['observed_subscription_rate_pct']:.2f}% "
            f"({gap:+.2f} pp versus the overall {overall_rate:.2f}%). "
            "This segment is close to the overall rate and can support volume campaigns."
        )

section_header("Input features used by K-Means")
st.write(
    "The saved Phase 3 pipeline expects these engineered customer and campaign-history "
    "features. `y` and `duration` are not inputs."
)
st.code(", ".join(kmeans_features), language="text")

insight_card(
    "How to use segmentation with prediction",
    "First rank customers by the supervised Phase 4 subscription probability. "
    "Then use the cluster profile to choose channel, offer, and contact intensity. "
    "The segments describe who customers are; the classifier estimates who is "
    "likely to subscribe.",
    tone="warning",
)

st.download_button(
    "Download cluster profiles",
    data=profiles.to_csv(index=False).encode("utf-8"),
    file_name="bank_marketing_cluster_profiles.csv",
    mime="text/csv",
)

render_footer()
