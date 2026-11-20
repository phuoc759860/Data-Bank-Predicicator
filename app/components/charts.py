"""Plotly chart builders shared by the Phase 5 pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.components.theme import (
    ACCENT_COLORS,
    CLUSTER_COLORS,
    GOLD,
    NAVY,
    SUBSCRIPTION_COLORS,
    TEAL,
    style_figure,
)
from src.app_utils import subscription_rate_by


def subscription_donut(df: pd.DataFrame) -> go.Figure:
    """Yes/No subscription distribution as a clean donut chart."""
    counts = (
        df["subscribed"]
        .map({0: "No", 1: "Yes"})
        .value_counts()
        .reindex(["No", "Yes"], fill_value=0)
        .reset_index()
    )
    counts.columns = ["Subscription", "Customers"]
    figure = px.pie(
        counts,
        names="Subscription",
        values="Customers",
        hole=0.58,
        color="Subscription",
        color_discrete_map=SUBSCRIPTION_COLORS,
        category_orders={"Subscription": ["No", "Yes"]},
    )
    total = int(counts["Customers"].sum())
    yes_count = int(counts.loc[counts["Subscription"] == "Yes", "Customers"].sum())
    rate = 100.0 * yes_count / total if total else 0.0
    figure.add_annotation(
        text=f"<b>{rate:.1f}%</b><br><span style='font-size:11px'>subscribed</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 19, "color": NAVY},
    )
    figure.update_traces(
        textinfo="percent",
        textfont_size=12,
        marker={"line": {"color": "#FFFFFF", "width": 2}},
        hovertemplate="%{label}: %{value:,} customers<extra></extra>",
    )
    return style_figure(figure, "Subscription distribution", height=320)


def age_distribution(df: pd.DataFrame) -> go.Figure:
    """Age distribution split by observed subscription outcome."""
    display = df.assign(
        Subscription=df["subscribed"].map({0: "No", 1: "Yes"})
    )
    figure = px.histogram(
        display,
        x="age",
        color="Subscription",
        nbins=30,
        barmode="overlay",
        opacity=0.78,
        color_discrete_map=SUBSCRIPTION_COLORS,
        category_orders={"Subscription": ["No", "Yes"]},
        labels={"age": "Age (years)", "count": "Customers"},
    )
    figure.update_traces(hovertemplate="Age %{x}: %{y:,} customers<extra></extra>")
    return style_figure(figure, "Customer age distribution", height=320)


def count_bar(
    df: pd.DataFrame,
    column: str,
    title: str,
    top_n: int | None = None,
    height: int = 330,
) -> go.Figure:
    """Horizontal bar chart of customer counts for a categorical field."""
    counts = (
        df[column]
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="Customers")
    )
    if top_n:
        counts = counts.head(top_n)
    counts = counts.sort_values("Customers", ascending=True)
    figure = px.bar(
        counts,
        x="Customers",
        y=column,
        orientation="h",
        text="Customers",
        color_discrete_sequence=[TEAL],
        labels={column: column.replace("_", " ").title()},
    )
    figure.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate="%{y}: %{x:,} customers<extra></extra>",
        marker_line_width=0,
    )
    figure.update_layout(xaxis_title="Customers", yaxis_title=None)
    return style_figure(figure, title, height=height)


def subscription_rate_bar(
    df: pd.DataFrame,
    column: str,
    title: str,
    min_customers: int = 1,
    height: int = 340,
    sort_by_rate: bool = True,
) -> go.Figure:
    """Observed subscription rate by group, annotated with group sample size."""
    summary = subscription_rate_by(df, column, min_customers=min_customers)
    if summary.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No groups meet the current filter/sample-size threshold.",
            showarrow=False,
            font={"color": "#5C677D"},
        )
        figure.update_layout(xaxis={"visible": False}, yaxis={"visible": False})
        return style_figure(figure, title, height=height, showlegend=False)

    if sort_by_rate:
        summary = summary.sort_values("subscription_rate_pct", ascending=True)
    summary = summary.copy()
    summary["label"] = summary.apply(
        lambda row: f"{row['subscription_rate_pct']:.1f}%  (n={int(row['n_customers']):,})",
        axis=1,
    )
    figure = px.bar(
        summary,
        x="subscription_rate_pct",
        y=column,
        orientation="h",
        text="label",
        color="subscription_rate_pct",
        color_continuous_scale=[[0, "#DCEAF4"], [0.55, "#82B9B4"], [1, TEAL]],
        labels={
            column: column.replace("_", " ").title(),
            "subscription_rate_pct": "Subscription rate (%)",
        },
        custom_data=["n_customers", "subscribers"],
    )
    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "%{y}<br>Subscription rate: %{x:.1f}%<br>"
            "Customers: %{customdata[0]:,}<br>Subscribers: %{customdata[1]:,}"
            "<extra></extra>"
        ),
        marker_line_width=0,
    )
    figure.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Observed subscription rate (%)",
        yaxis_title=None,
        xaxis_range=[0, max(20, float(summary["subscription_rate_pct"].max()) * 1.35)],
    )
    return style_figure(figure, title, height=height, showlegend=False)


def probability_histogram(probabilities: pd.Series, threshold: float) -> go.Figure:
    """Distribution of model scores with the selected decision threshold."""
    figure = px.histogram(
        x=probabilities,
        nbins=30,
        labels={"x": "Predicted probability of subscription", "y": "Customers"},
        color_discrete_sequence=[TEAL],
    )
    figure.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=GOLD,
        annotation_text=f"Threshold {threshold:.2f}",
        annotation_position="top",
        annotation_font_color=GOLD,
    )
    figure.update_traces(
        hovertemplate="Probability %{x:.3f}: %{y:,} customers<extra></extra>",
        marker_line_width=0,
    )
    return style_figure(figure, "Score distribution", height=330, showlegend=False)


def cluster_donut(profiles: pd.DataFrame) -> go.Figure:
    """Cluster size distribution."""
    display = profiles.assign(
        Segment=profiles["cluster"].map(lambda value: f"Cluster {int(value)}")
    )
    figure = px.pie(
        display,
        names="Segment",
        values="n_customers",
        hole=0.55,
        color="Segment",
        color_discrete_sequence=CLUSTER_COLORS,
    )
    figure.update_traces(
        textinfo="percent",
        marker={"line": {"color": "#FFFFFF", "width": 2}},
        hovertemplate="%{label}: %{value:,} customers<extra></extra>",
    )
    figure.add_annotation(
        text=f"<b>{int(profiles['n_customers'].sum()):,}</b><br>"
        "<span style='font-size:11px'>training customers</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 17, "color": NAVY},
    )
    return style_figure(figure, "Cluster distribution", height=330)


def cluster_rate_bar(profiles: pd.DataFrame) -> go.Figure:
    """Observed subscription rate by cluster (post-hoc descriptive view)."""
    display = profiles.copy()
    display["label"] = display["cluster"].map(lambda value: f"Cluster {int(value)}")
    display = display.sort_values("observed_subscription_rate_pct", ascending=True)
    figure = px.bar(
        display,
        x="observed_subscription_rate_pct",
        y="label",
        orientation="h",
        text=display["observed_subscription_rate_pct"].map(lambda value: f"{value:.1f}%"),
        color="observed_subscription_rate_pct",
        color_continuous_scale=[[0, "#DCEAF4"], [1, TEAL]],
        labels={
            "label": "Segment",
            "observed_subscription_rate_pct": "Observed subscription rate (%)",
        },
        custom_data=["n_customers"],
    )
    figure.update_traces(
        textposition="outside",
        hovertemplate=(
            "%{y}<br>Observed rate: %{x:.1f}%<br>Customers: %{customdata[0]:,}"
            "<extra></extra>"
        ),
        marker_line_width=0,
    )
    figure.update_layout(
        coloraxis_showscale=False,
        xaxis_range=[0, max(20, float(display["observed_subscription_rate_pct"].max()) * 1.35)],
    )
    return style_figure(figure, "Observed subscription rate by segment", height=330, showlegend=False)


def cluster_scatter(plot_df: pd.DataFrame, explained_variance: list[float] | None = None) -> go.Figure:
    """Interactive PCA view of actual customer clusters."""
    display = plot_df.copy()
    display["Segment"] = display["cluster"].map(lambda value: f"Cluster {int(value)}")
    display["Subscription"] = display["subscribed"].map({0: "No", 1: "Yes"})
    figure = px.scatter(
        display,
        x="pc1",
        y="pc2",
        color="Segment",
        hover_data={
            "age": True,
            "balance": ":.0f",
            "job": True,
            "Subscription": True,
            "pc1": False,
            "pc2": False,
        },
        opacity=0.55,
        color_discrete_sequence=CLUSTER_COLORS,
        labels={"pc1": "Principal component 1", "pc2": "Principal component 2"},
        render_mode="webgl",
    )
    figure.update_traces(marker={"size": 5, "line": {"width": 0}})
    if explained_variance is not None and len(explained_variance) >= 2:
        figure.update_xaxes(
            title=f"Principal component 1 ({explained_variance[0] * 100:.1f}% variance)"
        )
        figure.update_yaxes(
            title=f"Principal component 2 ({explained_variance[1] * 100:.1f}% variance)"
        )
    return style_figure(
        figure,
        "Customer segments in PCA space",
        height=500,
    )


def rate_by_campaign_group(df: pd.DataFrame) -> go.Figure:
    """Convenience wrapper used by campaign performance panels."""
    return subscription_rate_bar(
        df,
        "campaign_group",
        "Subscription rate by current-campaign contact frequency",
        height=320,
        sort_by_rate=False,
    )
