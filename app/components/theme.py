"""Consistent visual language for the Bank Marketing Intelligence app."""

from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go
import streamlit as st

NAVY = "#0B2545"
TEAL = "#1D7874"
GOLD = "#C9A227"
SLATE = "#5C677D"
BACKGROUND = "#F4F7FB"
CARD = "#FFFFFF"
GRID = "#E5ECF4"

ACCENT_COLORS = ["#1D7874", "#C9A227", "#4C72B0", "#C44E52", "#7A5195", "#7F8C8D"]
SUBSCRIPTION_COLORS = {"No": "#8FA3BF", "Yes": TEAL}
CLUSTER_COLORS = ["#1D7874", "#C9A227", "#4C72B0", "#C44E52"]


def configure_page(title: str, icon: str = "🏦") -> None:
    """Set the Streamlit page configuration consistently."""
    st.set_page_config(
        page_title=f"{title} | Bank Marketing Intelligence",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_theme() -> None:
    """Inject the application's professional banking dashboard styling."""
    st.markdown(
        """
        <style>
        :root {
            --navy: #0B2545;
            --navy-soft: #13315C;
            --teal: #1D7874;
            --gold: #C9A227;
            --slate: #5C677D;
            --background: #F4F7FB;
            --card: #FFFFFF;
            --line: #E3E9F2;
        }

        .stApp {
            background: var(--background);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.7rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, h4 {
            color: var(--navy);
            letter-spacing: -0.02em;
        }

        [data-testid="stSidebar"] {
            background: var(--navy);
            border-right: 1px solid #1B3A5C;
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #E9F0F8;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
            border-radius: 9px;
            margin: 0.12rem 0.4rem;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.16);
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #1B263B !important;
        }

        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * ,
        [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] * {
            color: #1B263B !important;
        }

        .app-hero {
            background: linear-gradient(135deg, #0B2545 0%, #13315C 62%, #1D4E6B 100%);
            border-radius: 20px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 14px 34px rgba(11, 37, 69, 0.18);
        }

        .app-hero .eyebrow {
            color: #E8C766;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .app-hero .hero-title {
            color: #FFFFFF;
            font-size: clamp(1.75rem, 2.6vw, 2.65rem);
            line-height: 1.12;
            font-weight: 700;
            margin: 0 0 0.65rem 0;
        }

        .app-hero .hero-subtitle {
            color: #CDDCEE;
            font-size: 1rem;
            max-width: 58rem;
            line-height: 1.55;
            margin: 0;
        }

        .page-eyebrow {
            color: var(--teal);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .page-title {
            color: var(--navy);
            font-size: clamp(1.55rem, 2.2vw, 2.15rem);
            line-height: 1.15;
            font-weight: 750;
            margin: 0 0 0.35rem 0;
        }

        .page-subtitle {
            color: var(--slate);
            font-size: 0.97rem;
            line-height: 1.55;
            max-width: 70rem;
            margin: 0 0 1.1rem 0;
        }

        .kpi-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.05rem 1.15rem;
            min-height: 118px;
            box-shadow: 0 7px 20px rgba(11, 37, 69, 0.055);
        }

        .kpi-label {
            color: var(--slate);
            font-size: 0.77rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.42rem;
        }

        .kpi-value {
            color: var(--navy);
            font-size: 1.85rem;
            line-height: 1.05;
            font-weight: 760;
            letter-spacing: -0.035em;
        }

        .kpi-note {
            color: #7A879B;
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.45rem;
        }

        .section-title {
            color: var(--navy);
            font-size: 1.12rem;
            font-weight: 740;
            margin: 1.25rem 0 0.25rem 0;
        }

        .section-subtitle {
            color: var(--slate);
            font-size: 0.88rem;
            line-height: 1.5;
            margin: 0 0 0.6rem 0;
        }

        .insight-card {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-left: 4px solid var(--teal);
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            margin: 0.55rem 0;
        }

        .insight-card.warning {
            border-left-color: var(--gold);
        }

        .insight-card.caution {
            border-left-color: #C44E52;
        }

        .insight-card .insight-title {
            color: var(--navy);
            font-size: 0.88rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }

        .insight-card .insight-text {
            color: #46546A;
            font-size: 0.86rem;
            line-height: 1.5;
        }

        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #2E9D77;
            margin-right: 7px;
        }

        .small-muted {
            color: var(--slate);
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button {
            background: var(--teal);
            color: #FFFFFF;
            border: 0;
            border-radius: 10px;
            font-weight: 700;
            padding: 0.55rem 1rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #16645F;
            color: #FFFFFF;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        @media (max-width: 900px) {
            .app-hero { padding: 1.55rem 1.4rem; }
            .block-container { padding-top: 1.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str, hero: bool = False) -> None:
    """Render a page heading, either as a dashboard hero or a compact header."""
    if hero:
        st.markdown(
            f"""
            <div class="app-hero">
                <div class="eyebrow">{html.escape(eyebrow)}</div>
                <div class="hero-title">{html.escape(title)}</div>
                <div class="hero-subtitle">{html.escape(subtitle)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="page-eyebrow">{html.escape(eyebrow)}</div>
        <div class="page-title">{html.escape(title)}</div>
        <div class="page-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    """Render a section heading with optional explanatory copy."""
    subtitle_html = (
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict[str, Any]]) -> None:
    """Render a responsive row of custom KPI cards.

    Each item supports: label, value, note, and optional tone.
    """
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        label = html.escape(str(item.get("label", "")))
        value = html.escape(str(item.get("value", "")))
        note = html.escape(str(item.get("note", "")))
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def insight_card(title: str, text: str, tone: str = "info") -> None:
    """Render a compact evidence or caveat callout."""
    tone_class = tone if tone in {"warning", "caution"} else ""
    st.markdown(
        f"""
        <div class="insight-card {tone_class}">
            <div class="insight-title">{html.escape(title)}</div>
            <div class="insight-text">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(
    figure: go.Figure,
    title: str | None = None,
    height: int = 380,
    showlegend: bool = True,
) -> go.Figure:
    """Apply the shared Plotly theme and accessible layout defaults."""
    if title:
        figure.update_layout(title=title)
    figure.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#1B263B", "size": 12},
        title_font={"size": 15, "color": NAVY},
        margin={"l": 16, "r": 22, "t": 58, "b": 28},
        showlegend=showlegend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
        hoverlabel={"bgcolor": "#FFFFFF", "font_size": 12, "bordercolor": GRID},
        transition={"duration": 0},
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return figure


def render_plot(figure: go.Figure, height: int = 380) -> None:
    """Render a Plotly chart without an obstructive toolbar."""
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def render_footer() -> None:
    """Small provenance footer used at the bottom of every page."""
    st.markdown(
        '<div class="small-muted" style="margin-top: 1.6rem;">'
        "Data source: UCI Bank Marketing dataset (<code>data/raw/bank-full.csv</code>). "
        "Model: real Phase 4 Random Forest artifact. Segmentation: real Phase 3 K-Means artifact. "
        "Observed rates are descriptive and are not causal estimates."
        "</div>",
        unsafe_allow_html=True,
    )
