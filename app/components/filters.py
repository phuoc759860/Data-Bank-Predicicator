"""Sidebar filters for the executive and marketing analytics pages."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


def apply_common_filters(
    df: pd.DataFrame,
    jobs: Iterable[str] | None = None,
    education: Iterable[str] | None = None,
    marital: Iterable[str] | None = None,
    contact: Iterable[str] | None = None,
    age_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Apply the shared dashboard filters without mutating the source data."""
    filtered = df
    if jobs:
        filtered = filtered[filtered["job"].isin(list(jobs))]
    if education:
        filtered = filtered[filtered["education"].isin(list(education))]
    if marital:
        filtered = filtered[filtered["marital"].isin(list(marital))]
    if contact:
        filtered = filtered[filtered["contact"].isin(list(contact))]
    if age_range:
        low, high = age_range
        filtered = filtered[filtered["age"].between(low, high)]
    return filtered.copy()


def render_sidebar_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Render the common customer filters and return the filtered frame."""
    st.sidebar.markdown("### Customer filters")
    st.sidebar.caption("Filters apply to the charts and observed statistics on this page.")

    job_options = sorted(df["job"].dropna().astype(str).unique().tolist())
    education_options = sorted(df["education"].dropna().astype(str).unique().tolist())
    marital_options = sorted(df["marital"].dropna().astype(str).unique().tolist())
    contact_options = sorted(df["contact"].dropna().astype(str).unique().tolist())
    age_min = int(df["age"].min())
    age_max = int(df["age"].max())

    jobs = st.sidebar.multiselect(
        "Job",
        options=job_options,
        default=[],
        placeholder="All jobs",
        key=f"{key_prefix}_jobs",
    )
    education = st.sidebar.multiselect(
        "Education",
        options=education_options,
        default=[],
        placeholder="All levels",
        key=f"{key_prefix}_education",
    )
    marital = st.sidebar.multiselect(
        "Marital status",
        options=marital_options,
        default=[],
        placeholder="All statuses",
        key=f"{key_prefix}_marital",
    )
    contact = st.sidebar.multiselect(
        "Contact channel",
        options=contact_options,
        default=[],
        placeholder="All channels",
        key=f"{key_prefix}_contact",
    )
    age_range = st.sidebar.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        step=1,
        key=f"{key_prefix}_age_range",
    )

    filtered = apply_common_filters(
        df,
        jobs=jobs,
        education=education,
        marital=marital,
        contact=contact,
        age_range=age_range,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Showing **{len(filtered):,}** of **{len(df):,}** customers."
    )
    return filtered
