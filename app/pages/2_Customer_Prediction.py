"""Individual customer prediction page."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.theme import (
    apply_theme,
    configure_page,
    insight_card,
    page_header,
    render_footer,
    section_header,
)
from app.utils.formatting import probability_text
from src.app_utils import (
    load_best_model_bundle,
    load_categorical_options,
    load_numeric_ranges,
    predict_customer,
)

configure_page("Customer Prediction")
apply_theme()

try:
    artifact = load_best_model_bundle()
    categories = load_categorical_options()
    ranges = load_numeric_ranges()
except (RuntimeError, FileNotFoundError, ValueError) as exc:
    st.error("The prediction model could not be loaded.")
    st.exception(exc)
    st.stop()

page_header(
    "Decision support",
    "Score an individual customer",
    "Enter pre-call customer and campaign information. The app applies the "
    "saved Phase 2 preprocessing pipeline and the selected Phase 4 model. "
    "Call duration is intentionally excluded because it is only known after "
    "the call.",
)

with st.form("customer_prediction_form", clear_on_submit=False):
    section_header("Client profile")
    left, right = st.columns(2)
    with left:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=40,
            step=1,
            help="Client age in years.",
        )
        job = st.selectbox("Job", categories["job"], index=categories["job"].index("management"))
        marital = st.selectbox("Marital status", categories["marital"], index=categories["marital"].index("married"))
        education = st.selectbox(
            "Education",
            categories["education"],
            index=categories["education"].index("secondary"),
        )
        default = st.selectbox("Credit in default?", ["no", "yes"])
        balance = st.number_input(
            "Average yearly balance (EUR)",
            min_value=int(ranges["balance"][0]) - 1000,
            max_value=int(ranges["balance"][1]) + 1000,
            value=1500,
            step=100,
            help="Average yearly account balance in euros.",
        )
    with right:
        housing = st.selectbox("Housing loan?", ["no", "yes"])
        loan = st.selectbox("Personal loan?", ["no", "yes"])
        contact = st.selectbox("Contact channel", categories["contact"])
        month = st.selectbox("Last contact month", categories["month"])
        day = st.number_input(
            "Last contact day of month",
            min_value=1,
            max_value=31,
            value=15,
            step=1,
        )
        campaign = st.number_input(
            "Contacts in current campaign",
            min_value=1,
            max_value=63,
            value=1,
            step=1,
            help="Number of contacts performed during this campaign, including the last contact.",
        )

    section_header("Campaign history")
    left, right = st.columns(2)
    with left:
        previous = st.number_input(
            "Contacts before this campaign",
            min_value=0,
            max_value=275,
            value=0,
            step=1,
            help="Number of contacts performed before this campaign for this client.",
        )
        pdays = st.number_input(
            "Days since previous contact (-1 = never)",
            min_value=-1,
            max_value=871,
            value=-1,
            step=1,
            help="Use -1 when the client was never contacted in a previous campaign.",
        )
    with right:
        poutcome = st.selectbox(
            "Previous campaign outcome",
            categories["poutcome"],
            index=categories["poutcome"].index("unknown"),
        )
        st.caption(
            "The model receives engineered versions of previous campaign history. "
            "The raw pdays sentinel -1 is converted to a 'no previous contact' flag."
        )

    submitted = st.form_submit_button("Estimate subscription probability")

if submitted:
    record = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": int(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome,
    }
    try:
        st.session_state["customer_prediction"] = predict_customer(record)
    except ValueError as exc:
        st.session_state.pop("customer_prediction", None)
        st.error("The customer record could not be scored.")
        st.error(str(exc))

result = st.session_state.get("customer_prediction")
if result:
    section_header("Model estimate")
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Predicted probability", probability_text(result["probability"]))
        st.metric("Decision threshold", f"{result['threshold']:.2f}")
    with right:
        if result["prediction"] == "Yes":
            st.success("### Likely to subscribe")
            st.write(
                "This customer's estimated probability is at or above the selected "
                "decision threshold. The customer is a candidate for prioritised "
                "outreach, subject to campaign capacity and offer economics."
            )
        else:
            st.warning("### Below the decision threshold")
            st.write(
                "This customer's estimated probability is below the selected "
                "decision threshold. They may still subscribe; the estimate is a "
                "ranking signal, not a certainty."
            )

    st.progress(
        min(1.0, max(0.0, float(result["probability"]))),
        text=f"Estimated probability: {probability_text(result['probability'])}",
    )
    insight_card(
        "Important interpretation",
        "The probability is a model estimate based on historical patterns, not a "
        "guarantee. A high-probability customer can decline, and a lower-probability "
        "customer can accept. Use the result to prioritise outreach, not to deny "
        "service or make automated decisions without human review.",
        tone="warning",
    )
    st.caption(
        f"Model used: {artifact['model_name']} | Selected Phase 4 threshold: "
        f"{float(artifact['threshold']):.2f} | Duration excluded."
    )

render_footer()
