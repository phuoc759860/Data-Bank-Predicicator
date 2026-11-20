"""K-Means artifact, assignment, profile, and projection tests."""

from __future__ import annotations

import numpy as np
import pytest

from src import app_utils as au


def test_kmeans_artifact_contract() -> None:
    kmeans = au.load_kmeans_bundle()
    assert list(kmeans.named_steps) == ["preprocessor", "kmeans"]
    assert kmeans.named_steps["kmeans"].n_clusters == 4
    features = au.get_kmeans_features()
    assert len(features) == 12
    assert not {"y", "duration"}.intersection(features)
    assert features == [
        "age",
        "balance",
        "previous",
        "default",
        "housing",
        "loan",
        "has_previous_contact",
        "previous_campaign_success",
        "job",
        "marital",
        "education",
        "poutcome",
    ]


def test_training_cluster_assignments_match_phase3() -> None:
    clusters = au.load_training_cluster_data()
    assert len(clusters) == 36_168
    assert clusters["cluster"].value_counts().sort_index().to_dict() == {
        0: 11_833,
        1: 890,
        2: 5_696,
        3: 17_749,
    }
    assert set(clusters["subscribed"].unique()) == {0, 1}


def test_cluster_profiles_are_actual_training_statistics() -> None:
    profiles = au.load_cluster_profiles()
    assert len(profiles) == 4
    assert profiles["n_customers"].sum() == 36_168
    assert profiles["share_pct"].sum() == pytest.approx(100.0)
    assert profiles["observed_subscription_rate_pct"].between(0, 100).all()
    assert profiles["overall_subscription_rate_pct"].nunique() == 1
    assert profiles["overall_subscription_rate_pct"].iloc[0] == pytest.approx(
        100.0 * 4_231 / 36_168
    )
    assert set(profiles["cluster"]) == {0, 1, 2, 3}


def test_cluster_projection_is_display_only_and_bounded() -> None:
    projection, explained = au.build_cluster_projection(max_points=500)
    assert len(projection) == 500
    assert set(projection["cluster"]).issubset({0, 1, 2, 3})
    assert {"pc1", "pc2", "age", "balance", "job", "subscribed"}.issubset(
        projection.columns
    )
    assert len(explained) == 2
    assert np.all((explained >= 0) & (explained <= 1))


def test_clustering_metrics_contain_selected_k() -> None:
    metrics = au.load_clustering_metrics()
    assert set(["k", "inertia", "silhouette_score"]).issubset(metrics.columns)
    selected = metrics.loc[metrics["k"] == 4].iloc[0]
    assert selected["cluster_sizes"] == "11833, 890, 5696, 17749"
    assert selected["silhouette_score"] == pytest.approx(0.16744697009203502)
    assert selected["inertia"] == pytest.approx(150512.02463733518)


def test_kmeans_can_predict_from_engineered_customer_features(
    raw_sample,
) -> None:
    engineered = au.build_engineered_features(raw_sample)
    kmeans = au.load_kmeans_bundle()
    labels = kmeans.predict(engineered[au.get_kmeans_features()])
    assert len(labels) == len(raw_sample)
    assert set(labels).issubset({0, 1, 2, 3})
