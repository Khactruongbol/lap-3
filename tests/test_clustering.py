import numpy as np
import pandas as pd

from config import AGGLOMERATIVE_LARGE_DATA_K_RANGE
from src.clustering import ClusteringCandidate, run_agglomerative_grid, run_dbscan_grid, run_kmeans_grid
from src.evaluation import evaluate_candidates, select_best_candidate


def test_kmeans_grid_skips_invalid_large_k_for_small_dataset():
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0]])
    candidates = run_kmeans_grid(matrix, "numeric_only")
    assert all(candidate.params["n_clusters"] < len(matrix) for candidate in candidates)
    assert all(candidate.algorithm == "kmeans" for candidate in candidates)


def test_dbscan_candidates_report_noise_ratio():
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [8.0, 8.0], [8.1, 8.0], [30.0, 30.0]])
    candidates = run_dbscan_grid(matrix, "numeric_only")
    assert candidates
    assert all(candidate.noise_ratio is not None for candidate in candidates)


def test_agglomerative_grid_uses_compact_grid_for_large_datasets():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(1001, 2))

    candidates = run_agglomerative_grid(matrix, "rfm_log")

    assert len(candidates) == len(list(AGGLOMERATIVE_LARGE_DATA_K_RANGE))
    assert {candidate.params["linkage"] for candidate in candidates} == {"ward"}


def test_evaluation_skips_single_cluster_metrics(tmp_path):
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    candidate = ClusteringCandidate(
        algorithm="dbscan",
        feature_set="numeric_only",
        params={"eps": 10, "min_samples": 2},
        labels=np.array([0, 0, 0]),
        noise_ratio=0.0,
    )
    feature_sets = {"numeric_only": type("FeatureSetStub", (), {"matrix": matrix})()}
    metrics = evaluate_candidates([candidate], feature_sets, tmp_path / "metrics.csv")
    assert not bool(metrics.loc[0, "valid_candidate"])
    assert metrics.loc[0, "silhouette_score"] != metrics.loc[0, "silhouette_score"]


def test_selection_prefers_interpretable_candidate_when_scores_are_close(tmp_path):
    candidates = [
        ClusteringCandidate(
            algorithm="agglomerative",
            feature_set="rfm",
            params={"n_clusters": 3, "linkage": "single"},
            labels=np.array([0, 1, 2, 0, 1, 2]),
        ),
        ClusteringCandidate(
            algorithm="kmeans",
            feature_set="rfm_log",
            params={"n_clusters": 3, "random_state": 42},
            labels=np.array([0, 1, 2, 0, 1, 2]),
            inertia=10.0,
            noise_ratio=0.0,
        ),
    ]
    metrics = pd.DataFrame(
        [
            {
                "candidate_id": 0,
                "algorithm": "agglomerative",
                "feature_set": "rfm",
                "params": '{"linkage": "single", "n_clusters": 3}',
                "n_clusters": 3,
                "silhouette_score": 0.91,
                "davies_bouldin_score": 0.2,
                "calinski_harabasz_score": 100.0,
                "valid_candidate": True,
                "preferred_cluster_count": True,
                "interpretability_rank": 6,
                "feature_set_rank": 1,
            },
            {
                "candidate_id": 1,
                "algorithm": "kmeans",
                "feature_set": "rfm_log",
                "params": '{"n_clusters": 3, "random_state": 42}',
                "n_clusters": 3,
                "silhouette_score": 0.90,
                "davies_bouldin_score": 0.25,
                "calinski_harabasz_score": 95.0,
                "valid_candidate": True,
                "preferred_cluster_count": True,
                "interpretability_rank": 0,
                "feature_set_rank": 0,
            },
        ]
    )

    best = select_best_candidate(metrics, candidates, tmp_path / "best_model.json")

    assert best.algorithm == "kmeans"
    assert best.feature_set == "rfm_log"
