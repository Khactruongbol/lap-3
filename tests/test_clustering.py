import numpy as np

from config import K_RANGE
from src.clustering import ClusteringCandidate, run_dbscan_grid, run_kmeans_grid
from src.evaluation import evaluate_candidates


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


def test_evaluation_skips_single_cluster_metrics():
    matrix = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    candidate = ClusteringCandidate(
        algorithm="dbscan",
        feature_set="numeric_only",
        params={"eps": 10, "min_samples": 2},
        labels=np.array([0, 0, 0]),
        noise_ratio=0.0,
    )
    feature_sets = {"numeric_only": type("FeatureSetStub", (), {"matrix": matrix})()}
    metrics = evaluate_candidates([candidate], feature_sets)
    assert not bool(metrics.loc[0, "valid_candidate"])
    assert metrics.loc[0, "silhouette_score"] != metrics.loc[0, "silhouette_score"]
