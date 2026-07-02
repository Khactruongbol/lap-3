from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from config import (
    BEST_MODEL_JSON,
    CLUSTERED_DATA_FILE,
    MAX_DBSCAN_NOISE_RATIO,
    METRICS_CSV,
    SILHOUETTE_CLOSE_TOLERANCE,
)
from src.clustering import ClusteringCandidate


def _interpretability_rank(candidate: ClusteringCandidate) -> int:
    if candidate.algorithm == "kmeans":
        return 0
    if candidate.algorithm == "agglomerative":
        linkage = candidate.params.get("linkage")
        return {"ward": 1, "complete": 2, "average": 3, "single": 6}.get(str(linkage), 5)
    if candidate.algorithm == "dbscan":
        return 4
    return 5


def _feature_set_rank(feature_set_name: str) -> int:
    return {
        "income_spending": 0,
        "rfm_log": 0,
        "numeric_only": 1,
        "rfm": 1,
        "numeric_plus_gender": 2,
    }.get(feature_set_name, 3)


def _valid_metric_data(matrix: object, labels: np.ndarray, algorithm: str) -> tuple[np.ndarray, np.ndarray]:
    matrix_array = np.asarray(matrix)
    labels_array = np.asarray(labels)
    if algorithm == "dbscan":
        mask = labels_array != -1
        return matrix_array[mask], labels_array[mask]
    return matrix_array, labels_array


def _can_compute_internal_metrics(matrix: np.ndarray, labels: np.ndarray) -> bool:
    unique_labels = set(int(label) for label in labels)
    return len(unique_labels) >= 2 and len(unique_labels) < len(labels)


def evaluate_candidates(
    candidates: list[ClusteringCandidate],
    feature_sets: dict[str, Any],
    metrics_csv=METRICS_CSV,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        matrix, labels = _valid_metric_data(
            feature_sets[candidate.feature_set].matrix, candidate.labels, candidate.algorithm
        )
        can_score = _can_compute_internal_metrics(matrix, labels)
        silhouette = davies_bouldin = calinski_harabasz = np.nan
        acceptable_noise = (
            candidate.algorithm != "dbscan"
            or candidate.noise_ratio is None
            or candidate.noise_ratio <= MAX_DBSCAN_NOISE_RATIO
        )
        valid_candidate = candidate.n_clusters >= 2 and can_score and acceptable_noise
        if valid_candidate:
            silhouette = float(silhouette_score(matrix, labels))
            davies_bouldin = float(davies_bouldin_score(matrix, labels))
            calinski_harabasz = float(calinski_harabasz_score(matrix, labels))

        preferred_cluster_count = 3 <= candidate.n_clusters <= 6
        rows.append(
            {
                "candidate_id": index,
                "algorithm": candidate.algorithm,
                "feature_set": candidate.feature_set,
                "params": json.dumps(candidate.params, sort_keys=True),
                "n_clusters": candidate.n_clusters,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
                "calinski_harabasz_score": calinski_harabasz,
                "inertia": candidate.inertia,
                "noise_ratio": candidate.noise_ratio,
                "acceptable_noise": bool(acceptable_noise),
                "valid_candidate": bool(valid_candidate),
                "preferred_cluster_count": bool(preferred_cluster_count),
                "interpretability_rank": _interpretability_rank(candidate),
                "feature_set_rank": _feature_set_rank(candidate.feature_set),
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")
    return metrics_df


def select_best_candidate(
    metrics_df: pd.DataFrame,
    candidates: list[ClusteringCandidate],
    best_model_json=BEST_MODEL_JSON,
) -> ClusteringCandidate:
    valid = metrics_df[metrics_df["valid_candidate"]].copy()
    if valid.empty:
        raise ValueError("No valid clustering candidate produced at least two valid clusters.")

    valid["preferred_rank"] = valid["preferred_cluster_count"].astype(int)
    best_silhouette = valid["silhouette_score"].max()
    close_candidates = valid[
        valid["silhouette_score"] >= best_silhouette - SILHOUETTE_CLOSE_TOLERANCE
    ].copy()
    if close_candidates["preferred_cluster_count"].any():
        valid = close_candidates[close_candidates["preferred_cluster_count"]].copy()
    else:
        valid = close_candidates
    valid = valid.sort_values(
        by=[
            "interpretability_rank",
            "feature_set_rank",
            "silhouette_score",
            "preferred_rank",
            "davies_bouldin_score",
            "calinski_harabasz_score",
        ],
        ascending=[True, True, False, False, True, False],
    )
    best_row = valid.iloc[0]
    best = candidates[int(best_row["candidate_id"])]
    payload = best_row.where(pd.notna(best_row), None).to_dict()
    payload["labels"] = [int(label) for label in best.labels]
    best_model_json.parent.mkdir(parents=True, exist_ok=True)
    best_model_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return best


def attach_best_labels(clean_df: pd.DataFrame, best: ClusteringCandidate, output_file=CLUSTERED_DATA_FILE) -> pd.DataFrame:
    clustered = clean_df.copy()
    clustered["cluster"] = [int(label) for label in best.labels]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(output_file, index=False, encoding="utf-8")
    return clustered
