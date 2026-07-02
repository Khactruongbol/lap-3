from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

from config import (
    AGGLOMERATIVE_LARGE_DATA_K_RANGE,
    AGGLOMERATIVE_LARGE_DATA_LINKAGES,
    AGGLOMERATIVE_LARGE_DATA_THRESHOLD,
    CANDIDATES_CSV,
    CANDIDATES_JSON,
    DBSCAN_EPS_GRID,
    DBSCAN_MIN_SAMPLES_GRID,
    K_RANGE,
    RANDOM_STATE,
)


@dataclass
class ClusteringCandidate:
    algorithm: str
    feature_set: str
    params: dict[str, Any]
    labels: np.ndarray
    inertia: float | None = None
    noise_ratio: float | None = None

    @property
    def n_clusters(self) -> int:
        unique = set(int(label) for label in self.labels)
        unique.discard(-1)
        return len(unique)

    def to_record(self, include_labels: bool = False) -> dict[str, Any]:
        record = {
            "algorithm": self.algorithm,
            "feature_set": self.feature_set,
            "params": json.dumps(self.params, sort_keys=True),
            "n_clusters": self.n_clusters,
            "inertia": self.inertia,
            "noise_ratio": self.noise_ratio,
        }
        if include_labels:
            record["labels"] = [int(label) for label in self.labels]
        return record


def _fit_kmeans(matrix: object, k: int) -> KMeans:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    try:
        model.fit(matrix)
    except TypeError:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(matrix)
    return model


def run_kmeans_grid(matrix: object, feature_set_name: str) -> list[ClusteringCandidate]:
    candidates: list[ClusteringCandidate] = []
    n_samples = len(matrix)
    for k in K_RANGE:
        if k >= n_samples:
            continue
        model = _fit_kmeans(matrix, k)
        candidates.append(
            ClusteringCandidate(
                algorithm="kmeans",
                feature_set=feature_set_name,
                params={"n_clusters": k, "random_state": RANDOM_STATE},
                labels=model.labels_,
                inertia=float(model.inertia_),
                noise_ratio=0.0,
            )
        )
    return candidates


def run_agglomerative_grid(matrix: object, feature_set_name: str) -> list[ClusteringCandidate]:
    candidates: list[ClusteringCandidate] = []
    n_samples = len(matrix)
    if n_samples > AGGLOMERATIVE_LARGE_DATA_THRESHOLD:
        k_values = AGGLOMERATIVE_LARGE_DATA_K_RANGE
        linkages = AGGLOMERATIVE_LARGE_DATA_LINKAGES
    else:
        k_values = K_RANGE
        linkages = ["ward", "complete", "average", "single"]
    for k in k_values:
        if k >= n_samples:
            continue
        for linkage in linkages:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(matrix)
            candidates.append(
                ClusteringCandidate(
                    algorithm="agglomerative",
                    feature_set=feature_set_name,
                    params={"n_clusters": k, "linkage": linkage},
                    labels=labels,
                    inertia=None,
                    noise_ratio=0.0,
                )
            )
    return candidates


def run_dbscan_grid(matrix: object, feature_set_name: str) -> list[ClusteringCandidate]:
    candidates: list[ClusteringCandidate] = []
    n_samples = len(matrix)
    for eps in DBSCAN_EPS_GRID:
        for min_samples in DBSCAN_MIN_SAMPLES_GRID:
            if min_samples >= n_samples:
                continue
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(matrix)
            noise_ratio = float(np.mean(labels == -1))
            candidates.append(
                ClusteringCandidate(
                    algorithm="dbscan",
                    feature_set=feature_set_name,
                    params={"eps": eps, "min_samples": min_samples},
                    labels=labels,
                    inertia=None,
                    noise_ratio=noise_ratio,
                )
            )
    return candidates


def run_all_clustering(feature_sets: dict[str, Any]) -> list[ClusteringCandidate]:
    candidates: list[ClusteringCandidate] = []
    for feature_set_name, feature_set in feature_sets.items():
        matrix = feature_set.matrix
        candidates.extend(run_kmeans_grid(matrix, feature_set_name))
        candidates.extend(run_agglomerative_grid(matrix, feature_set_name))
        candidates.extend(run_dbscan_grid(matrix, feature_set_name))
    return candidates


def save_candidates(
    candidates: list[ClusteringCandidate],
    candidates_csv=CANDIDATES_CSV,
    candidates_json=CANDIDATES_JSON,
) -> None:
    candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([candidate.to_record(include_labels=False) for candidate in candidates]).to_csv(
        candidates_csv, index=False, encoding="utf-8"
    )
    candidates_json.write_text(
        json.dumps([candidate.to_record(include_labels=True) for candidate in candidates], indent=2),
        encoding="utf-8",
    )
