from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import (
    BEST_MODEL_JSON,
    CANDIDATES_CSV,
    CANDIDATES_JSON,
    CLUSTERED_DATA_FILE,
    CLUSTER_PROFILE_CSV,
    FIGURES_DIR,
    METRICS_CSV,
    MODEL_FILE,
    ONLINE_RETAIL_BEST_MODEL_JSON,
    ONLINE_RETAIL_CANDIDATES_CSV,
    ONLINE_RETAIL_CANDIDATES_JSON,
    ONLINE_RETAIL_CLUSTERED_FILE,
    ONLINE_RETAIL_CLUSTER_PROFILE_CSV,
    ONLINE_RETAIL_FIGURES_DIR,
    ONLINE_RETAIL_METRICS_CSV,
    ONLINE_RETAIL_MODEL_FILE,
)
from src.clustering import ClusteringCandidate, run_all_clustering, save_candidates
from src.data_balance import load_balanced_mall_data, load_balanced_rfm_data
from src.evaluation import attach_best_labels, evaluate_candidates, select_best_candidate
from src.preprocessing import FeatureSet, build_feature_sets, build_rfm_feature_sets
from src.reporting import (
    build_cluster_profile,
    build_rfm_cluster_profile,
    write_final_report,
    write_notebook_stub,
    write_online_retail_report,
)
from src.visualization import (
    generate_clustering_figures,
    generate_profile_heatmap,
    generate_rfm_clustering_figures,
)


def save_model_artifact(
    model_file: Path,
    dataset_name: str,
    best: ClusteringCandidate,
    feature_sets: dict[str, FeatureSet],
    clustered_df: pd.DataFrame,
    profile_path: Path,
    metrics_path: Path,
) -> Path:
    selected_feature_set = feature_sets[best.feature_set]
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "algorithm": best.algorithm,
        "params": best.params,
        "feature_set": best.feature_set,
        "feature_names": selected_feature_set.feature_names,
        "scaler": selected_feature_set.scaler,
        "labels": [int(label) for label in best.labels],
        "n_clusters": best.n_clusters,
        "profile_path": str(profile_path.as_posix()),
        "metrics_path": str(metrics_path.as_posix()),
        "clustered_rows": int(len(clustered_df)),
    }
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_file)
    return model_file


def train_mall_model() -> dict[str, Any]:
    clean_df = load_balanced_mall_data()
    feature_sets = build_feature_sets(clean_df)
    candidates = run_all_clustering(feature_sets)
    save_candidates(candidates, CANDIDATES_CSV, CANDIDATES_JSON)
    metrics_df = evaluate_candidates(candidates, feature_sets, METRICS_CSV)
    best = select_best_candidate(metrics_df, candidates, BEST_MODEL_JSON)
    clustered_df = attach_best_labels(clean_df, best, CLUSTERED_DATA_FILE)
    generate_clustering_figures(clean_df, metrics_df, candidates, best, feature_sets)
    profile_df = build_cluster_profile(clustered_df)
    generate_profile_heatmap(profile_df, FIGURES_DIR)
    write_final_report(profile_df)
    write_notebook_stub()
    model_path = save_model_artifact(
        MODEL_FILE,
        "mall",
        best,
        feature_sets,
        clustered_df,
        CLUSTER_PROFILE_CSV,
        METRICS_CSV,
    )
    return {
        "dataset": "mall",
        "best": best,
        "metrics": metrics_df,
        "profile": profile_df,
        "model_path": model_path,
    }


def train_online_retail_model() -> dict[str, Any]:
    rfm_df = load_balanced_rfm_data()
    feature_sets = build_rfm_feature_sets(rfm_df)
    candidates = run_all_clustering(feature_sets)
    save_candidates(candidates, ONLINE_RETAIL_CANDIDATES_CSV, ONLINE_RETAIL_CANDIDATES_JSON)
    metrics_df = evaluate_candidates(candidates, feature_sets, ONLINE_RETAIL_METRICS_CSV)
    best = select_best_candidate(metrics_df, candidates, ONLINE_RETAIL_BEST_MODEL_JSON)
    clustered_df = attach_best_labels(rfm_df, best, ONLINE_RETAIL_CLUSTERED_FILE)
    generate_rfm_clustering_figures(rfm_df, metrics_df, best, feature_sets)
    profile_df = build_rfm_cluster_profile(clustered_df)
    generate_profile_heatmap(
        profile_df,
        ONLINE_RETAIL_FIGURES_DIR,
        ["mean_recency_days", "mean_frequency", "mean_monetary_value"],
    )
    write_online_retail_report(profile_df)
    model_path = save_model_artifact(
        ONLINE_RETAIL_MODEL_FILE,
        "online_retail",
        best,
        feature_sets,
        clustered_df,
        ONLINE_RETAIL_CLUSTER_PROFILE_CSV,
        ONLINE_RETAIL_METRICS_CSV,
    )
    return {
        "dataset": "online_retail",
        "best": best,
        "metrics": metrics_df,
        "profile": profile_df,
        "model_path": model_path,
    }
