from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from config import FIGURES_DIR, NUMERIC_FEATURES, ONLINE_RETAIL_FIGURES_DIR, RANDOM_STATE, RFM_FEATURES
from src.clustering import ClusteringCandidate


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_eda_figures(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    labels = {
        "age": "Age",
        "annual_income_k": "Annual Income (k$)",
        "spending_score": "Spending Score (1-100)",
    }
    for column in NUMERIC_FEATURES:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[column], kde=True, bins=20)
        plt.xlabel(labels[column])
        plt.title(f"{labels[column]} Distribution")
        _save_current_figure(FIGURES_DIR / f"{column}_histogram.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df[NUMERIC_FEATURES])
    plt.title("Numerical Feature Boxplots")
    _save_current_figure(FIGURES_DIR / "numeric_boxplots.png")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="gender")
    plt.title("Gender Distribution")
    _save_current_figure(FIGURES_DIR / "gender_distribution.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(df[NUMERIC_FEATURES].corr(), annot=True, cmap="viridis", vmin=-1, vmax=1)
    plt.title("Numerical Feature Correlation")
    _save_current_figure(FIGURES_DIR / "correlation_heatmap.png")

    scatter_specs = [
        ("annual_income_k", "spending_score", "income_vs_spending_scatter.png"),
        ("age", "spending_score", "age_vs_spending_scatter.png"),
        ("age", "annual_income_k", "age_vs_income_scatter.png"),
    ]
    for x_col, y_col, filename in scatter_specs:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue="gender")
        plt.xlabel(labels[x_col])
        plt.ylabel(labels[y_col])
        plt.title(f"{labels[x_col]} vs {labels[y_col]}")
        _save_current_figure(FIGURES_DIR / filename)


def generate_balance_figures(clean_df: pd.DataFrame, balanced_df: pd.DataFrame) -> None:
    comparison = pd.concat(
        [
            clean_df[NUMERIC_FEATURES].assign(dataset="clean"),
            balanced_df[NUMERIC_FEATURES].assign(dataset="balanced"),
        ],
        ignore_index=True,
    )
    melted = comparison.melt(id_vars="dataset", var_name="feature", value_name="value")
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=melted, x="feature", y="value", hue="dataset")
    plt.title("Mall Customers Data Balance: Before vs After")
    _save_current_figure(FIGURES_DIR / "data_balance_boxplots.png")


def generate_clustering_figures(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    candidates: list[ClusteringCandidate],
    best: ClusteringCandidate,
    feature_sets: dict[str, Any],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    kmeans_metrics = metrics_df[metrics_df["algorithm"] == "kmeans"].copy()
    if not kmeans_metrics.empty:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=kmeans_metrics, x="n_clusters", y="inertia", hue="feature_set", marker="o")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("K-Means Elbow Plot")
        _save_current_figure(FIGURES_DIR / "kmeans_elbow_plot.png")

    valid_metrics = metrics_df[metrics_df["valid_candidate"]].copy()
    if not valid_metrics.empty:
        valid_metrics["candidate_label"] = (
            valid_metrics["algorithm"]
            + " | "
            + valid_metrics["feature_set"]
            + " | k="
            + valid_metrics["n_clusters"].astype(str)
        )
        top = valid_metrics.sort_values("silhouette_score", ascending=False).head(20)
        plt.figure(figsize=(11, 6))
        sns.barplot(data=top, y="candidate_label", x="silhouette_score")
        plt.xlabel("Silhouette Score")
        plt.ylabel("Candidate")
        plt.title("Top Silhouette Scores")
        _save_current_figure(FIGURES_DIR / "silhouette_score_comparison.png")

    matrix = feature_sets[best.feature_set].matrix
    sample_matrix = matrix
    if len(matrix) > 250:
        sample_df = pd.DataFrame(matrix).sample(n=250, random_state=RANDOM_STATE)
        sample_matrix = sample_df.to_numpy()
    plt.figure(figsize=(12, 6))
    dendrogram(linkage(sample_matrix, method="ward"), no_labels=True, color_threshold=None)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    _save_current_figure(FIGURES_DIR / "hierarchical_dendrogram.png")

    clustered = df.copy()
    clustered["cluster"] = best.labels
    scatter_specs = [
        ("annual_income_k", "spending_score", "best_clusters_income_vs_spending.png"),
        ("age", "spending_score", "best_clusters_age_vs_spending.png"),
    ]
    labels = {
        "age": "Age",
        "annual_income_k": "Annual Income (k$)",
        "spending_score": "Spending Score (1-100)",
    }
    for x_col, y_col, filename in scatter_specs:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=clustered, x=x_col, y=y_col, hue="cluster", palette="tab10")
        plt.xlabel(labels[x_col])
        plt.ylabel(labels[y_col])
        plt.title(f"Best Clusters: {labels[x_col]} vs {labels[y_col]}")
        _save_current_figure(FIGURES_DIR / filename)

    plt.figure(figsize=(7, 4))
    sns.countplot(data=clustered, x="cluster")
    plt.title("Cluster Size Distribution")
    _save_current_figure(FIGURES_DIR / "cluster_size_distribution.png")


def generate_rfm_eda_figures(rfm_df: pd.DataFrame) -> None:
    ONLINE_RETAIL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = {
        "recency_days": "Recency (days)",
        "frequency": "Frequency (invoice count)",
        "monetary_value": "Monetary Value",
        "average_order_value": "Average Order Value",
    }
    for column in RFM_FEATURES:
        plt.figure(figsize=(7, 4))
        sns.histplot(rfm_df[column], kde=True, bins=30)
        plt.xlabel(labels[column])
        plt.title(f"{labels[column]} Distribution")
        _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / f"{column}_histogram.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(rfm_df[RFM_FEATURES].corr(), annot=True, cmap="viridis", vmin=-1, vmax=1)
    plt.title("RFM Feature Correlation")
    _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "rfm_correlation_heatmap.png")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=rfm_df, x="recency_days", y="monetary_value")
    plt.xlabel(labels["recency_days"])
    plt.ylabel(labels["monetary_value"])
    plt.title("Recency vs Monetary Value")
    _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "recency_vs_monetary_scatter.png")


def generate_rfm_balance_figures(rfm_df: pd.DataFrame, balanced_rfm_df: pd.DataFrame) -> None:
    comparison = pd.concat(
        [
            rfm_df[RFM_FEATURES].assign(dataset="clean_rfm"),
            balanced_rfm_df[RFM_FEATURES].assign(dataset="balanced_rfm"),
        ],
        ignore_index=True,
    )
    melted = comparison.melt(id_vars="dataset", var_name="feature", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=melted, x="feature", y="value", hue="dataset")
    plt.yscale("log")
    plt.title("Online Retail RFM Balance: Before vs After")
    _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "rfm_balance_boxplots.png")


def generate_rfm_clustering_figures(
    rfm_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best: ClusteringCandidate,
    feature_sets: dict[str, Any],
) -> None:
    ONLINE_RETAIL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    valid_metrics = metrics_df[metrics_df["valid_candidate"]].copy()
    if not valid_metrics.empty:
        valid_metrics["candidate_label"] = (
            valid_metrics["algorithm"] + " | k=" + valid_metrics["n_clusters"].astype(str)
        )
        top = valid_metrics.sort_values("silhouette_score", ascending=False).head(20)
        plt.figure(figsize=(11, 6))
        sns.barplot(data=top, y="candidate_label", x="silhouette_score")
        plt.xlabel("Silhouette Score")
        plt.ylabel("Candidate")
        plt.title("Top Online Retail RFM Silhouette Scores")
        _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "rfm_silhouette_score_comparison.png")

    kmeans_metrics = metrics_df[metrics_df["algorithm"] == "kmeans"].copy()
    if not kmeans_metrics.empty:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=kmeans_metrics, x="n_clusters", y="inertia", marker="o")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.title("Online Retail RFM K-Means Elbow Plot")
        _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "rfm_kmeans_elbow_plot.png")

    matrix = feature_sets[best.feature_set].matrix
    sample_matrix = matrix
    if len(matrix) > 250:
        sample_matrix = pd.DataFrame(matrix).sample(n=250, random_state=RANDOM_STATE).to_numpy()
    plt.figure(figsize=(12, 6))
    dendrogram(linkage(sample_matrix, method="ward"), no_labels=True, color_threshold=None)
    plt.title("Online Retail RFM Hierarchical Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "rfm_hierarchical_dendrogram.png")

    clustered = rfm_df.copy()
    clustered["cluster"] = best.labels
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=clustered, x="recency_days", y="monetary_value", hue="cluster", palette="tab10")
    plt.xlabel("Recency (days)")
    plt.ylabel("Monetary Value")
    plt.title("Online Retail RFM Clusters: Recency vs Monetary")
    _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "rfm_clusters_recency_vs_monetary.png")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=clustered, x="cluster")
    plt.title("Online Retail RFM Cluster Size Distribution")
    _save_current_figure(ONLINE_RETAIL_FIGURES_DIR / "rfm_cluster_size_distribution.png")


def generate_profile_heatmap(profile_df: pd.DataFrame, output_dir: Path = FIGURES_DIR, metric_columns: list[str] | None = None) -> None:
    if metric_columns is None:
        metric_columns = ["mean_age", "mean_annual_income_k", "mean_spending_score"]
    if profile_df.empty:
        return
    heatmap_data = profile_df.set_index("cluster_id")[metric_columns]
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, cmap="mako", fmt=".1f")
    plt.title("Cluster Profile Heatmap")
    _save_current_figure(output_dir / "cluster_profile_heatmap.png")
