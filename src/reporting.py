from __future__ import annotations

import json
from typing import Any

import pandas as pd

from config import (
    BEST_MODEL_JSON,
    CLUSTER_PROFILE_CSV,
    DATA_QUALITY_REPORT,
    DATA_SOURCE_LOG,
    FINAL_NOTEBOOK,
    FINAL_REPORT,
    METRICS_CSV,
    ONLINE_RETAIL_BEST_MODEL_JSON,
    ONLINE_RETAIL_CLUSTER_PROFILE_CSV,
    ONLINE_RETAIL_FINAL_REPORT,
    ONLINE_RETAIL_METRICS_CSV,
    ONLINE_RETAIL_QUALITY_REPORT,
)


def _level(value: float, low: float, high: float) -> str:
    if value <= low:
        return "Low"
    if value >= high:
        return "High"
    return "Average"


def _segment_name(row: pd.Series, income_low: float, income_high: float, spending_low: float, spending_high: float) -> str:
    income_level = _level(row["mean_annual_income_k"], income_low, income_high)
    spending_level = _level(row["mean_spending_score"], spending_low, spending_high)
    if row["mean_age"] <= 30 and spending_level == "High":
        return "Young High Spenders"
    return f"{income_level} Income - {spending_level} Spending"


def _interpret_segment(row: pd.Series) -> str:
    return (
        f"Cluster {int(row['cluster_id'])} contains {row['percentage']:.1f}% of customers. "
        f"The segment has an average age of {row['mean_age']:.1f}, average annual income of "
        f"{row['mean_annual_income_k']:.1f}k USD, and average spending score of "
        f"{row['mean_spending_score']:.1f}. The dominant gender category is {row['top_gender']}."
    )


def build_cluster_profile(clustered_df: pd.DataFrame) -> pd.DataFrame:
    non_noise = clustered_df[clustered_df["cluster"] != -1].copy()
    if non_noise.empty:
        raise ValueError("Cannot build a cluster profile because the selected model has only DBSCAN noise.")

    total = len(clustered_df)
    rows: list[dict[str, Any]] = []
    for cluster_id, group in non_noise.groupby("cluster"):
        top_gender = group["gender"].mode().iloc[0] if not group["gender"].mode().empty else "Unknown"
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(group)),
                "percentage": float(len(group) / total * 100),
                "mean_age": float(group["age"].mean()),
                "median_age": float(group["age"].median()),
                "mean_annual_income_k": float(group["annual_income_k"].mean()),
                "median_annual_income_k": float(group["annual_income_k"].median()),
                "mean_spending_score": float(group["spending_score"].mean()),
                "median_spending_score": float(group["spending_score"].median()),
                "top_gender": top_gender,
            }
        )
    profile = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)

    income_low = clustered_df["annual_income_k"].quantile(0.33)
    income_high = clustered_df["annual_income_k"].quantile(0.67)
    spending_low = clustered_df["spending_score"].quantile(0.33)
    spending_high = clustered_df["spending_score"].quantile(0.67)

    profile["segment_name"] = profile.apply(
        _segment_name,
        axis=1,
        income_low=income_low,
        income_high=income_high,
        spending_low=spending_low,
        spending_high=spending_high,
    )
    profile["business_interpretation"] = profile.apply(_interpret_segment, axis=1)

    CLUSTER_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(CLUSTER_PROFILE_CSV, index=False, encoding="utf-8")
    return profile


def build_rfm_cluster_profile(clustered_df: pd.DataFrame) -> pd.DataFrame:
    non_noise = clustered_df[clustered_df["cluster"] != -1].copy()
    if non_noise.empty:
        raise ValueError("Cannot build an RFM profile because the selected model has only DBSCAN noise.")
    total = len(clustered_df)
    rows: list[dict[str, Any]] = []
    for cluster_id, group in non_noise.groupby("cluster"):
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(group)),
                "percentage": float(len(group) / total * 100),
                "mean_recency_days": float(group["recency_days"].mean()),
                "mean_frequency": float(group["frequency"].mean()),
                "mean_monetary_value": float(group["monetary_value"].mean()),
                "mean_average_order_value": float(group["average_order_value"].mean()),
            }
        )
    profile = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    recency_low = clustered_df["recency_days"].quantile(0.33)
    frequency_high = clustered_df["frequency"].quantile(0.67)
    monetary_high = clustered_df["monetary_value"].quantile(0.67)

    def segment_name(row: pd.Series) -> str:
        if row["mean_recency_days"] <= recency_low and row["mean_monetary_value"] >= monetary_high:
            return "Recent High-Value Customers"
        if row["mean_frequency"] >= frequency_high and row["mean_monetary_value"] >= monetary_high:
            return "Frequent High-Value Customers"
        if row["mean_recency_days"] > recency_low and row["mean_frequency"] < frequency_high:
            return "At-Risk Low-Frequency Customers"
        return "Moderate RFM Customers"

    profile["segment_name"] = profile.apply(segment_name, axis=1)
    profile["business_interpretation"] = profile.apply(
        lambda row: (
            f"Cluster {int(row['cluster_id'])} contains {row['percentage']:.1f}% of customers. "
            f"Mean recency is {row['mean_recency_days']:.1f} days, mean frequency is "
            f"{row['mean_frequency']:.1f} invoices, and mean monetary value is "
            f"{row['mean_monetary_value']:.1f}."
        ),
        axis=1,
    )
    ONLINE_RETAIL_CLUSTER_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(ONLINE_RETAIL_CLUSTER_PROFILE_CSV, index=False, encoding="utf-8")
    return profile


def _read_json_if_exists(path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_source_summary(preferred_keyword: str | None = None) -> str:
    if not DATA_SOURCE_LOG.exists():
        return "No data source log was found."
    source_df = pd.read_csv(DATA_SOURCE_LOG, encoding="utf-8")
    if source_df.empty:
        return "No data source rows were recorded."
    if preferred_keyword is not None:
        matching = source_df[source_df["source_name"].astype(str).str.contains(preferred_keyword, case=False, na=False)]
        if not matching.empty:
            source_df = matching
    latest = source_df.iloc[-1]
    return f"{latest['source_name']} via {latest['access_method']} from {latest['url']}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def write_final_report(profile_df: pd.DataFrame) -> None:
    quality = _read_json_if_exists(DATA_QUALITY_REPORT)
    best = _read_json_if_exists(BEST_MODEL_JSON)
    metrics_summary = "Metrics file was not generated."
    if METRICS_CSV.exists():
        metrics_df = pd.read_csv(METRICS_CSV, encoding="utf-8")
        valid_count = int(metrics_df["valid_candidate"].sum()) if "valid_candidate" in metrics_df else 0
        metrics_summary = f"{len(metrics_df)} candidates evaluated; {valid_count} candidates were valid."

    profile_markdown = _dataframe_to_markdown(profile_df)
    report = f"""# Customer Segmentation Clustering Report

## Problem Statement

This project clusters retail-store customers using age, gender, annual income, and spending score. The task is unsupervised learning and does not use any supervised target label.

## Data Source

{_read_source_summary("kaggle")}

## Data Exploration Summary

- Raw shape: {quality.get("raw_shape", "not available")}
- Clean shape: {quality.get("clean_shape", "not available")}
- Missing values: {quality.get("missing_values_raw", "not available")}
- IQR outlier counts: {quality.get("outlier_counts_iqr", "not available")}

## Feature Scaling Summary

The pipeline compares three standardized feature sets: income-spending behavior features, numerical features only, and numerical features plus one-hot encoded gender. `customer_id` is excluded from clustering.

## Model Comparison

{metrics_summary}

## Best Model

- Algorithm: {best.get("algorithm", "not available")}
- Feature set: {best.get("feature_set", "not available")}
- Parameters: {best.get("params", "not available")}
- Number of clusters: {best.get("n_clusters", "not available")}
- Silhouette score: {best.get("silhouette_score", "not available")}
- Davies-Bouldin score: {best.get("davies_bouldin_score", "not available")}
- Calinski-Harabasz score: {best.get("calinski_harabasz_score", "not available")}

## Cluster Profile

{profile_markdown}

## Conclusion

The workflow follows the required stages of Data Exploration, Feature Scaling, Clustering, and Evaluation. It compares K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN, then selects an interpretable customer segmentation result using internal clustering metrics and customer profile analysis.
"""
    FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT.write_text(report, encoding="utf-8")


def write_online_retail_report(profile_df: pd.DataFrame) -> None:
    quality = _read_json_if_exists(ONLINE_RETAIL_QUALITY_REPORT)
    best = _read_json_if_exists(ONLINE_RETAIL_BEST_MODEL_JSON)
    metrics_summary = "Metrics file was not generated."
    if ONLINE_RETAIL_METRICS_CSV.exists():
        metrics_df = pd.read_csv(ONLINE_RETAIL_METRICS_CSV, encoding="utf-8")
        valid_count = int(metrics_df["valid_candidate"].sum()) if "valid_candidate" in metrics_df else 0
        metrics_summary = f"{len(metrics_df)} candidates evaluated; {valid_count} candidates were valid."

    report = f"""# Online Retail RFM Customer Segmentation Report

## Purpose

This is an extended raw-data track for customer segmentation. It uses UCI Online Retail transactions to build RFM customer features. It is not row-merged with the Mall Customers dataset because the two sources do not share customer identifiers or compatible schemas.

## Data Cleaning Summary

- Raw shape: {quality.get("raw_shape", "not available")}
- Clean shape: {quality.get("clean_shape", "not available")}
- Removed rows: {quality.get("removed_rows", "not available")}
- Issue counts: {quality.get("issue_counts", "not available")}
- Customer count: {quality.get("customer_count", "not available")}

## RFM Feature Summary

Feature sets used for clustering: raw RFM features (`recency_days`, `frequency`, `monetary_value`, `average_order_value`) and log-transformed RFM features. The log-transformed set reduces the dominance of highly skewed transaction values.

## Model Comparison

{metrics_summary}

## Best Model

- Algorithm: {best.get("algorithm", "not available")}
- Feature set: {best.get("feature_set", "not available")}
- Parameters: {best.get("params", "not available")}
- Number of clusters: {best.get("n_clusters", "not available")}
- Silhouette score: {best.get("silhouette_score", "not available")}

## Cluster Profile

{_dataframe_to_markdown(profile_df)}
"""
    ONLINE_RETAIL_FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    ONLINE_RETAIL_FINAL_REPORT.write_text(report, encoding="utf-8")


def write_notebook_stub() -> None:
    from src.notebook_builder import write_final_notebook

    write_final_notebook()
