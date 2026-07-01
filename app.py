from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")


def artifact_paths(dataset: str) -> dict[str, Path]:
    if dataset == "Online Retail RFM":
        base = PROJECT_ROOT / "reports" / "online_retail"
        return {
            "best": base / "metrics" / "best_model.json",
            "metrics": base / "metrics" / "clustering_metrics.csv",
            "profile": base / "tables" / "cluster_profile.csv",
            "clustered": PROJECT_ROOT / "data" / "processed" / "online_retail_rfm_clustered.csv",
            "report": base / "online_retail_customer_segments.md",
        }
    return {
        "best": PROJECT_ROOT / "reports" / "metrics" / "best_model.json",
        "metrics": PROJECT_ROOT / "reports" / "metrics" / "clustering_metrics.csv",
        "profile": PROJECT_ROOT / "reports" / "tables" / "cluster_profile.csv",
        "clustered": PROJECT_ROOT / "data" / "processed" / "mall_customers_clustered.csv",
        "report": PROJECT_ROOT / "reports" / "final_customer_segments.md",
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Customer Segmentation", layout="wide")
    st.title("Customer Segmentation Clustering")
    st.caption("Unsupervised customer segmentation demo. The UI reads generated artifacts and does not retrain models.")

    dataset = st.sidebar.selectbox("Dataset", ["Mall Customers", "Online Retail RFM"])
    paths = artifact_paths(dataset)
    best = load_json(paths["best"])
    metrics = load_csv(paths["metrics"])
    profile = load_csv(paths["profile"])
    clustered = load_csv(paths["clustered"])

    if not best:
        st.warning("Artifacts are missing. Run `python main.py --run-all` first.")
        return

    st.subheader("Best Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algorithm", best.get("algorithm", "N/A"))
    col2.metric("Feature Set", best.get("feature_set", "N/A"))
    col3.metric("Clusters", best.get("n_clusters", "N/A"))
    col4.metric("Silhouette", round(float(best.get("silhouette_score", 0)), 4))

    st.subheader("Cluster Profile")
    st.dataframe(profile, use_container_width=True)

    st.subheader("Candidate Metrics")
    if not metrics.empty:
        st.dataframe(metrics.sort_values("silhouette_score", ascending=False), use_container_width=True)

    st.subheader("Clustered Data Preview")
    st.dataframe(clustered.head(100), use_container_width=True)

    st.subheader("Report")
    if paths["report"].exists():
        st.markdown(paths["report"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
