import os
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_MALL_DATA_DIR = RAW_DATA_DIR / "mall_customers"
RAW_ONLINE_RETAIL_DIR = RAW_DATA_DIR / "online_retail"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
TABLES_DIR = REPORTS_DIR / "tables"
ONLINE_RETAIL_REPORTS_DIR = REPORTS_DIR / "online_retail"
ONLINE_RETAIL_FIGURES_DIR = ONLINE_RETAIL_REPORTS_DIR / "figures"
ONLINE_RETAIL_METRICS_DIR = ONLINE_RETAIL_REPORTS_DIR / "metrics"
ONLINE_RETAIL_TABLES_DIR = ONLINE_RETAIL_REPORTS_DIR / "tables"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_SOURCES_DIR = PROJECT_ROOT / "data_sources"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_SLUG = "vjchoudhary7/customer-segmentation-tutorial-in-python"
FALLBACK_DATASET_SLUG = "zubairmustafa/shopping-mall-customer-segmentation-data"
ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ONLINE_RETAIL_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/352/online%2Bretail"

RAW_DATA_FILE = RAW_MALL_DATA_DIR / "Mall_Customers.csv"
RAW_ONLINE_RETAIL_ARCHIVE = RAW_ONLINE_RETAIL_DIR / "online_retail.zip"
RAW_ONLINE_RETAIL_FILE = RAW_ONLINE_RETAIL_DIR / "Online Retail.xlsx"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "mall_customers_clean.csv"
CLUSTERED_DATA_FILE = PROCESSED_DATA_DIR / "mall_customers_clustered.csv"
ONLINE_RETAIL_CLEAN_FILE = PROCESSED_DATA_DIR / "online_retail_clean_transactions.csv"
ONLINE_RETAIL_RFM_FILE = PROCESSED_DATA_DIR / "online_retail_rfm.csv"
ONLINE_RETAIL_CLUSTERED_FILE = PROCESSED_DATA_DIR / "online_retail_rfm_clustered.csv"
DATA_SOURCE_LOG = DATA_SOURCES_DIR / "data_links.csv"
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality_report.json"
DATA_QUALITY_ISSUES = REPORTS_DIR / "data_quality_issues.csv"
ONLINE_RETAIL_QUALITY_REPORT = ONLINE_RETAIL_REPORTS_DIR / "online_retail_quality_report.json"
ONLINE_RETAIL_QUALITY_ISSUES = ONLINE_RETAIL_REPORTS_DIR / "online_retail_quality_issues.csv"
CANDIDATES_CSV = METRICS_DIR / "clustering_candidates.csv"
CANDIDATES_JSON = METRICS_DIR / "clustering_candidates.json"
METRICS_CSV = METRICS_DIR / "clustering_metrics.csv"
BEST_MODEL_JSON = METRICS_DIR / "best_model.json"
CLUSTER_PROFILE_CSV = TABLES_DIR / "cluster_profile.csv"
FINAL_REPORT = REPORTS_DIR / "final_customer_segments.md"
FINAL_NOTEBOOK = NOTEBOOKS_DIR / "99_customer_segmentation_workflow.ipynb"
MODEL_FILE = MODELS_DIR / "customer_segmentation_pipeline.joblib"
ONLINE_RETAIL_CANDIDATES_CSV = ONLINE_RETAIL_METRICS_DIR / "clustering_candidates.csv"
ONLINE_RETAIL_CANDIDATES_JSON = ONLINE_RETAIL_METRICS_DIR / "clustering_candidates.json"
ONLINE_RETAIL_METRICS_CSV = ONLINE_RETAIL_METRICS_DIR / "clustering_metrics.csv"
ONLINE_RETAIL_BEST_MODEL_JSON = ONLINE_RETAIL_METRICS_DIR / "best_model.json"
ONLINE_RETAIL_CLUSTER_PROFILE_CSV = ONLINE_RETAIL_TABLES_DIR / "cluster_profile.csv"
ONLINE_RETAIL_FINAL_REPORT = ONLINE_RETAIL_REPORTS_DIR / "online_retail_customer_segments.md"
ONLINE_RETAIL_MODEL_FILE = MODELS_DIR / "online_retail_segmentation_pipeline.joblib"

RANDOM_STATE = 42
K_RANGE = range(2, 11)
REQUIRED_COLUMNS = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

STANDARD_COLUMNS = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Age": "age",
    "Annual Income (k$)": "annual_income_k",
    "Spending Score (1-100)": "spending_score",
}

CANONICAL_COLUMNS = list(STANDARD_COLUMNS.values())
NUMERIC_FEATURES = ["age", "annual_income_k", "spending_score"]
CATEGORICAL_FEATURES = ["gender"]
RFM_FEATURES = ["recency_days", "frequency", "monetary_value", "average_order_value"]
VALID_GENDERS = {"Male", "Female"}
DBSCAN_EPS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
DBSCAN_MIN_SAMPLES_GRID = [3, 4, 5, 8, 10]
MAX_DBSCAN_NOISE_RATIO = 0.10
SILHOUETTE_CLOSE_TOLERANCE = 0.02
AGGLOMERATIVE_LARGE_DATA_THRESHOLD = 1000
AGGLOMERATIVE_LARGE_DATA_K_RANGE = range(2, 7)
AGGLOMERATIVE_LARGE_DATA_LINKAGES = ["ward"]
DATASET_CHOICES = {"mall", "online_retail", "all"}


def ensure_directories() -> None:
    for path in [
        RAW_DATA_DIR,
        RAW_MALL_DATA_DIR,
        RAW_ONLINE_RETAIL_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        TABLES_DIR,
        ONLINE_RETAIL_REPORTS_DIR,
        ONLINE_RETAIL_FIGURES_DIR,
        ONLINE_RETAIL_METRICS_DIR,
        ONLINE_RETAIL_TABLES_DIR,
        NOTEBOOKS_DIR,
        DATA_SOURCES_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
