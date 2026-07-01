# Customer Segmentation Clustering Implementation Specification

This document is an implementation specification for an AI code generator or project team member building the Practice Exercise 3 customer segmentation program. Version 1 is limited to a command-line pipeline plus notebook/report artifacts. A web UI is intentionally out of scope.

## 1. Program Objective

Build a Machine Learning pipeline that clusters retail-store customers using:

- `age`
- `gender`
- `annual_income_k`
- `spending_score`

The final program must produce:

- A raw dataset downloaded from a traceable standard source.
- A cleaned dataset with a canonical schema.
- An EDA report covering missing values, outliers, and feature distributions.
- Clustering results from K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN.
- A metric table comparing algorithms and parameters.
- Scatter plots, histograms, an elbow plot, a silhouette comparison plot, and a dendrogram.
- A cluster profile table with customer-segment interpretations.
- A final notebook or Markdown report summarizing the complete workflow.

## 2. Recommended Project Structure

```text
customer_segmentation_clustering/
|-- config.py
|-- main.py
|-- requirements.txt
|-- data/
|   |-- raw/
|   |-- processed/
|-- data_sources/
|   |-- data_links.csv
|-- notebooks/
|   |-- 99_customer_segmentation_workflow.ipynb
|-- reports/
|   |-- figures/
|   |-- metrics/
|   |-- tables/
|   |-- data_quality_report.json
|   |-- final_customer_segments.md
|-- src/
|   |-- __init__.py
|   |-- data_acquisition.py
|   |-- data_quality.py
|   |-- preprocessing.py
|   |-- clustering.py
|   |-- evaluation.py
|   |-- visualization.py
|   |-- reporting.py
|-- tests/
|   |-- __init__.py
|   |-- test_data_quality.py
|   |-- test_preprocessing.py
|   |-- test_clustering.py
```

Do not build a web interface in v1. The notebook and files under `reports/` are the required deliverables.

## 3. Runtime Dependencies

Minimum `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
kagglehub
jupyter
ipython
ipykernel
pytest
```

Optional dependency:

```text
scikit-learn-extra
```

Use `scikit-learn-extra` only if K-Medoids is added as an optional extension.

## 4. Required Configuration

`config.py` must define the following constants:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
TABLES_DIR = REPORTS_DIR / "tables"
DATA_SOURCES_DIR = PROJECT_ROOT / "data_sources"

DATASET_SLUG = "vjchoudhary7/customer-segmentation-tutorial-in-python"
FALLBACK_DATASET_SLUG = "zubairmustafa/shopping-mall-customer-segmentation-data"

RAW_DATA_FILE = RAW_DATA_DIR / "Mall_Customers.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "mall_customers_clean.csv"
CLUSTERED_DATA_FILE = PROCESSED_DATA_DIR / "mall_customers_clustered.csv"

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

NUMERIC_FEATURES = ["age", "annual_income_k", "spending_score"]
CATEGORICAL_FEATURES = ["gender"]
DBSCAN_EPS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
DBSCAN_MIN_SAMPLES_GRID = [3, 4, 5, 8, 10]
```

## 5. Public Command-Line Interface

`main.py` must support:

```powershell
python main.py --fetch-data
python main.py --run-eda
python main.py --run-clustering
python main.py --run-all
```

Command behavior:

| Command | Expected behavior |
| --- | --- |
| `--fetch-data` | Download the dataset, save the raw CSV, and update the source log |
| `--run-eda` | Validate schema, clean data, generate EDA figures, and write the data quality report |
| `--run-clustering` | Scale features, run clustering candidates, evaluate metrics, and write clustered outputs |
| `--run-all` | Execute the full workflow from acquisition to final reporting |

If raw data is missing and the user runs `--run-clustering`, the program must fail with a clear message that recommends `python main.py --fetch-data` or `python main.py --run-all`.

## 6. Detailed Workflow

### Step 1 - Data Acquisition

File: `src/data_acquisition.py`

Responsibilities:

- Create required output directories.
- Download the primary dataset using `kagglehub.dataset_download(DATASET_SLUG)`.
- Locate a CSV file matching `REQUIRED_COLUMNS`.
- Copy the raw CSV to `data/raw/Mall_Customers.csv`.
- Write `data_sources/data_links.csv`.
- If the primary source fails, try the fallback source and record the fallback decision in the source log and final report.

Expected outputs:

```text
data/raw/Mall_Customers.csv
data_sources/data_links.csv
```

### Step 2 - Data Quality and EDA

File: `src/data_quality.py`

Responsibilities:

- Read the raw CSV using UTF-8.
- Validate all `REQUIRED_COLUMNS`.
- Rename columns to `STANDARD_COLUMNS`.
- Check missing values, duplicated rows, and duplicated `customer_id` values.
- Validate value ranges:
  - `age > 0`
  - `annual_income_k >= 0`
  - `1 <= spending_score <= 100`
  - `gender` belongs to the documented valid category set.
- Write a JSON data quality report.
- Save the cleaned CSV.

File: `src/visualization.py`

Required EDA figures:

- `age_histogram.png`
- `annual_income_histogram.png`
- `spending_score_histogram.png`
- `numeric_boxplots.png`
- `gender_distribution.png`
- `correlation_heatmap.png`
- `income_vs_spending_scatter.png`
- `age_vs_spending_scatter.png`
- `age_vs_income_scatter.png`

Expected outputs:

```text
data/processed/mall_customers_clean.csv
reports/data_quality_report.json
reports/figures/*.png
```

### Step 3 - Feature Scaling

File: `src/preprocessing.py`

Responsibilities:

- Build feature set A with numerical features only:

```text
age, annual_income_k, spending_score
```

- Build feature set B with numerical features plus one-hot encoded gender.
- Apply `StandardScaler` to each feature matrix.
- Return:
  - `X_numeric_scaled`
  - `X_with_gender_scaled`
  - feature names
  - fitted preprocessors if future cluster assignment is needed.

Rules:

- Never include `customer_id` in the feature matrix.
- Do not run scaling on the unvalidated raw schema.
- Do not encode `gender` as ordinal values such as `Male=0` and `Female=1`; use one-hot encoding.

### Step 4 - Clustering

File: `src/clustering.py`

Required functions:

- `run_kmeans_grid(X, feature_set_name)`
  - Iterate over `k` in `K_RANGE`.
  - Use `KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")`.
  - Store labels, inertia, and parameters.

- `run_agglomerative_grid(X, feature_set_name)`
  - Iterate over `k` in `K_RANGE`.
  - Try `linkage` in `["ward", "complete", "average", "single"]`.
  - Use Euclidean distance for `ward`.
  - Store labels and parameters.

- `run_dbscan_grid(X, feature_set_name)`
  - Iterate over `eps` in `DBSCAN_EPS_GRID`.
  - Iterate over `min_samples` in `DBSCAN_MIN_SAMPLES_GRID`.
  - Store labels, valid cluster count, noise ratio, and parameters.

Intermediate outputs:

```text
reports/metrics/clustering_candidates.csv
reports/metrics/clustering_candidates.json
```

### Step 5 - Evaluation

File: `src/evaluation.py`

Responsibilities:

- Compute metrics for each valid candidate:
  - `silhouette_score`
  - `davies_bouldin_score`
  - `calinski_harabasz_score`
  - `inertia` for K-Means
  - `noise_ratio` for DBSCAN
- Skip internal clustering metrics when the result has fewer than two valid clusters.
- Rank candidates by:
  - valid candidates first
  - descending silhouette score
  - ascending Davies-Bouldin score
  - descending Calinski-Harabasz score
  - preferred cluster count between 3 and 6 when metrics are close.
- Select the best model and attach its labels to the cleaned dataframe.

Expected outputs:

```text
reports/metrics/clustering_metrics.csv
reports/metrics/best_model.json
data/processed/mall_customers_clustered.csv
```

### Step 6 - Post-Clustering Visualization

File: `src/visualization.py`

Required clustering figures:

- `kmeans_elbow_plot.png`
- `silhouette_score_comparison.png`
- `hierarchical_dendrogram.png`
- `best_clusters_income_vs_spending.png`
- `best_clusters_age_vs_spending.png`
- `cluster_size_distribution.png`
- `cluster_profile_heatmap.png`

Dendrogram rules:

- Use `scipy.cluster.hierarchy.linkage` and `scipy.cluster.hierarchy.dendrogram`.
- If the dataset is too large, draw the dendrogram from a reproducible sample using `random_state=42`.
- For the Mall Customers dataset, rendering the full dendrogram is usually acceptable.

### Step 7 - Cluster Profiling and Final Report

File: `src/reporting.py`

Responsibilities:

- Build a cluster profile table with:

```text
cluster_id, size, percentage, mean_age, median_age,
mean_annual_income_k, median_annual_income_k,
mean_spending_score, median_spending_score,
top_gender, segment_name, business_interpretation
```

- Assign segment names from income and spending patterns:
  - high/medium/low income based on quantiles or comparison against the dataset median.
  - high/medium/low spending based on quantiles or comparison against the dataset median.
  - if the mean age is low and spending is high, `Young High Spenders` is acceptable.

- Generate `reports/final_customer_segments.md` with:
  - problem statement
  - data source
  - EDA summary
  - feature scaling summary
  - model comparison
  - best model
  - cluster profile
  - conclusion aligned with Practice Exercise 3.

Expected outputs:

```text
reports/tables/cluster_profile.csv
reports/final_customer_segments.md
```

## 7. Final Notebook

Final notebook path:

```text
notebooks/99_customer_segmentation_workflow.ipynb
```

Required sections:

1. Problem Definition
2. Data Source and Schema
3. Data Exploration
4. Missing Values, Outliers, and Distributions
5. Feature Scaling
6. K-Means Clustering
7. Agglomerative/Hierarchical Clustering
8. DBSCAN Clustering
9. Evaluation Metrics
10. Customer Segment Interpretation
11. Final Checklist

The notebook should reuse artifacts generated by the pipeline when available. If required artifacts are missing, it should tell the user to run:

```powershell
python main.py --run-all
```

## 8. AI Code Generation Prompt

Use the following prompt to generate the implementation:

```text
You are a senior Machine Learning engineer. Create a Python project for Practice Exercise 3: Customer Segmentation Clustering.

Objective:
- Cluster retail customers using age, gender, annual income, and spending score.
- This is an unsupervised learning task. Do not create or use a supervised target label.
- The workflow must include Data Exploration, Feature Scaling, Clustering, and Evaluation.

Data source:
- Use Kaggle dataset slug "vjchoudhary7/customer-segmentation-tutorial-in-python" as the primary source.
- The raw dataset must contain CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).
- If the primary source cannot be downloaded, use only a schema-compatible fallback source and record it in data_sources/data_links.csv.

Create these files:
- config.py
- main.py
- requirements.txt
- src/data_acquisition.py
- src/data_quality.py
- src/preprocessing.py
- src/clustering.py
- src/evaluation.py
- src/visualization.py
- src/reporting.py
- tests/test_data_quality.py
- tests/test_preprocessing.py
- tests/test_clustering.py

Required CLI:
- python main.py --fetch-data
- python main.py --run-eda
- python main.py --run-clustering
- python main.py --run-all

Required algorithms:
- KMeans for k=2..10
- AgglomerativeClustering for k=2..10 and linkage ward/complete/average/single
- DBSCAN using eps and min_samples grid search

Required evaluation:
- silhouette_score
- davies_bouldin_score
- calinski_harabasz_score
- inertia for KMeans
- noise_ratio for DBSCAN

Required outputs:
- data/raw/Mall_Customers.csv
- data/processed/mall_customers_clean.csv
- data/processed/mall_customers_clustered.csv
- reports/data_quality_report.json
- reports/metrics/clustering_metrics.csv
- reports/metrics/best_model.json
- reports/tables/cluster_profile.csv
- reports/final_customer_segments.md
- reports/figures/*.png

Code rules:
- Save all text files as UTF-8.
- Use snake_case for Python file names, functions, and variables.
- Never include customer_id in clustering.
- Use StandardScaler.
- Compare a numeric-only feature set against a numeric-plus-gender feature set.
- Do not compute clustering metrics when the result has fewer than two valid clusters.
- Add tests for schema validation, preprocessing, and clustering edge cases.
```

## 9. Execution and Validation Commands

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python main.py --run-all
```

Run individual stages:

```powershell
python main.py --fetch-data
python main.py --run-eda
python main.py --run-clustering
```

Run tests:

```powershell
python -m pytest -q
```

Check Python syntax:

```powershell
python -m compileall -q config.py main.py src tests
```

Validate notebook JSON if a notebook is generated:

```powershell
python -m json.tool notebooks/99_customer_segmentation_workflow.ipynb > $null
```

Validate Markdown files as UTF-8:

```powershell
Get-Content CUSTOMER_SEGMENTATION_RULES.md -Encoding UTF8 | Select-Object -First 5
Get-Content CUSTOMER_SEGMENTATION_IMPLEMENTATION.md -Encoding UTF8 | Select-Object -First 5
```

## 10. Acceptance Checklist

Before submission, verify that:

- [ ] The dataset is downloaded from Kaggle or a valid schema-compatible fallback source.
- [ ] `data_sources/data_links.csv` records the dataset source.
- [ ] Raw data is preserved in `data/raw/`.
- [ ] Cleaned data uses `customer_id`, `gender`, `age`, `annual_income_k`, and `spending_score`.
- [ ] EDA covers missing values, outliers, and distributions.
- [ ] Scatter plots are available for pattern inspection.
- [ ] Feature scaling uses `StandardScaler`.
- [ ] The implementation compares numeric-only and numeric-plus-gender feature sets.
- [ ] K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN are implemented.
- [ ] Silhouette coefficient is reported.
- [ ] Davies-Bouldin and Calinski-Harabasz scores are reported.
- [ ] A dendrogram is generated.
- [ ] A cluster profile table with segment names is generated.
- [ ] The final report explains customer segments in clear technical language.
- [ ] No supervised target label or prediction target is used.
- [ ] No web UI is added in v1 unless explicitly requested.

## 11. Alignment Review

This workflow and rule set match Practice Exercise 3 because:

- The dataset directly contains the required customer attributes: age, gender, annual income, and spending score.
- Data exploration covers missing values, outliers, and feature distributions.
- Feature scaling is a separate required stage before clustering.
- The clustering stage compares multiple algorithms and parameter settings, matching the exercise requirement to choose and experiment with clustering algorithms.
- Evaluation includes silhouette coefficient and additional clustering metrics.
- Visualization includes scatter plots and a dendrogram to support customer-segment interpretation.
- The final output focuses on distinct customer segments and does not drift into classification, regression, a web UI, or a dashboard outside the exercise scope.

## 12. References

- [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Shopping Mall Customer Segmentation Data](https://www.kaggle.com/datasets/zubairmustafa/shopping-mall-customer-segmentation-data)
- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
