# Customer Segmentation Clustering Rules

This document defines the mandatory project rules for Practice Exercise 3: clustering retail-store customers using age, gender, annual income, and spending score.

## 1. Problem Scope

- This is an unsupervised learning problem, not a classification or regression task.
- The primary goal is to group customers by similarity and identify interpretable customer segments.
- The workflow must follow the four required stages from the exercise: Data Exploration, Feature Scaling, Clustering, and Evaluation.
- No supervised target label may be created or used for model training.
- The final analysis must answer:
  - How many customer segments are reasonable?
  - What are the demographic and behavioral characteristics of each segment?
  - Which clustering algorithm and parameter setting provide the best balance between metric quality and interpretability?

## 2. Data Source Rules

Primary source:

- Kaggle dataset: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Required dataset slug:

```python
DATASET_SLUG = "vjchoudhary7/customer-segmentation-tutorial-in-python"
```

Fallback source:

- [Shopping Mall Customer Segmentation Data](https://www.kaggle.com/datasets/zubairmustafa/shopping-mall-customer-segmentation-data)
- A public CSV mirror may be used only if its schema and semantic meaning match the primary dataset.

Acquisition rules:

- Use `kagglehub` or the official Kaggle API as the default acquisition method.
- Do not use fabricated links, synthetic data, or undocumented datasets as the main source.
- Do not scrape a website unless its terms of service allow programmatic access.
- Every dataset source must be recorded in `data_sources/data_links.csv` with these fields: `source_name`, `url`, `access_method`, `download_date`, `license_note`, and `local_file`.
- Raw data must remain unchanged under `data/raw/`; all transformations must be written to `data/processed/`.

## 3. Canonical Dataset Schema

Expected raw columns:

```text
CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)
```

Canonical processed schema:

```text
customer_id,gender,age,annual_income_k,spending_score
```

Column definitions:

| Canonical column | Data type | Rule |
| --- | --- | --- |
| `customer_id` | integer/string | Customer identifier; never used as a clustering feature |
| `gender` | categorical | Accepted values must map cleanly to `Male` or `Female` unless the source documents another valid category |
| `age` | numeric | Must be greater than 0; outliers must be reviewed |
| `annual_income_k` | numeric | Annual income in thousand USD; must be non-negative |
| `spending_score` | numeric | Spending score on a 1-100 scale |

Schema validation rules:

- Stop the pipeline if any required column is missing.
- Preserve extra columns in the raw file, but exclude them from clustering unless explicitly justified.
- Do not change the unit meaning of `annual_income_k`.
- Do not scale, encode, or impute values directly in the raw dataset.

## 4. Data Exploration Rules

Required exploratory analysis:

- Report dataset shape, data types, missing values, and duplicated rows.
- Report descriptive statistics for `age`, `annual_income_k`, and `spending_score`.
- Report the frequency distribution of `gender`.
- Generate histograms for `age`, `annual_income_k`, and `spending_score`.
- Generate boxplots for numerical outlier inspection.
- Generate scatter plots for:
  - `annual_income_k` vs `spending_score`
  - `age` vs `spending_score`
  - `age` vs `annual_income_k`
- Generate a numerical correlation heatmap.

Data quality rules:

- Missing numeric values may be dropped when rare; otherwise, imputation must be reported and justified.
- Missing categorical values should be dropped by default; impute `Unknown` only with an explicit rationale.
- Duplicate `customer_id` values must be reported.
- If duplicate customer records are identical, keep the first record; if they conflict, write them to an issue report.
- Outliers must not be removed automatically. They must be reported and assessed for their impact on clustering.

## 5. Feature Engineering and Scaling Rules

Required numerical features:

```text
age, annual_income_k, spending_score
```

Optional categorical feature:

```text
gender
```

Feature-set rules:

- Feature set A is required: numerical features only.
- Feature set B is required: numerical features plus one-hot encoded `gender`.
- Feature sets A and B must be compared using clustering metrics and interpretability.
- `customer_id` must never be included in the clustering feature matrix.

Scaling rules:

- Use `StandardScaler` for K-Means, Agglomerative Clustering, and DBSCAN.
- Fit the scaler only on the cleaned feature matrix.
- Persist the fitted scaler only if the implementation later supports assigning clusters to new customers.
- Do not use `MinMaxScaler` as the default unless the report explains why it is preferable.

## 6. Clustering Rules

Required algorithms:

- K-Means with `k` from 2 to 10.
- Agglomerative/Hierarchical Clustering with `k` from 2 to 10.
- DBSCAN with a grid search over `eps` and `min_samples`.

Optional algorithms:

- K-Medoids may be added only if `scikit-learn-extra` is installed.
- Gaussian Mixture Models may be used as an additional baseline, but must not replace the required algorithms.

Required parameter defaults:

```python
RANDOM_STATE = 42
K_RANGE = range(2, 11)
DBSCAN_EPS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
DBSCAN_MIN_SAMPLES_GRID = [3, 4, 5, 8, 10]
```

Metric calculation rules:

- Compute `silhouette_score`, `davies_bouldin_score`, and `calinski_harabasz_score` only when the result contains at least two valid clusters.
- For DBSCAN, label `-1` represents noise and must be reported separately through `noise_ratio`.
- If DBSCAN produces zero or one valid non-noise cluster, mark the candidate as invalid for internal clustering metrics.
- K-Means candidates must record `inertia` for the elbow analysis.

## 7. Evaluation and Model Selection Rules

Required metric fields:

```text
algorithm, feature_set, params, n_clusters, silhouette_score,
davies_bouldin_score, calinski_harabasz_score, inertia, noise_ratio
```

Model-selection guidance:

- Prefer higher silhouette score.
- Prefer lower Davies-Bouldin score.
- Prefer higher Calinski-Harabasz score.
- Do not select a model based only on metrics if its customer segments are not interpretable.
- If multiple models perform similarly, prefer 3 to 6 clusters because this range is usually easier to explain in a coursework segmentation analysis.
- The final result must include a cluster profile table.

Required cluster profile fields:

```text
cluster_id, size, percentage, mean_age, mean_annual_income_k,
mean_spending_score, top_gender, segment_name, business_interpretation
```

Recommended segment names:

- High Income - High Spending
- High Income - Low Spending
- Low Income - High Spending
- Low Income - Low Spending
- Average Income - Average Spending
- Young High Spenders

## 8. Visualization Rules

Required figures:

- Numerical feature histograms.
- Numerical feature boxplots.
- Gender distribution bar chart.
- Numerical correlation heatmap.
- Scatter plot of `annual_income_k` vs `spending_score`.
- K-Means elbow plot.
- Silhouette comparison plot.
- Hierarchical clustering dendrogram.
- Best-model cluster scatter plot.
- Cluster profile chart.

Figure rules:

- Save all figures under `reports/figures/`.
- Use snake_case file names, for example `kmeans_elbow_plot.png`.
- Axis labels and titles must include units, especially `Annual Income (k$)` and `Spending Score (1-100)`.
- Do not include sample images that were not generated from the actual dataset in the final report.

## 9. File and Encoding Rules

- All `.py`, `.md`, `.csv`, `.json`, and `.ipynb` files must be saved as UTF-8.
- Use snake_case for file and directory names, except for the two required handoff files:
  - `CUSTOMER_SEGMENTATION_RULES.md`
  - `CUSTOMER_SEGMENTATION_IMPLEMENTATION.md`
- Do not leave mojibake or replacement characters in any submitted document.
- If Vietnamese text is later added to a notebook or report, validate the file content through UTF-8 reads rather than relying only on terminal rendering.

## 10. Acceptance Rules

The workflow is considered aligned with Practice Exercise 3 only if it satisfies all of the following:

- It uses a valid and traceable customer segmentation dataset.
- It includes data exploration for missing values, outliers, and feature distributions.
- It applies feature scaling before clustering.
- It runs K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN.
- It evaluates results with silhouette coefficient and at least two additional clustering metrics.
- It includes scatter plots and a dendrogram.
- It produces an interpretable customer segment profile.
- It does not convert the task into supervised learning.
- It does not add a web UI, heavy business dashboard, or prediction service in v1 unless explicitly requested.

## 11. Technical References

- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
