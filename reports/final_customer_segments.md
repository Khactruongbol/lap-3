# Customer Segmentation Clustering Report

## Problem Statement

This project clusters retail-store customers using age, gender, annual income, and spending score. The task is unsupervised learning and does not use any supervised target label.

## Data Source

kaggle_primary via kagglehub.dataset_download from https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

## Data Exploration Summary

- Raw shape: [200, 5]
- Clean shape: [200, 5]
- Missing values: {'CustomerID': 0, 'Gender': 0, 'Age': 0, 'Annual Income (k$)': 0, 'Spending Score (1-100)': 0}
- IQR outlier counts: {'age': 0, 'annual_income_k': 2, 'spending_score': 0}

## Feature Scaling Summary

The pipeline compares two standardized feature sets: numerical features only, and numerical features plus one-hot encoded gender. `customer_id` is excluded from clustering.

## Model Comparison

190 candidates evaluated; 115 candidates were valid.

## Best Model

- Algorithm: kmeans
- Feature set: numeric_only
- Parameters: {"n_clusters": 6, "random_state": 42}
- Number of clusters: 6
- Silhouette score: 0.43106526216603014
- Davies-Bouldin score: 0.8349663784782569
- Calinski-Harabasz score: 134.47517082663606

## Cluster Profile

| cluster_id | size | percentage | mean_age | median_age | mean_annual_income_k | median_annual_income_k | mean_spending_score | median_spending_score | top_gender | segment_name | business_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 45 | 22.5 | 56.333 | 54 | 54.267 | 54 | 49.067 | 49 | Female | Average Income - Average Spending | Cluster 0 contains 22.5% of customers. The segment has an average age of 56.3, average annual income of 54.3k USD, and average spending score of 49.1. The dominant gender category is Female. |
| 1 | 39 | 19.5 | 32.692 | 32 | 86.538 | 79 | 82.128 | 83 | Female | High Income - High Spending | Cluster 1 contains 19.5% of customers. The segment has an average age of 32.7, average annual income of 86.5k USD, and average spending score of 82.1. The dominant gender category is Female. |
| 2 | 25 | 12.5 | 25.56 | 24 | 26.48 | 25 | 76.24 | 76 | Female | Young High Spenders | Cluster 2 contains 12.5% of customers. The segment has an average age of 25.6, average annual income of 26.5k USD, and average spending score of 76.2. The dominant gender category is Female. |
| 3 | 40 | 20 | 26.125 | 25 | 59.425 | 60 | 44.45 | 48 | Female | Average Income - Average Spending | Cluster 3 contains 20.0% of customers. The segment has an average age of 26.1, average annual income of 59.4k USD, and average spending score of 44.5. The dominant gender category is Female. |
| 4 | 30 | 15 | 44 | 43.5 | 90.133 | 87 | 17.933 | 16.5 | Male | High Income - Low Spending | Cluster 4 contains 15.0% of customers. The segment has an average age of 44.0, average annual income of 90.1k USD, and average spending score of 17.9. The dominant gender category is Male. |
| 5 | 21 | 10.5 | 45.524 | 46 | 26.286 | 25 | 19.381 | 15 | Female | Low Income - Low Spending | Cluster 5 contains 10.5% of customers. The segment has an average age of 45.5, average annual income of 26.3k USD, and average spending score of 19.4. The dominant gender category is Female. |

## Conclusion

The workflow follows the required stages of Data Exploration, Feature Scaling, Clustering, and Evaluation. It compares K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN, then selects an interpretable customer segmentation result using internal clustering metrics and customer profile analysis.
