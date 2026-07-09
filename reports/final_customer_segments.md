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

The pipeline compares three standardized feature sets: income-spending behavior features, numerical features only, and numerical features plus one-hot encoded gender. `customer_id` is excluded from clustering.

## Data Balance Summary

- Original clean rows: 200
- Balanced training rows: 198
- Removed outlier rows: 2
- Synthetic rows added: 0
- Strategy: remove_iqr_outliers_for_training_only

## Model Comparison

285 candidates evaluated; 168 candidates were valid.

## Best Model

- Algorithm: kmeans
- Feature set: income_spending
- Parameters: {"n_clusters": 6, "random_state": 42}
- Number of clusters: 6
- Silhouette score: 0.5401295190538672
- Davies-Bouldin score: 0.6578738097172782
- Calinski-Harabasz score: 251.59852528535643

## Cluster Profile

| cluster_id | size | percentage | mean_age | median_age | mean_annual_income_k | median_annual_income_k | mean_spending_score | median_spending_score | top_gender | segment_name | business_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 80 | 40.404 | 42.938 | 46.5 | 55.087 | 54 | 49.712 | 50 | Female | Average Income - Average Spending | Cluster 0 contains 40.4% of customers. The segment has an average age of 42.9, average annual income of 55.1k USD, and average spending score of 49.7. The dominant gender category is Female. |
| 1 | 10 | 5.051 | 32.7 | 32 | 105.3 | 102 | 82.7 | 85.5 | Female | High Income - High Spending | Cluster 1 contains 5.1% of customers. The segment has an average age of 32.7, average annual income of 105.3k USD, and average spending score of 82.7. The dominant gender category is Female. |
| 2 | 22 | 11.111 | 25.273 | 23.5 | 25.727 | 24.5 | 79.364 | 77 | Female | Young High Spenders | Cluster 2 contains 11.1% of customers. The segment has an average age of 25.3, average annual income of 25.7k USD, and average spending score of 79.4. The dominant gender category is Female. |
| 3 | 35 | 17.677 | 40.914 | 42 | 86.343 | 81 | 17.571 | 16 | Male | High Income - Low Spending | Cluster 3 contains 17.7% of customers. The segment has an average age of 40.9, average annual income of 86.3k USD, and average spending score of 17.6. The dominant gender category is Male. |
| 4 | 28 | 14.141 | 32.786 | 32 | 78.036 | 78 | 81.893 | 80.5 | Female | High Income - High Spending | Cluster 4 contains 14.1% of customers. The segment has an average age of 32.8, average annual income of 78.0k USD, and average spending score of 81.9. The dominant gender category is Female. |
| 5 | 23 | 11.616 | 45.217 | 46 | 26.304 | 25 | 20.913 | 17 | Female | Low Income - Low Spending | Cluster 5 contains 11.6% of customers. The segment has an average age of 45.2, average annual income of 26.3k USD, and average spending score of 20.9. The dominant gender category is Female. |

## Conclusion

The workflow follows the required stages of Data Exploration, Feature Scaling, Clustering, and Evaluation. It compares K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN, then selects an interpretable customer segmentation result using internal clustering metrics and customer profile analysis.
