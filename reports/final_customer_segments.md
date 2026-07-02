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

## Model Comparison

285 candidates evaluated; 168 candidates were valid.

## Best Model

- Algorithm: kmeans
- Feature set: income_spending
- Parameters: {"n_clusters": 5, "random_state": 42}
- Number of clusters: 5
- Silhouette score: 0.5546571631111091
- Davies-Bouldin score: 0.5722356162263352
- Calinski-Harabasz score: 248.64932001536357

## Cluster Profile

| cluster_id | size | percentage | mean_age | median_age | mean_annual_income_k | median_annual_income_k | mean_spending_score | median_spending_score | top_gender | segment_name | business_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 81 | 40.5 | 42.716 | 46 | 55.296 | 54 | 49.519 | 50 | Female | Average Income - Average Spending | Cluster 0 contains 40.5% of customers. The segment has an average age of 42.7, average annual income of 55.3k USD, and average spending score of 49.5. The dominant gender category is Female. |
| 1 | 39 | 19.5 | 32.692 | 32 | 86.538 | 79 | 82.128 | 83 | Female | High Income - High Spending | Cluster 1 contains 19.5% of customers. The segment has an average age of 32.7, average annual income of 86.5k USD, and average spending score of 82.1. The dominant gender category is Female. |
| 2 | 22 | 11 | 25.273 | 23.5 | 25.727 | 24.5 | 79.364 | 77 | Female | Young High Spenders | Cluster 2 contains 11.0% of customers. The segment has an average age of 25.3, average annual income of 25.7k USD, and average spending score of 79.4. The dominant gender category is Female. |
| 3 | 35 | 17.5 | 41.114 | 42 | 88.2 | 85 | 17.114 | 16 | Male | High Income - Low Spending | Cluster 3 contains 17.5% of customers. The segment has an average age of 41.1, average annual income of 88.2k USD, and average spending score of 17.1. The dominant gender category is Male. |
| 4 | 23 | 11.5 | 45.217 | 46 | 26.304 | 25 | 20.913 | 17 | Female | Low Income - Low Spending | Cluster 4 contains 11.5% of customers. The segment has an average age of 45.2, average annual income of 26.3k USD, and average spending score of 20.9. The dominant gender category is Female. |

## Conclusion

The workflow follows the required stages of Data Exploration, Feature Scaling, Clustering, and Evaluation. It compares K-Means, Agglomerative/Hierarchical Clustering, and DBSCAN, then selects an interpretable customer segmentation result using internal clustering metrics and customer profile analysis.
