# Online Retail RFM Customer Segmentation Report

## Purpose

This is an extended raw-data track for customer segmentation. It uses UCI Online Retail transactions to build RFM customer features. It is not row-merged with the Mall Customers dataset because the two sources do not share customer identifiers or compatible schemas.

## Data Cleaning Summary

- Raw shape: [541909, 8]
- Clean shape: [397884, 9]
- Removed rows: 144025
- Issue counts: {'cancellation': 9288, 'missing_customer_id': 135080, 'non_positive_quantity': 10624, 'non_positive_unit_price': 2517, 'malformed_invoice_date': 0}
- Customer count: 4338

## RFM Feature Summary

Feature sets used for clustering: raw RFM features (`recency_days`, `frequency`, `monetary_value`, `average_order_value`) and log-transformed RFM features. The log-transformed set reduces the dominance of highly skewed transaction values.

## Model Comparison

128 candidates evaluated; 71 candidates were valid.

## Best Model

- Algorithm: kmeans
- Feature set: rfm
- Parameters: {"n_clusters": 2, "random_state": 42}
- Number of clusters: 2
- Silhouette score: 0.9668333381045542

## Cluster Profile

| cluster_id | size | percentage | mean_recency_days | mean_frequency | mean_monetary_value | mean_average_order_value | segment_name | business_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4336 | 99.954 | 92.504 | 4.273 | 1998.559 | 382.132 | Frequent High-Value Customers | Cluster 0 contains 100.0% of customers. Mean recency is 92.5 days, mean frequency is 4.3 invoices, and mean monetary value is 1998.6. |
| 1 | 2 | 0.046 | 163.5 | 1.5 | 122828.05 | 80709.925 | At-Risk Low-Frequency Customers | Cluster 1 contains 0.0% of customers. Mean recency is 163.5 days, mean frequency is 1.5 invoices, and mean monetary value is 122828.1. |
