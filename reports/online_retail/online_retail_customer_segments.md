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

Features used for clustering: `recency_days`, `frequency`, `monetary_value`, and `average_order_value`.

## Model Comparison

95 candidates evaluated; 66 candidates were valid.

## Best Model

- Algorithm: agglomerative
- Feature set: rfm
- Parameters: {"linkage": "single", "n_clusters": 3}
- Number of clusters: 3
- Silhouette score: 0.947344641998791

## Cluster Profile

| cluster_id | size | percentage | mean_recency_days | mean_frequency | mean_monetary_value | mean_average_order_value | segment_name | business_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 0.046 | 163.5 | 1.5 | 122828.05 | 80709.925 | At-Risk Low-Frequency Customers | Cluster 0 contains 0.0% of customers. Mean recency is 163.5 days, mean frequency is 1.5 invoices, and mean monetary value is 122828.1. |
| 1 | 1 | 0.023 | 1 | 201 | 143825.06 | 715.548 | Recent High-Value Customers | Cluster 1 contains 0.0% of customers. Mean recency is 1.0 days, mean frequency is 201.0 invoices, and mean monetary value is 143825.1. |
| 2 | 4335 | 99.931 | 92.525 | 4.228 | 1965.842 | 382.055 | Frequent High-Value Customers | Cluster 2 contains 99.9% of customers. Mean recency is 92.5 days, mean frequency is 4.2 invoices, and mean monetary value is 1965.8. |
