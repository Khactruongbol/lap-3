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

## Data Balance Summary

- Original RFM rows: 4338
- Balanced training rows: 4250
- Removed upper-tail rows: 88
- Synthetic rows added: 0
- Strategy: remove_upper_tail_rfm_outliers_for_training_only

## Model Comparison

128 candidates evaluated; 89 candidates were valid.

## Best Model

- Algorithm: dbscan
- Feature set: rfm
- Parameters: {"eps": 0.8, "min_samples": 4}
- Number of clusters: 3
- Silhouette score: 0.6974975963079184

## Cluster Profile

| cluster_id | size | percentage | mean_recency_days | mean_frequency | mean_monetary_value | mean_average_order_value | segment_name | business_interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4202 | 98.871 | 93.519 | 3.613 | 1241.435 | 336.465 | At-Risk Low-Frequency Customers | Cluster 0 contains 98.9% of customers. Mean recency is 93.5 days, mean frequency is 3.6 invoices, and mean monetary value is 1241.4. |
| 1 | 6 | 0.141 | 12.167 | 19.5 | 13796.78 | 711.323 | Recent High-Value Customers | Cluster 1 contains 0.1% of customers. Mean recency is 12.2 days, mean frequency is 19.5 invoices, and mean monetary value is 13796.8. |
| 2 | 5 | 0.118 | 3.6 | 28.6 | 17260.64 | 604.118 | Recent High-Value Customers | Cluster 2 contains 0.1% of customers. Mean recency is 3.6 days, mean frequency is 28.6 invoices, and mean monetary value is 17260.6. |
