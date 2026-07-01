import pandas as pd

from src.online_retail import build_rfm_features, clean_online_retail_transactions


def test_online_retail_cleaning_removes_invalid_rows():
    raw = pd.DataFrame(
        {
            "InvoiceNo": ["100001", "C10002", "100003", "100004", "100005"],
            "StockCode": ["A", "B", "C", "D", "E"],
            "Description": ["ok", "cancel", "bad quantity", "bad price", "missing customer"],
            "Quantity": [2, 1, -1, 2, 1],
            "InvoiceDate": pd.to_datetime(["2024-01-01"] * 5),
            "UnitPrice": [10.0, 5.0, 2.0, 0.0, 4.0],
            "CustomerID": [1, 2, 3, 4, None],
            "Country": ["UK"] * 5,
        }
    )
    cleaned = clean_online_retail_transactions(raw, write_outputs=False)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["customer_id"] == 1
    assert cleaned.iloc[0]["line_total"] == 20.0


def test_rfm_builder_returns_one_row_per_customer_with_non_negative_values():
    cleaned = pd.DataFrame(
        {
            "invoice_no": ["1", "2", "3"],
            "customer_id": [10, 10, 20],
            "invoice_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
            "line_total": [20.0, 30.0, 15.0],
        }
    )
    rfm = build_rfm_features(cleaned, write_outputs=False)
    assert len(rfm) == 2
    assert set(["recency_days", "frequency", "monetary_value", "average_order_value"]).issubset(rfm.columns)
    assert (rfm["frequency"] >= 1).all()
    assert (rfm["monetary_value"] >= 0).all()
