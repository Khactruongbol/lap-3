import pandas as pd

import src.online_retail as online_retail
from src.online_retail import build_rfm_features, clean_online_retail_transactions, load_or_process_online_retail


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


def test_online_retail_cache_uses_processed_rfm_when_fresh(tmp_path, monkeypatch):
    raw_file = tmp_path / "Online Retail.xlsx"
    raw_file.write_bytes(b"raw-placeholder")
    clean_file = tmp_path / "online_retail_clean_transactions.csv"
    clean_file.write_text("invoice_no\n1\n", encoding="utf-8")
    rfm_file = tmp_path / "online_retail_rfm.csv"
    pd.DataFrame(
        {
            "customer_id": [1],
            "recency_days": [1],
            "frequency": [1],
            "monetary_value": [10.0],
            "average_order_value": [10.0],
            "last_purchase_date": ["2024-01-01"],
        }
    ).to_csv(rfm_file, index=False, encoding="utf-8")

    monkeypatch.setattr(online_retail, "ONLINE_RETAIL_CLEAN_FILE", clean_file)
    monkeypatch.setattr(online_retail, "ONLINE_RETAIL_RFM_FILE", rfm_file)

    cached = load_or_process_online_retail(raw_file)

    assert len(cached) == 1
    assert cached.loc[0, "customer_id"] == 1
