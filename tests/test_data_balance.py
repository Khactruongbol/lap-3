import json

import pandas as pd

from src.data_balance import balance_mall_customers, balance_online_retail_rfm


def test_balance_mall_customers_removes_iqr_outlier_without_synthetic_rows(tmp_path):
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "age": [20, 22, 24, 26, 28, 30],
            "annual_income_k": [40, 42, 44, 46, 48, 200],
            "spending_score": [50, 52, 54, 56, 58, 60],
        }
    )
    report_file = tmp_path / "mall_balance.json"

    balanced = balance_mall_customers(df, tmp_path / "mall_balanced.csv", report_file)
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert len(balanced) == 5
    assert 6 not in set(balanced["customer_id"])
    assert report["removed_rows"] == 1
    assert report["synthetic_rows_added"] == 0


def test_balance_online_retail_rfm_removes_upper_tail_outliers(tmp_path):
    df = pd.DataFrame(
        {
            "customer_id": list(range(1, 102)),
            "recency_days": [10] * 101,
            "frequency": [2] * 100 + [200],
            "monetary_value": [100.0] * 100 + [100000.0],
            "average_order_value": [50.0] * 100 + [50000.0],
            "last_purchase_date": pd.to_datetime(["2024-01-01"] * 101),
        }
    )
    report_file = tmp_path / "rfm_balance.json"

    balanced = balance_online_retail_rfm(
        df,
        tmp_path / "rfm_balanced.csv",
        report_file,
        upper_quantile=0.98,
    )
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert len(balanced) == 100
    assert 101 not in set(balanced["customer_id"])
    assert report["removed_rows"] == 1
    assert report["synthetic_rows_added"] == 0
