import pandas as pd

from src.preprocessing import build_feature_sets, build_rfm_feature_sets


def test_build_feature_sets_excludes_customer_id_and_scales_features():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "gender": ["Male", "Female", "Female", "Male"],
            "age": [20, 30, 40, 50],
            "annual_income_k": [15, 30, 45, 60],
            "spending_score": [20, 40, 60, 80],
        }
    )
    feature_sets = build_feature_sets(df)

    assert set(feature_sets) == {"income_spending", "numeric_only", "numeric_plus_gender"}
    assert "customer_id" not in feature_sets["numeric_only"].feature_names
    assert "customer_id" not in feature_sets["numeric_plus_gender"].feature_names
    assert feature_sets["income_spending"].feature_names == ["annual_income_k", "spending_score"]
    assert feature_sets["income_spending"].matrix.shape == (4, 2)
    assert feature_sets["numeric_only"].matrix.shape == (4, 3)
    assert feature_sets["numeric_plus_gender"].matrix.shape[0] == 4
    assert all(name.startswith("gender_") or name in {"age", "annual_income_k", "spending_score"} for name in feature_sets["numeric_plus_gender"].feature_names)


def test_build_rfm_feature_sets_adds_log_scaled_features():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "recency_days": [0, 10, 90],
            "frequency": [1, 5, 20],
            "monetary_value": [50.0, 500.0, 5000.0],
            "average_order_value": [50.0, 100.0, 250.0],
        }
    )

    feature_sets = build_rfm_feature_sets(df)

    assert set(feature_sets) == {"rfm", "rfm_log"}
    assert feature_sets["rfm"].matrix.shape == (3, 4)
    assert feature_sets["rfm_log"].matrix.shape == (3, 4)
    assert feature_sets["rfm_log"].feature_names == [
        "log_recency_days",
        "log_frequency",
        "log_monetary_value",
        "log_average_order_value",
    ]
