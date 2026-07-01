import pandas as pd

from src.preprocessing import build_feature_sets


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

    assert set(feature_sets) == {"numeric_only", "numeric_plus_gender"}
    assert "customer_id" not in feature_sets["numeric_only"].feature_names
    assert "customer_id" not in feature_sets["numeric_plus_gender"].feature_names
    assert feature_sets["numeric_only"].matrix.shape == (4, 3)
    assert feature_sets["numeric_plus_gender"].matrix.shape[0] == 4
    assert all(name.startswith("gender_") or name in {"age", "annual_income_k", "spending_score"} for name in feature_sets["numeric_plus_gender"].feature_names)
