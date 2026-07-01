import pandas as pd
import pytest

from src.data_quality import DataQualityError, validate_raw_schema


def test_validate_raw_schema_accepts_required_columns():
    df = pd.DataFrame(
        columns=[
            "CustomerID",
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
        ]
    )
    validate_raw_schema(df)


def test_validate_raw_schema_rejects_missing_columns():
    df = pd.DataFrame(columns=["CustomerID", "Gender", "Age"])
    with pytest.raises(DataQualityError):
        validate_raw_schema(df)
