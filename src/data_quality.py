from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CANONICAL_COLUMNS,
    DATA_QUALITY_ISSUES,
    DATA_QUALITY_REPORT,
    PROCESSED_DATA_FILE,
    REQUIRED_COLUMNS,
    STANDARD_COLUMNS,
    VALID_GENDERS,
    ensure_directories,
)


class DataQualityError(ValueError):
    """Raised when input data violates required schema or value rules."""


def validate_raw_schema(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise DataQualityError(f"Missing required raw columns: {missing}")


def _normalize_gender(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text == "male":
        return "Male"
    if text == "female":
        return "Female"
    return str(value).strip()


def _iqr_outlier_count(series: pd.Series) -> int:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_customer_data(raw_file: Path) -> pd.DataFrame:
    ensure_directories()
    raw_df = pd.read_csv(raw_file, encoding="utf-8")
    validate_raw_schema(raw_df)

    df = raw_df.rename(columns=STANDARD_COLUMNS)[CANONICAL_COLUMNS].copy()
    df["gender"] = df["gender"].map(_normalize_gender)
    for column in ["age", "annual_income_k", "spending_score"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    issue_rows: list[dict[str, Any]] = []
    for idx, row in df[df.isna().any(axis=1)].iterrows():
        issue_rows.append({"row_index": int(idx), "issue": "missing_value", "values": row.to_dict()})

    duplicate_customer_mask = df.duplicated("customer_id", keep=False)
    for idx, row in df[duplicate_customer_mask].iterrows():
        issue_rows.append({"row_index": int(idx), "issue": "duplicate_customer_id", "values": row.to_dict()})

    invalid_mask = (
        (df["age"] <= 0)
        | (df["annual_income_k"] < 0)
        | (df["spending_score"] < 1)
        | (df["spending_score"] > 100)
        | (~df["gender"].isin(VALID_GENDERS))
    )
    for idx, row in df[invalid_mask].iterrows():
        issue_rows.append({"row_index": int(idx), "issue": "invalid_value_range", "values": row.to_dict()})

    cleaned = df.dropna(subset=CANONICAL_COLUMNS).copy()
    cleaned = cleaned[
        (cleaned["age"] > 0)
        & (cleaned["annual_income_k"] >= 0)
        & (cleaned["spending_score"].between(1, 100))
        & (cleaned["gender"].isin(VALID_GENDERS))
    ]
    cleaned = cleaned.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)

    if cleaned.empty:
        raise DataQualityError("No valid rows remain after schema and value validation.")

    issue_df = pd.DataFrame(issue_rows)
    if not issue_df.empty:
        DATA_QUALITY_ISSUES.parent.mkdir(parents=True, exist_ok=True)
        issue_df.to_csv(DATA_QUALITY_ISSUES, index=False, encoding="utf-8")

    report = {
        "raw_shape": list(raw_df.shape),
        "clean_shape": list(cleaned.shape),
        "raw_columns": raw_df.columns.tolist(),
        "canonical_columns": cleaned.columns.tolist(),
        "missing_values_raw": raw_df[REQUIRED_COLUMNS].isna().sum().to_dict(),
        "duplicated_rows_raw": int(raw_df.duplicated().sum()),
        "duplicated_customer_id_raw": int(df.duplicated("customer_id").sum()),
        "gender_distribution": cleaned["gender"].value_counts().to_dict(),
        "numeric_describe": cleaned[["age", "annual_income_k", "spending_score"]].describe().to_dict(),
        "outlier_counts_iqr": {
            column: _iqr_outlier_count(cleaned[column])
            for column in ["age", "annual_income_k", "spending_score"]
        },
        "issues_file": str(DATA_QUALITY_ISSUES.as_posix()) if issue_rows else None,
    }
    _write_json(DATA_QUALITY_REPORT, report)
    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_FILE, index=False, encoding="utf-8")
    return cleaned


def load_clean_data(processed_file: Path = PROCESSED_DATA_FILE) -> pd.DataFrame:
    if not processed_file.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {processed_file}. Run `python main.py --run-eda`.")
    df = pd.read_csv(processed_file, encoding="utf-8")
    missing = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise DataQualityError(f"Processed data is missing canonical columns: {missing}")
    return df
