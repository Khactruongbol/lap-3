from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    ONLINE_RETAIL_CLEAN_FILE,
    ONLINE_RETAIL_QUALITY_ISSUES,
    ONLINE_RETAIL_QUALITY_REPORT,
    ONLINE_RETAIL_RFM_FILE,
    RAW_ONLINE_RETAIL_FILE,
    ensure_directories,
)


ONLINE_RETAIL_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


class OnlineRetailDataError(ValueError):
    """Raised when Online Retail transaction data cannot be cleaned or converted to RFM."""


def validate_online_retail_schema(df: pd.DataFrame) -> None:
    missing = [column for column in ONLINE_RETAIL_COLUMNS if column not in df.columns]
    if missing:
        raise OnlineRetailDataError(f"Missing Online Retail columns: {missing}")


def load_online_retail_raw(raw_file: Path = RAW_ONLINE_RETAIL_FILE) -> pd.DataFrame:
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Online Retail raw file not found at {raw_file}. "
            "Run `python main.py --fetch-data --dataset online_retail`."
        )
    df = pd.read_excel(raw_file)
    validate_online_retail_schema(df)
    return df


def clean_online_retail_transactions(raw_df: pd.DataFrame, write_outputs: bool = True) -> pd.DataFrame:
    validate_online_retail_schema(raw_df)
    df = raw_df.copy()
    original_rows = len(df)

    df["invoice_no"] = df["InvoiceNo"].astype(str).str.strip()
    df["customer_id"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["invoice_date"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["country"] = df["Country"].astype(str).str.strip()
    df["description"] = df["Description"].astype(str).str.strip()
    df["stock_code"] = df["StockCode"].astype(str).str.strip()

    issues = []
    issue_specs = {
        "cancellation": df["invoice_no"].str.lower().str.startswith("c", na=False),
        "missing_customer_id": df["customer_id"].isna(),
        "non_positive_quantity": df["quantity"].isna() | (df["quantity"] <= 0),
        "non_positive_unit_price": df["unit_price"].isna() | (df["unit_price"] <= 0),
        "malformed_invoice_date": df["invoice_date"].isna(),
    }
    for issue_name, mask in issue_specs.items():
        issues.append({"issue": issue_name, "rows": int(mask.sum())})

    valid_mask = ~(
        issue_specs["cancellation"]
        | issue_specs["missing_customer_id"]
        | issue_specs["non_positive_quantity"]
        | issue_specs["non_positive_unit_price"]
        | issue_specs["malformed_invoice_date"]
    )
    cleaned = df.loc[
        valid_mask,
        [
            "invoice_no",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "unit_price",
            "customer_id",
            "country",
        ],
    ].copy()
    cleaned["customer_id"] = cleaned["customer_id"].astype(int)
    cleaned["line_total"] = cleaned["quantity"] * cleaned["unit_price"]

    if cleaned.empty:
        raise OnlineRetailDataError("No valid Online Retail rows remain after cleaning.")

    if write_outputs:
        ONLINE_RETAIL_QUALITY_ISSUES.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(issues).to_csv(ONLINE_RETAIL_QUALITY_ISSUES, index=False, encoding="utf-8")

        report = {
            "raw_shape": [original_rows, raw_df.shape[1]],
            "clean_shape": list(cleaned.shape),
            "removed_rows": int(original_rows - len(cleaned)),
            "issue_counts": {row["issue"]: row["rows"] for row in issues},
            "customer_count": int(cleaned["customer_id"].nunique()),
            "invoice_count": int(cleaned["invoice_no"].nunique()),
            "country_count": int(cleaned["country"].nunique()),
        }
        ONLINE_RETAIL_QUALITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
        ONLINE_RETAIL_QUALITY_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        ONLINE_RETAIL_CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(ONLINE_RETAIL_CLEAN_FILE, index=False, encoding="utf-8")
    return cleaned


def build_rfm_features(cleaned_df: pd.DataFrame, write_outputs: bool = True) -> pd.DataFrame:
    required = ["customer_id", "invoice_no", "invoice_date", "line_total"]
    missing = [column for column in required if column not in cleaned_df.columns]
    if missing:
        raise OnlineRetailDataError(f"Cleaned Online Retail data is missing columns: {missing}")

    snapshot_date = cleaned_df["invoice_date"].max() + pd.Timedelta(days=1)
    grouped = cleaned_df.groupby("customer_id")
    rfm = grouped.agg(
        last_purchase_date=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary_value=("line_total", "sum"),
    ).reset_index()
    rfm["recency_days"] = (snapshot_date - rfm["last_purchase_date"]).dt.days
    rfm["average_order_value"] = rfm["monetary_value"] / rfm["frequency"].clip(lower=1)
    rfm = rfm[
        [
            "customer_id",
            "recency_days",
            "frequency",
            "monetary_value",
            "average_order_value",
            "last_purchase_date",
        ]
    ].sort_values("customer_id").reset_index(drop=True)
    if write_outputs:
        ONLINE_RETAIL_RFM_FILE.parent.mkdir(parents=True, exist_ok=True)
        rfm.to_csv(ONLINE_RETAIL_RFM_FILE, index=False, encoding="utf-8")
    return rfm


def process_online_retail(raw_file: Path = RAW_ONLINE_RETAIL_FILE) -> pd.DataFrame:
    ensure_directories()
    raw_df = load_online_retail_raw(raw_file)
    cleaned = clean_online_retail_transactions(raw_df, write_outputs=True)
    return build_rfm_features(cleaned, write_outputs=True)


def load_or_process_online_retail(raw_file: Path = RAW_ONLINE_RETAIL_FILE, force: bool = False) -> pd.DataFrame:
    if (
        not force
        and ONLINE_RETAIL_RFM_FILE.exists()
        and ONLINE_RETAIL_CLEAN_FILE.exists()
        and raw_file.exists()
        and ONLINE_RETAIL_RFM_FILE.stat().st_mtime >= raw_file.stat().st_mtime
    ):
        return load_rfm_data(ONLINE_RETAIL_RFM_FILE)
    return process_online_retail(raw_file)


def load_rfm_data(rfm_file: Path = ONLINE_RETAIL_RFM_FILE) -> pd.DataFrame:
    if not rfm_file.exists():
        raise FileNotFoundError(
            f"Online Retail RFM file not found at {rfm_file}. Run `python main.py --run-eda --dataset online_retail`."
        )
    df = pd.read_csv(rfm_file, encoding="utf-8", parse_dates=["last_purchase_date"])
    return df
