from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    BALANCED_DATA_FILE,
    DATA_BALANCE_REPORT,
    NUMERIC_FEATURES,
    ONLINE_RETAIL_BALANCE_REPORT,
    ONLINE_RETAIL_BALANCED_RFM_FILE,
    ONLINE_RETAIL_RFM_FILE,
    PROCESSED_DATA_FILE,
    RFM_FEATURES,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    stats = df[columns].describe().to_dict()
    return {
        column: {metric: float(value) for metric, value in metrics.items()}
        for column, metrics in stats.items()
    }


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def balance_mall_customers(
    clean_df: pd.DataFrame,
    output_file: Path = BALANCED_DATA_FILE,
    report_file: Path = DATA_BALANCE_REPORT,
) -> pd.DataFrame:
    bounds = {column: _iqr_bounds(clean_df[column]) for column in NUMERIC_FEATURES}
    keep_mask = pd.Series(True, index=clean_df.index)
    rules: dict[str, dict[str, float | int]] = {}
    for column, (lower, upper) in bounds.items():
        column_mask = clean_df[column].between(lower, upper)
        keep_mask &= column_mask
        rules[column] = {
            "method": "iqr_1_5",
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "removed_rows": int((~column_mask).sum()),
        }

    balanced = clean_df.loc[keep_mask].reset_index(drop=True)
    removed = clean_df.loc[~keep_mask].copy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_file, index=False, encoding="utf-8")

    report = {
        "dataset": "mall",
        "strategy": "remove_iqr_outliers_for_training_only",
        "reason": "Distance-based clustering is sensitive to extreme values; raw and clean data are preserved separately.",
        "original_rows": int(len(clean_df)),
        "balanced_rows": int(len(balanced)),
        "removed_rows": int(len(removed)),
        "removed_ratio": float(len(removed) / len(clean_df)) if len(clean_df) else 0.0,
        "removed_customer_ids": removed["customer_id"].astype(int).tolist() if "customer_id" in removed else [],
        "rules": rules,
        "summary_before": _summary(clean_df, NUMERIC_FEATURES),
        "summary_after": _summary(balanced, NUMERIC_FEATURES),
        "synthetic_rows_added": 0,
    }
    _write_json(report_file, report)
    return balanced


def balance_online_retail_rfm(
    rfm_df: pd.DataFrame,
    output_file: Path = ONLINE_RETAIL_BALANCED_RFM_FILE,
    report_file: Path = ONLINE_RETAIL_BALANCE_REPORT,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    skew_sensitive_features = ["frequency", "monetary_value", "average_order_value"]
    thresholds = {
        column: float(rfm_df[column].quantile(upper_quantile))
        for column in skew_sensitive_features
    }
    keep_mask = pd.Series(True, index=rfm_df.index)
    rules: dict[str, dict[str, float | int]] = {}
    for column, upper in thresholds.items():
        column_mask = rfm_df[column] <= upper
        keep_mask &= column_mask
        rules[column] = {
            "method": f"upper_quantile_{upper_quantile}",
            "upper_bound": upper,
            "removed_rows": int((~column_mask).sum()),
        }

    balanced = rfm_df.loc[keep_mask].reset_index(drop=True)
    removed = rfm_df.loc[~keep_mask].copy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_file, index=False, encoding="utf-8")

    report = {
        "dataset": "online_retail",
        "strategy": "remove_upper_tail_rfm_outliers_for_training_only",
        "reason": "RFM transaction values are highly right-skewed; removing the top tail avoids clusters dominated by a few extreme customers.",
        "original_rows": int(len(rfm_df)),
        "balanced_rows": int(len(balanced)),
        "removed_rows": int(len(removed)),
        "removed_ratio": float(len(removed) / len(rfm_df)) if len(rfm_df) else 0.0,
        "removed_customer_ids": removed["customer_id"].astype(int).tolist() if "customer_id" in removed else [],
        "rules": rules,
        "summary_before": _summary(rfm_df, RFM_FEATURES),
        "summary_after": _summary(balanced, RFM_FEATURES),
        "synthetic_rows_added": 0,
    }
    _write_json(report_file, report)
    return balanced


def load_balanced_mall_data(output_file: Path = BALANCED_DATA_FILE) -> pd.DataFrame:
    if not output_file.exists():
        clean_df = pd.read_csv(PROCESSED_DATA_FILE, encoding="utf-8")
        return balance_mall_customers(clean_df, output_file)
    return pd.read_csv(output_file, encoding="utf-8")


def load_balanced_rfm_data(output_file: Path = ONLINE_RETAIL_BALANCED_RFM_FILE) -> pd.DataFrame:
    if not output_file.exists():
        rfm_df = pd.read_csv(ONLINE_RETAIL_RFM_FILE, encoding="utf-8", parse_dates=["last_purchase_date"])
        return balance_online_retail_rfm(rfm_df, output_file)
    return pd.read_csv(output_file, encoding="utf-8", parse_dates=["last_purchase_date"])
