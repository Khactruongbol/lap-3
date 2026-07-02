from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import NUMERIC_FEATURES, RFM_FEATURES


@dataclass
class FeatureSet:
    name: str
    matrix: object
    feature_names: list[str]
    scaler: StandardScaler


def build_feature_sets(df: pd.DataFrame) -> dict[str, FeatureSet]:
    missing = [column for column in NUMERIC_FEATURES + ["gender", "customer_id"] if column not in df.columns]
    if missing:
        raise ValueError(f"Cleaned data is missing required preprocessing columns: {missing}")

    income_spending_features = ["annual_income_k", "spending_score"]
    income_spending_df = df[income_spending_features].copy()
    income_spending_scaler = StandardScaler()
    income_spending_scaled = income_spending_scaler.fit_transform(income_spending_df)

    numeric_df = df[NUMERIC_FEATURES].copy()
    numeric_scaler = StandardScaler()
    numeric_scaled = numeric_scaler.fit_transform(numeric_df)

    gender_df = pd.get_dummies(df["gender"], prefix="gender", dtype=float)
    with_gender_df = pd.concat([numeric_df, gender_df], axis=1)
    with_gender_scaler = StandardScaler()
    with_gender_scaled = with_gender_scaler.fit_transform(with_gender_df)

    return {
        "income_spending": FeatureSet(
            name="income_spending",
            matrix=income_spending_scaled,
            feature_names=income_spending_features,
            scaler=income_spending_scaler,
        ),
        "numeric_only": FeatureSet(
            name="numeric_only",
            matrix=numeric_scaled,
            feature_names=NUMERIC_FEATURES.copy(),
            scaler=numeric_scaler,
        ),
        "numeric_plus_gender": FeatureSet(
            name="numeric_plus_gender",
            matrix=with_gender_scaled,
            feature_names=with_gender_df.columns.tolist(),
            scaler=with_gender_scaler,
        ),
    }


def build_rfm_feature_sets(df: pd.DataFrame) -> dict[str, FeatureSet]:
    missing = [column for column in RFM_FEATURES + ["customer_id"] if column not in df.columns]
    if missing:
        raise ValueError(f"RFM data is missing required preprocessing columns: {missing}")

    rfm_df = df[RFM_FEATURES].copy()
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)

    rfm_log_df = pd.DataFrame(
        {
            f"log_{column}": np.log1p(rfm_df[column].clip(lower=0))
            for column in RFM_FEATURES
        },
        index=rfm_df.index,
    )
    log_scaler = StandardScaler()
    rfm_log_scaled = log_scaler.fit_transform(rfm_log_df)

    return {
        "rfm": FeatureSet(
            name="rfm",
            matrix=rfm_scaled,
            feature_names=RFM_FEATURES.copy(),
            scaler=scaler,
        ),
        "rfm_log": FeatureSet(
            name="rfm_log",
            matrix=rfm_log_scaled,
            feature_names=rfm_log_df.columns.tolist(),
            scaler=log_scaler,
        ),
    }
