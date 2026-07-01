from __future__ import annotations

import csv
import shutil
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    DATA_SOURCE_LOG,
    DATASET_SLUG,
    FALLBACK_DATASET_SLUG,
    ONLINE_RETAIL_DATASET_PAGE,
    ONLINE_RETAIL_URL,
    RAW_DATA_FILE,
    RAW_ONLINE_RETAIL_ARCHIVE,
    RAW_ONLINE_RETAIL_FILE,
    REQUIRED_COLUMNS,
    ensure_directories,
)


class DataAcquisitionError(RuntimeError):
    """Raised when no schema-compatible customer segmentation dataset is found."""


def _has_required_schema(csv_path: Path) -> bool:
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception:
        return False
    return all(column in columns for column in REQUIRED_COLUMNS)


def _candidate_csv_files(dataset_dir: Path) -> Iterable[Path]:
    return sorted(dataset_dir.rglob("*.csv"))


def _find_schema_compatible_csv(dataset_dir: Path) -> Path:
    for csv_path in _candidate_csv_files(dataset_dir):
        if _has_required_schema(csv_path):
            return csv_path
    raise DataAcquisitionError(
        f"No CSV with required columns found under {dataset_dir}. "
        f"Required columns: {REQUIRED_COLUMNS}"
    )


def _write_source_log(source_name: str, url: str, access_method: str, local_file: Path, license_note: str) -> None:
    DATA_SOURCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    file_exists = DATA_SOURCE_LOG.exists()
    with DATA_SOURCE_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_name",
                "url",
                "access_method",
                "download_date",
                "license_note",
                "local_file",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "source_name": source_name,
                "url": url,
                "access_method": access_method,
                "download_date": date.today().isoformat(),
                "license_note": license_note,
                "local_file": str(local_file.as_posix()),
            }
        )


def _download_with_kagglehub(dataset_slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise DataAcquisitionError(
            "kagglehub is not installed. Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    return Path(kagglehub.dataset_download(dataset_slug))


def fetch_data(force: bool = False) -> Path:
    """Download a schema-compatible Mall Customers dataset and preserve it as raw data."""

    ensure_directories()
    if RAW_DATA_FILE.exists() and not force and _has_required_schema(RAW_DATA_FILE):
        return RAW_DATA_FILE

    attempts = [
        (DATASET_SLUG, "primary"),
        (FALLBACK_DATASET_SLUG, "fallback"),
    ]
    errors: list[str] = []
    for slug, source_name in attempts:
        try:
            dataset_dir = _download_with_kagglehub(slug)
            source_csv = _find_schema_compatible_csv(dataset_dir)
            shutil.copy2(source_csv, RAW_DATA_FILE)
            _write_source_log(
                source_name=f"kaggle_{source_name}",
                url=f"https://www.kaggle.com/datasets/{slug}",
                access_method="kagglehub.dataset_download",
                local_file=RAW_DATA_FILE,
                license_note="Refer to the Kaggle dataset page for the active license and usage notes.",
            )
            return RAW_DATA_FILE
        except Exception as exc:
            errors.append(f"{slug}: {exc}")

    raise DataAcquisitionError("Unable to acquire a schema-compatible dataset. " + " | ".join(errors))


def fetch_online_retail_data(force: bool = False) -> Path:
    """Download the UCI Online Retail transaction workbook and preserve it as raw data."""

    ensure_directories()
    if RAW_ONLINE_RETAIL_FILE.exists() and not force:
        return RAW_ONLINE_RETAIL_FILE

    RAW_ONLINE_RETAIL_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(ONLINE_RETAIL_URL, RAW_ONLINE_RETAIL_ARCHIVE)
        with zipfile.ZipFile(RAW_ONLINE_RETAIL_ARCHIVE) as archive:
            xlsx_members = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
            if not xlsx_members:
                raise DataAcquisitionError("The UCI Online Retail archive does not contain an .xlsx file.")
            with archive.open(xlsx_members[0]) as source, RAW_ONLINE_RETAIL_FILE.open("wb") as target:
                shutil.copyfileobj(source, target)
        _write_source_log(
            source_name="uci_online_retail",
            url=ONLINE_RETAIL_DATASET_PAGE,
            access_method="direct_zip_download",
            local_file=RAW_ONLINE_RETAIL_FILE,
            license_note="UCI Machine Learning Repository, CC BY 4.0 according to the dataset page.",
        )
        return RAW_ONLINE_RETAIL_FILE
    except Exception as exc:
        raise DataAcquisitionError(f"Unable to acquire UCI Online Retail data: {exc}") from exc


def fetch_all_data(force: bool = False) -> dict[str, Path]:
    return {
        "mall": fetch_data(force=force),
        "online_retail": fetch_online_retail_data(force=force),
    }


def require_raw_data(dataset: str = "mall") -> Path:
    if dataset == "online_retail":
        if not RAW_ONLINE_RETAIL_FILE.exists():
            raise FileNotFoundError(
                f"Online Retail raw dataset not found at {RAW_ONLINE_RETAIL_FILE}. "
                "Run `python main.py --fetch-data --dataset online_retail` or `python main.py --run-all`."
            )
        return RAW_ONLINE_RETAIL_FILE

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_FILE}. Run `python main.py --fetch-data` or `python main.py --run-all`."
        )
    if not _has_required_schema(RAW_DATA_FILE):
        raise DataAcquisitionError(f"Raw dataset exists but does not match the required schema: {RAW_DATA_FILE}")
    return RAW_DATA_FILE
