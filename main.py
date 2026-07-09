from __future__ import annotations

import argparse

from config import ensure_directories
from src.data_acquisition import fetch_all_data, fetch_data, fetch_online_retail_data, require_raw_data
from src.data_balance import balance_mall_customers, balance_online_retail_rfm
from src.data_quality import clean_customer_data
from src.model_train import train_mall_model, train_online_retail_model
from src.online_retail import load_or_process_online_retail
from src.reporting import write_notebook_stub
from src.visualization import (
    generate_balance_figures,
    generate_eda_figures,
    generate_rfm_balance_figures,
    generate_rfm_eda_figures,
)


def run_mall_eda() -> None:
    raw_file = require_raw_data("mall")
    clean_df = clean_customer_data(raw_file)
    balanced_df = balance_mall_customers(clean_df)
    generate_eda_figures(clean_df)
    generate_balance_figures(clean_df, balanced_df)


def run_online_retail_eda(force_refresh: bool = False) -> None:
    raw_file = require_raw_data("online_retail")
    rfm_df = load_or_process_online_retail(raw_file, force=force_refresh)
    balanced_rfm_df = balance_online_retail_rfm(rfm_df)
    generate_rfm_eda_figures(rfm_df)
    generate_rfm_balance_figures(rfm_df, balanced_rfm_df)


def run_eda(dataset: str, force_refresh: bool = False) -> None:
    if dataset in {"mall", "all"}:
        run_mall_eda()
    if dataset in {"online_retail", "all"}:
        run_online_retail_eda(force_refresh=force_refresh)


def train_model(dataset: str) -> None:
    if dataset in {"mall", "all"}:
        train_mall_model()
    if dataset in {"online_retail", "all"}:
        train_online_retail_model()
    write_notebook_stub()


def fetch_selected_data(dataset: str, force_fetch: bool = False) -> None:
    if dataset == "mall":
        fetch_data(force=force_fetch)
    elif dataset == "online_retail":
        fetch_online_retail_data(force=force_fetch)
    else:
        fetch_all_data(force=force_fetch)


def run_all(dataset: str, force_fetch: bool = False) -> None:
    fetch_selected_data(dataset, force_fetch)
    run_eda(dataset, force_refresh=force_fetch)
    train_model(dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Segmentation Clustering pipeline")
    parser.add_argument("--fetch-data", action="store_true", help="Download and store the raw customer dataset")
    parser.add_argument("--run-eda", action="store_true", help="Run schema validation, cleaning, and EDA figures")
    parser.add_argument("--run-clustering", action="store_true", help="Run feature scaling, clustering, evaluation, and reporting")
    parser.add_argument("--train-model", action="store_true", help="Train and save the selected unsupervised model artifact")
    parser.add_argument("--run-all", action="store_true", help="Run the complete workflow")
    parser.add_argument(
        "--dataset",
        choices=["mall", "online_retail", "all"],
        default="all",
        help="Dataset track to process. Defaults to all.",
    )
    parser.add_argument("--force-fetch", action="store_true", help="Download the raw dataset even if it already exists")
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    if args.run_all:
        run_all(dataset=args.dataset, force_fetch=args.force_fetch)
        return
    if args.fetch_data:
        fetch_selected_data(args.dataset, force_fetch=args.force_fetch)
    if args.run_eda:
        run_eda(args.dataset, force_refresh=args.force_fetch)
    if args.run_clustering or args.train_model:
        train_model(args.dataset)
    if not any([args.fetch_data, args.run_eda, args.run_clustering, args.train_model, args.run_all]):
        raise SystemExit("No action selected. Use --fetch-data, --run-eda, --train-model, --run-clustering, or --run-all.")


if __name__ == "__main__":
    main()
