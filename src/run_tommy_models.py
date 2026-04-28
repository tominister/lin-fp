import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data.datasets import load_recovery, load_welfake
from models.lstm_model import run_lstm
from models.xgboost_model import run_xgboost
from preprocessing.keywords import DEFAULT_SOURCE_TERMS
from results_utils.plotting import save_model_metric_plot
from results_utils.reporting import (
    compute_label_balance,
    detect_overfitting,
    generate_report,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_dir(root: Path) -> Path:
    candidates = [root / "News_Dataset", root / "news_datasets"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_DIR = _resolve_data_dir(ROOT)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WELFAKE_PATH = DATA_DIR / "WELFake_Dataset.csv"
RECOVERY_PATH = DATA_DIR / "recovery-news-data.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost and LSTM on WELFake (80/20 split) and test on Recovery."
    )
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--lstm-epochs", type=int, default=6)
    parser.add_argument("--lstm-batch-size", type=int, default=96)
    parser.add_argument(
        "--disable-keyword-removal",
        action="store_true",
        help="Disable source keyword stripping in preprocessing.",
    )
    parser.add_argument(
        "--extra-blocked-terms",
        nargs="*",
        default=[],
        help="Additional terms to remove during preprocessing.",
    )
    return parser.parse_args()


def build_payload(
    welfake_df,
    recovery_df,
    X_train,
    X_val,
    welfake_balance,
    recovery_balance,
    train_balance,
    val_balance,
    xgboost_results,
    lstm_results,
    xgb_overfitting,
    lstm_overfitting,
):
    return {
        "train_setup": {
            "train_dataset": "WELFake_Dataset.csv",
            "external_test_dataset": "recovery-news-data.csv",
            "split": "WELFake stratified 80/20",
            "seed": SEED,
        },
        "dataset_sizes": {
            "welfake_total": int(len(welfake_df)),
            "welfake_train": int(len(X_train)),
            "welfake_val": int(len(X_val)),
            "recovery_total": int(len(recovery_df)),
        },
        "label_balance": {
            "welfake_full": welfake_balance,
            "welfake_train": train_balance,
            "welfake_val": val_balance,
            "recovery_full": recovery_balance,
        },
        "models": {
            "xgboost": xgboost_results,
            "lstm": lstm_results,
        },
    }


def main():
    args = parse_args()

    # Determine blocked terms
    blocked_terms = list(DEFAULT_SOURCE_TERMS)
    if args.extra_blocked_terms:
        blocked_terms.extend(args.extra_blocked_terms)
    if args.disable_keyword_removal:
        blocked_terms = None

    # Load data
    print("Loading datasets...")
    welfake_df = load_welfake(WELFAKE_PATH, blocked_terms=blocked_terms)
    recovery_df = load_recovery(RECOVERY_PATH, blocked_terms=blocked_terms)

    # Compute label balance
    welfake_balance = compute_label_balance(welfake_df["label"])
    recovery_balance = compute_label_balance(recovery_df["label"])

    # Split WELFake into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        welfake_df["input_text"],
        welfake_df["label"],
        test_size=0.2,
        random_state=SEED,
        stratify=welfake_df["label"],
    )

    train_balance = compute_label_balance(y_train)
    val_balance = compute_label_balance(y_val)
    y_recovery = recovery_df["label"]

    # Run XGBoost
    print("\n" + "=" * 80)
    print("Training XGBoost...")
    print("=" * 80)
    xgboost_results = run_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        recovery_df["input_text"],
        y_recovery,
        args,
    )
    xgb_overfitting = detect_overfitting(
        xgboost_results["welfake_validation_20pct"],
        xgboost_results["recovery_external_test"],
        "XGBoost",
    )

    # Run LSTM
    print("\n" + "=" * 80)
    print("Training LSTM...")
    print("=" * 80)
    lstm_results = run_lstm(
        X_train,
        y_train,
        X_val,
        y_val,
        recovery_df["input_text"],
        y_recovery,
        args,
    )
    lstm_overfitting = detect_overfitting(
        lstm_results["welfake_validation_20pct"],
        lstm_results["recovery_external_test"],
        "LSTM",
    )

    # Build results payload
    payload = build_payload(
        welfake_df,
        recovery_df,
        X_train,
        X_val,
        welfake_balance,
        recovery_balance,
        train_balance,
        val_balance,
        xgboost_results,
        lstm_results,
        xgb_overfitting,
        lstm_overfitting,
    )

    # Save metrics
    output_path = RESULTS_DIR / "direct_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics: {output_path}")

    # Save plots
    xgb_plot_path = RESULTS_DIR / "xgboost_model_metrics.png"
    save_model_metric_plot("XGBoost", xgboost_results, xgb_plot_path)
    print(f"Saved plot: {xgb_plot_path}")

    lstm_plot_path = RESULTS_DIR / "lstm_model_metrics.png"
    save_model_metric_plot("LSTM", lstm_results, lstm_plot_path)
    print(f"Saved plot: {lstm_plot_path}")

    # Generate report
    report_path = RESULTS_DIR / "REPORT.txt"
    generate_report(
        payload,
        welfake_balance,
        recovery_balance,
        xgb_overfitting,
        lstm_overfitting,
        report_path,
    )
    print(f"Saved report: {report_path}")

    print("\n" + "=" * 80)
    print("LABEL BALANCE:")
    print(f"  WELFake full: {welfake_balance['fake_count']} fake, {welfake_balance['real_count']} real")
    print(f"  Recovery full: {recovery_balance['fake_count']} fake, {recovery_balance['real_count']} real")
    print("\nOVERFITTING ANALYSIS:")
    print(f"  XGBoost: {xgb_overfitting['overfitting_severity']} (accuracy drop: {xgb_overfitting['accuracy_drop']})")
    print(f"  LSTM: {lstm_overfitting['overfitting_severity']} (accuracy drop: {lstm_overfitting['accuracy_drop']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
