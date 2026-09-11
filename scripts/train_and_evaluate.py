"""CLI Script: Train 3-Tier LightGBM Models and Evaluate Against Baselines (Phase 4).

Executes the full pipeline:
  1. Loads verified modeling data from Parquet
  2. Builds leak-free lag, rolling, and calendar features
  3. Splits chronologically into Train (70%), Validation (15%), Test (15%)
  4. Trains Large, Medium, and Small tier LightGBM models with early stopping
  5. Evaluates on the unseen test split against:
     - Naive Persistence (demand[t])
     - EIA Day-Ahead Forecast Benchmark
  6. Outputs the headline metrics comparison table (MAE, SMAPE, WMAPE, R²)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate import load_modeling_frame, load_modeling_respondents
from src.evaluation.metrics import evaluate_dataframe
from src.features.build_features import (
    FEATURE_COLUMNS,
    build_features,
    split_time_series,
)
from src.models.tier_model import predict_with_tier_models, train_all_tiers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("eia.pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate 3-tier forecasting models.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "eia",
        help="Path to raw EIA Parquet store",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "_logs" / "validation_report.json",
        help="Path to validation report JSON",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Analysis window start date",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Analysis window end date",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation_summary.json",
        help="Path to output evaluation summary JSON",
    )
    return parser.parse_args()


def print_comparison_table(
    eval_model: dict[str, dict],
    eval_eia: dict[str, dict],
    eval_naive: dict[str, dict],
) -> None:
    """Print an ASCII comparison table matching PROBLEM_STATEMENT.md requirements."""
    tiers = ["Large", "Medium", "Small", "Overall"]

    print("\n" + "=" * 94)
    print("  MODEL VS. BASELINES PERFORMANCE COMPARISON (TEST SPLIT - 15% UNSEEN)")
    print("=" * 94)
    print(
        f"{'Tier':<9} | {'Approach':<20} | {'MAE (MWh)':<12} | {'SMAPE (%)':<10} | {'WMAPE (%)':<10} | {'R2':<7} | {'Samples':<8}"
    )
    print("-" * 94)

    for tier in tiers:
        m = eval_model.get(tier, {})
        e = eval_eia.get(tier, {})
        n = eval_naive.get(tier, {})

        # 1. Tier LightGBM Model
        print(
            f"{tier:<9} | {'LightGBM (Ours)':<20} | {m.get('mae', 0):>12,.1f} | {m.get('smape', 0):>9.2f}% | {m.get('wmape', 0):>9.2f}% | {m.get('r2', 0):>7.4f} | {m.get('n_samples', 0):>8,}"
        )
        # 2. EIA Day-Ahead Benchmark
        print(
            f"{'':<9} | {'EIA Day-Ahead':<20} | {e.get('mae', 0):>12,.1f} | {e.get('smape', 0):>9.2f}% | {e.get('wmape', 0):>9.2f}% | {e.get('r2', 0):>7.4f} | {e.get('n_samples', 0):>8,}"
        )
        # 3. Naive Persistence Baseline
        print(
            f"{'':<9} | {'Naive Persistence':<20} | {n.get('mae', 0):>12,.1f} | {n.get('smape', 0):>9.2f}% | {n.get('wmape', 0):>9.2f}% | {n.get('r2', 0):>7.4f} | {n.get('n_samples', 0):>8,}"
        )
        print("-" * 94)

    print("=" * 94 + "\n")


def print_delta_report(prior_summary: dict, new_eval: dict) -> None:
    """Print performance changes comparing prior model to updated model."""
    prior_models = prior_summary.get("lightgbm_models", {})
    if not prior_models:
        return

    print("\n" + "=" * 76)
    print("  PERFORMANCE DELTA REPORT (IMPACT OF CODE, HOLIDAY & GUARDRAIL FIXES)")
    print("=" * 76)
    print(
        f"{'Tier':<9} | {'Prior SMAPE':<12} | {'New SMAPE':<12} | {'Delta':<10} | {'Prior WMAPE':<12} | {'New WMAPE':<10}"
    )
    print("-" * 76)

    for tier in ["Large", "Medium", "Small", "Overall"]:
        old_m = prior_models.get(tier, {})
        new_m = new_eval.get(tier, {})
        old_smape = old_m.get("smape")
        new_smape = new_m.get("smape")
        old_wmape = old_m.get("wmape")
        new_wmape = new_m.get("wmape")

        if old_smape is not None and new_smape is not None:
            delta = new_smape - old_smape
            delta_str = f"{delta:>+6.2f}%"
            print(
                f"{tier:<9} | {old_smape:>10.2f}% | {new_smape:>10.2f}% | {delta_str:>10} | {old_wmape:>10.2f}% | {new_wmape:>8.2f}%"
            )
    print("=" * 76 + "\n")


def main() -> int:
    args = parse_args()

    # Read prior summary if exists for delta comparison
    prior_summary = None
    if args.output_summary.exists():
        try:
            prior_summary = json.loads(args.output_summary.read_text())
        except Exception:
            prior_summary = None

    # 1. Determine respondents to model
    respondents = None
    if args.report_path.exists():
        respondents = load_modeling_respondents(args.report_path)
        logger.info("loaded %d approved respondents from validation report", len(respondents))
    else:
        logger.warning(
            "validation report %s not found; reading all respondents available in raw store",
            args.report_path,
        )

    # 2. Load modeling panel dataframe
    logger.info("loading raw panel data from %s (start=%s, end=%s)...", args.raw_dir, args.start, args.end)
    panel_df = load_modeling_frame(
        args.raw_dir,
        respondents=respondents,
        start=args.start,
        end=args.end,
        wide=True,
    )
    logger.info("loaded %d panel observations across %d respondents", len(panel_df), panel_df["respondent"].nunique())

    # 3. Build features
    logger.info("engineering lag, rolling, calendar, holiday, and trend features...")
    features_df = build_features(panel_df, drop_na_target=True)

    # Optionally persist features to processed parquet
    proc_dir = PROJECT_ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(proc_dir / "features.parquet", index=False)
    logger.info("saved processed feature dataset to %s", proc_dir / "features.parquet")

    # 4. Split chronologically
    splits = split_time_series(features_df, train_frac=0.70, val_frac=0.15, test_frac=0.15)

    # 5. Train 3 Tier Models
    logger.info("training 3-tier LightGBM models...")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    models = train_all_tiers(
        train_df=splits.train,
        val_df=splits.val,
        feature_cols=FEATURE_COLUMNS,
        target_col="target",
        models_dir=args.models_dir,
    )

    # Print guardrail fallback summary
    print("\n" + "=" * 60)
    print("  GUARDRAIL FALLBACK REGIONS (DATA-DRIVEN FROM VALIDATION)")
    print("=" * 60)
    for tier, tm in models.items():
        fb = tm.fallback_regions
        fb_str = ", ".join(fb) if fb else "None (Model beats naive in all regions)"
        print(f"  {tier:<8} Tier: {fb_str}")
    print("=" * 60 + "\n")

    # 6. Predict on unseen Test split
    logger.info("evaluating on unseen chronological test split (%d rows)...", len(splits.test))
    test_scored = predict_with_tier_models(splits.test, models, pred_col_name="model_prediction")

    # 7. Evaluate Model, EIA Benchmark, and Naive Baseline
    eval_model = evaluate_dataframe(test_scored, pred_col="model_prediction")
    eval_eia = evaluate_dataframe(test_scored, pred_col="benchmark_eia_forecast")
    eval_naive = evaluate_dataframe(test_scored, pred_col="naive_persistence")

    # 8. Print terminal comparison
    print_comparison_table(eval_model, eval_eia, eval_naive)

    if prior_summary:
        print_delta_report(prior_summary, eval_model)

    # 9. Save JSON summary
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "test_period_start": str(splits.test["target_date"].min().date()),
        "test_period_end": str(splits.test["target_date"].max().date()),
        "test_row_count": len(splits.test),
        "guardrail_fallback_regions": {tier: tm.fallback_regions for tier, tm in models.items()},
        "lightgbm_models": eval_model,
        "eia_benchmark": eval_eia,
        "naive_baseline": eval_naive,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2))
    logger.info("saved evaluation summary to %s", args.output_summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
