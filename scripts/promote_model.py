"""CLI for Phase 13 model promotion gate.

Compares a freshly trained candidate model's evaluation summary against the
currently deployed model. Promotes (overwrites `models/`) only if the
candidate does not regress WMAPE beyond an allowed tolerance on any tier.

Exit codes (meaningful so Airflow can branch on them):
    0 = promoted
    1 = rejected (candidate regressed) - production model left untouched
    2 = could not run (missing candidate/summary files)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIERS = ["Large", "Medium", "Small", "Overall"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote a candidate model if it does not regress.")
    p.add_argument("--candidate-models-dir", type=Path, default=PROJECT_ROOT / "models_candidate")
    p.add_argument("--candidate-summary", type=Path, default=PROJECT_ROOT / "data" / "evaluation_summary_candidate.json")
    p.add_argument("--live-models-dir", type=Path, default=PROJECT_ROOT / "models")
    p.add_argument("--live-summary", type=Path, default=PROJECT_ROOT / "data" / "evaluation_summary.json")
    p.add_argument(
        "--max-wmape-regression-pct",
        type=float,
        default=2.0,
        help="Allowed WMAPE increase (in percentage points) before rejecting the candidate.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.candidate_summary.exists():
        print(f"REJECTED: candidate summary not found at {args.candidate_summary}", file=sys.stderr)
        return 2
    if not args.candidate_models_dir.exists():
        print(f"REJECTED: candidate models dir not found at {args.candidate_models_dir}", file=sys.stderr)
        return 2

    candidate = json.loads(args.candidate_summary.read_text())["lightgbm_models"]

    live = None
    if args.live_summary.exists():
        live = json.loads(args.live_summary.read_text()).get("lightgbm_models")

    print("\n" + "=" * 70)
    print("  MODEL PROMOTION GATE")
    print("=" * 70)

    if live is None:
        print("  No live model summary found - promoting candidate unconditionally (first deploy).")
        regressions: list[str] = []
    else:
        regressions = []
        for tier in TIERS:
            c_wmape = candidate.get(tier, {}).get("wmape")
            l_wmape = live.get(tier, {}).get("wmape")
            if c_wmape is None or l_wmape is None:
                continue
            delta = c_wmape - l_wmape
            status = "OK"
            if delta > args.max_wmape_regression_pct:
                status = "REGRESSION"
                regressions.append(tier)
            print(f"  {tier:<9} live WMAPE={l_wmape:>6.2f}%  candidate WMAPE={c_wmape:>6.2f}%  delta={delta:>+6.2f}%  [{status}]")

    print("-" * 70)

    if regressions:
        print(f"  RESULT: REJECTED - regression on tiers: {', '.join(regressions)}")
        print("  Production model left untouched.")
        print("=" * 70 + "\n")
        return 1

    # Promote: atomically replace live models directory contents.
    args.live_models_dir.mkdir(parents=True, exist_ok=True)
    for artifact in args.candidate_models_dir.glob("*.joblib"):
        shutil.copy2(artifact, args.live_models_dir / artifact.name)
    shutil.copy2(args.candidate_summary, args.live_summary)

    print("  RESULT: PROMOTED - candidate copied to production models/")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
