"""
run_pipeline.py
───────────────
One-command full ML pipeline runner.
Executes all steps in order:
  1. Generate synthetic dataset
  2. Run EDA and save visualizations
  3. Train and evaluate all models
  4. Run demo prediction

Usage:
    python run_pipeline.py          # Full pipeline
    python run_pipeline.py --skip-eda   # Skip EDA (faster)
"""

import os
import sys
import argparse
import time

# ── Force UTF-8 output so emojis print correctly on Windows terminals ─────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
SRC_DIR  = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, DATA_DIR)


def banner(step: str, total: int, current: int) -> None:
    print(f"\n{'='*60}")
    print(f"  STEP [{current}/{total}]: {step}")
    print(f"{'='*60}")


def run_pipeline(skip_eda: bool = False) -> None:
    start_time = time.time()
    total_steps = 3 if skip_eda else 4

    # ── STEP 1: Generate data ─────────────────────────────────────────────────
    banner("Generate Synthetic HR Dataset", total_steps, 1)
    csv_path = os.path.join(DATA_DIR, "employee_data.csv")

    if os.path.exists(csv_path):
        print(f"  Dataset already exists at: {csv_path}")
        print("  Skipping generation. Delete the file to regenerate.")
    else:
        from generate_data import generate_dataset
        df = generate_dataset()
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Dataset saved: {csv_path} ({len(df)} rows)")

    # ── STEP 2: EDA ───────────────────────────────────────────────────────────
    if not skip_eda:
        banner("Exploratory Data Analysis", total_steps, 2)
        from eda import run_eda
        run_eda(csv_path)

    # ── STEP 3: Train models ──────────────────────────────────────────────────
    step_num = 3 if not skip_eda else 2
    banner("Model Training & Evaluation", total_steps, step_num)
    from train_model import train_and_evaluate
    results = train_and_evaluate(csv_path)

    # ── STEP 4: Demo prediction ───────────────────────────────────────────────
    step_num = 4 if not skip_eda else 3
    banner("Demo Prediction (Dry Run)", total_steps, step_num)
    try:
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "predict.py"), "--demo"],
            check=True
        )
    except Exception as e:
        print(f"  ⚠️  Demo prediction skipped: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Best Model : {results.get('best_model', 'N/A')}")
    print(f"\n  🚀 Launch the web dashboard:")
    print(f"     streamlit run app/streamlit_app.py")
    print(f"\n  🖥️  Or use the console interface:")
    print(f"     python src/predict.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Employee Attrition ML Pipeline")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA step for faster execution")
    args = parser.parse_args()
    run_pipeline(skip_eda=args.skip_eda)
