"""Run the whole pipeline in order: tests, validation, (optional) tuning, final fit, predictions,
official scorer, report.

Usage:
    python run_all.py                  # everything except tuning (~9 min)
    python run_all.py --skip-evaluation  # reuse artifacts/cv_*.csv from a previous run (~1 min)
    python run_all.py --tune           # also run the small tuning grid (+~4 min)
"""
import argparse
import subprocess
import sys


def run(*args):
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-evaluation", action="store_true")
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    run("-m", "pytest", "-q", "tests")
    if not a.skip_evaluation:
        run("src/evaluate.py")
    if a.tune:
        run("src/tune.py")
    run("src/train.py")
    run("src/predict.py")
    run("score.py", "--predictions", "outputs/validation_predictions.csv",
        "--december-predictions", "outputs/december_chart_predictions.csv",
        "--output-dir", "outputs/scorer_results")
    run("src/report.py")
    print("\nDone: outputs/validation_predictions.csv, outputs/scorer_results/candidate_december.png, "
          "reports/assessment_report.pdf")


if __name__ == "__main__":
    main()
