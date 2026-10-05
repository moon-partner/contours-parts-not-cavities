"""One-command reproduction of every table, check, and figure.

    python run_all.py           # cached results (default): ~3 min, no training
    python run_all.py --fresh   # also train any missing prop2 runs (~3 min/run, CPU)

Steps
-----
1. core smoke tests                     (test_core.py)
2. proposition-1 experiment             (prop1/run_experiments.py -> results.csv, ~2 min)
3. proposition-2 statistics             (prop2 stats from cached runs; trains only if --fresh)
4. skeletonizer contact-gap baseline    (analyze_path_fn.py)
5. report-data verification             (verify_reports.py, 94 assertions)
6. figures                              (make_figures.py)
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def step(n, title, args):
    print(f"\n[{n}] {title}", flush=True)
    print("-" * 70, flush=True)
    r = subprocess.run([sys.executable] + args, cwd=HERE)
    if r.returncode != 0:
        print(f"*** step {n} ({title}) failed with exit {r.returncode} ***")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="train missing prop2 runs before statistics (slow)")
    args = ap.parse_args()

    step(1, "core smoke tests", ["test_core.py"])
    step(2, "proposition-1 experiment (7 conditions x 35 objects)",
         [os.path.join("prop1", "run_experiments.py")])
    if args.fresh:
        step(3, "proposition-2 training (missing runs only)",
             [os.path.join("prop2", "run_prop2.py"), "full"])
    step(3 if not args.fresh else 4, "proposition-2 statistics",
         [os.path.join("prop2", "run_prop2.py"), "stats"])
    step(4 if not args.fresh else 5, "paths-only baseline (C0/C4/C5)",
         ["analyze_path_fn.py"])
    step(5 if not args.fresh else 6, "report verification (94 assertions)",
         ["verify_reports.py"])
    step(6 if not args.fresh else 7, "figures", ["make_figures.py"])

    print("\n" + "=" * 70)
    print("done: tables + verification + figures reproduced from raw results.")
    print("  prop1/report.md  prop2/report.md  ZONG_REPORT.md   -> findings")
    print("  PITFALLS.md      SHOOTING.md                   -> reuse docs")
    print("=" * 70)


if __name__ == "__main__":
    main()
