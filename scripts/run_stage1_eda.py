"""Stage 1 entry point.

Steps:
  1. Load the raw CSV, clean and sort it, persist as parquet.
  2. Print global statistics (sanity numbers the user will verify against the plan).
  3. Generate three EDA figures under results/figures/.
  4. Report reciprocity and giant-component sizes on the whole-period static graph.
  5. Build ~63 monthly cumulative snapshots and persist them to data/interim/snapshots/.

Run:
    python -m scripts.run_stage1_eda
"""
from __future__ import annotations

from pprint import pprint

from src import cleaning, eda, snapshots


def main() -> None:
    print("[1/5] Cleaning raw CSV...")
    df = cleaning.run()
    print(f"       -> cleaned rows: {len(df):,}")

    print("[2/5] Global statistics:")
    pprint(eda.global_stats(df))

    print("[3/5] Writing EDA figures...")
    eda.plot_rating_histogram(df)
    eda.plot_edges_per_month(df)
    eda.plot_degree_distribution(df)
    print("       -> wrote rating_histogram.png, edges_per_month.png, "
          "degree_distributions.png")

    print("[4/5] Reciprocity and giant-component sizes:")
    pprint(eda.reciprocity_and_giant(df))

    print("[5/5] Building monthly cumulative snapshots...")
    summary = snapshots.build_all(df)
    print(f"       -> built {len(summary)} snapshots")
    print("       first:", summary[0])
    print("       last :", summary[-1])


if __name__ == "__main__":
    main()
