"""Stage 2 entry point.

For each snapshot t in [0, T-2] (we need t+1 to exist for labels):
  1. Compute core nodes.
  2. Generate 2-hop candidate ordered pairs, excluding existing negative u->v.
  3. Look up which of those pairs received a negative directed rating in the
     next-month window.
  4. Persist the per-snapshot (candidates, labels) table to parquet.

Produces:
  data/processed/labeled_pairs/snap_t=NNN.parquet
  results/stage2_summary.csv     (one row per eligible snapshot)
  prints: summary table, train/val/test split with positive-label counts,
          sanity-check results.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src import candidates as cand_mod
from src import config, core as core_mod, io_utils, labels as lbl_mod

PAIRS_DIR = config.DATA_PROCESSED / "labeled_pairs"


def pairs_path(t: int) -> Path:
    return PAIRS_DIR / f"snap_t={t:03d}.parquet"


def build_one_snapshot(
    df: pd.DataFrame, t: int, end_ts_next: pd.Timestamp
) -> dict:
    snap = io_utils.load_snapshot(t)
    end_ts_t = snap["end_ts"]

    core = core_mod.core_nodes(snap)
    candidates = cand_mod.build_candidates(snap, core)
    next_negs = lbl_mod.next_window_negatives(df, end_ts_t, end_ts_next)
    labels = lbl_mod.label_candidates(candidates, next_negs)

    table = pd.DataFrame(
        {
            "u": [u for u, _ in candidates],
            "v": [v for _, v in candidates],
            "y": labels,
            "snap_t": t,
        }
    )
    table.to_parquet(pairs_path(t), index=False)

    # Diagnostic counters — explain where positives are lost at each stage.
    raw_newneg = len(next_negs)
    core_retained = sum(1 for u, v in next_negs if u in core and v in core)
    candidate_retained = int(sum(labels))  # == positives found in candidates

    return {
        "t": t,
        "end_ts_t": end_ts_t,
        "end_ts_next": end_ts_next,
        "raw_node_count": snap["n_nodes_cum"],
        "core_node_count": len(core),
        "candidate_count": len(candidates),
        "positive_labels": candidate_retained,
        "negative_labels": len(labels) - candidate_retained,
        "positive_rate": (candidate_retained / len(labels)) if labels else 0.0,
        "raw_newneg_next_window": raw_newneg,
        "core_retained_pos": core_retained,
        "candidate_retained_pos": candidate_retained,
    }


def main() -> None:
    df = io_utils.load_clean_edges()
    # Reconstruct the month-end schedule to find end_ts_next for each t.
    # (We could also re-read them from snapshot pickles; cheaper to rebuild.)
    from src.snapshots import month_end_boundaries
    ends = list(month_end_boundaries(df))
    T = len(ends)
    print(f"Dataset spans {T} monthly snapshots (t=0..{T - 1}). "
          f"Eligible for labeling: t=0..{T - 2}.")

    rows = []
    for t in tqdm(range(T - 1), desc="snapshots"):
        row = build_one_snapshot(df, t, ends[t + 1])
        rows.append(row)

    summary = pd.DataFrame(rows)

    # --- Split-pool eligibility --------------------------------------------
    # A snapshot is eligible to enter the train/val/test split iff:
    #   (i) t >= T_WARMUP (at least 1 year of history available), AND
    #   (ii) it has non-empty candidates, AND
    #   (iii) its raw next-window negative volume >= MIN_POS_FOR_SPLIT.
    # This cuts the dead tail (late 2015 / Jan 2016) where the dataset has
    # too few new negatives to support a meaningful evaluation.
    summary["eligible_for_split"] = (
        (summary["t"] >= config.T_WARMUP)
        & (summary["candidate_count"] > 0)
        & (summary["raw_newneg_next_window"] >= config.MIN_POS_FOR_SPLIT)
    )

    # --- Summary table ------------------------------------------------------
    print("\n=== Per-snapshot summary ===")
    display_cols = [
        "t", "raw_node_count", "core_node_count", "candidate_count",
        "raw_newneg_next_window", "core_retained_pos", "candidate_retained_pos",
        "positive_rate", "eligible_for_split",
    ]
    # (in_split_pool is added below and re-saved to CSV.)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary[display_cols].to_string(index=False))

    # --- Split pool = longest contiguous eligible block --------------------
    # The dataset has a dead tail (Sep-2015 onward) with a sporadic eligible
    # island at t=53. Using the *longest contiguous run* of eligible snapshots
    # keeps train/val/test temporally adjacent and avoids a gap in test.
    def _longest_contiguous_run(ts: list[int]) -> list[int]:
        if not ts:
            return []
        ts = sorted(ts)
        best_start, best_end = 0, 0
        cur_start = 0
        for i in range(1, len(ts)):
            if ts[i] != ts[i - 1] + 1:
                if i - 1 - cur_start > best_end - best_start:
                    best_start, best_end = cur_start, i - 1
                cur_start = i
        if len(ts) - 1 - cur_start > best_end - best_start:
            best_start, best_end = cur_start, len(ts) - 1
        return ts[best_start : best_end + 1]

    all_eligible_ts = list(summary.loc[summary["eligible_for_split"], "t"])
    contiguous_ts = _longest_contiguous_run(all_eligible_ts)
    summary["in_split_pool"] = summary["t"].isin(contiguous_ts)
    print(f"\nAll eligible snapshots ({len(all_eligible_ts)}): {all_eligible_ts}")
    print(f"Longest contiguous block ({len(contiguous_ts)}): "
          f"t={contiguous_ts[0]}..{contiguous_ts[-1]}")

    eligible = summary[summary["in_split_pool"]].copy().reset_index(drop=True)

    # Persist the full summary (with both booleans) once everything is set.
    summary_path = config.PROJECT_ROOT / "results" / "stage2_summary.csv"
    summary.to_csv(summary_path, index=False)

    if len(eligible) < (config.N_TEST_SNAPSHOTS + config.N_VAL_SNAPSHOTS + 1):
        raise RuntimeError("Not enough eligible snapshots for the requested split.")

    test = eligible.tail(config.N_TEST_SNAPSHOTS)
    val = eligible.iloc[-(config.N_TEST_SNAPSHOTS + config.N_VAL_SNAPSHOTS):-config.N_TEST_SNAPSHOTS]
    train = eligible.iloc[:-(config.N_TEST_SNAPSHOTS + config.N_VAL_SNAPSHOTS)]

    def _fmt(name: str, block: pd.DataFrame) -> str:
        ts = list(block["t"])
        p = int(block["positive_labels"].sum())
        n = int(block["negative_labels"].sum())
        tot = p + n
        rate = (p / tot) if tot else 0.0
        return (f"{name:>5s}: snapshots={ts}  "
                f"pairs={tot:>7d}  pos={p:>5d}  neg={n:>7d}  pos_rate={rate:.4%}")

    print("\n=== Temporal split (positive-label counts) ===")
    print(_fmt("train", train))
    print(_fmt("val",   val))
    print(_fmt("test",  test))

    # --- Sanity checks ------------------------------------------------------
    print("\n=== Sanity checks ===")

    # 1. No self-loops in any labeled table.
    no_self_loops = True
    for t in summary["t"]:
        tbl = pd.read_parquet(pairs_path(int(t)))
        if (tbl["u"] == tbl["v"]).any():
            no_self_loops = False
            print(f"  [FAIL] self-loops present at t={t}")
            break
    print(f"  [{'OK' if no_self_loops else 'FAIL'}] no self-loops in candidates")

    # 2. No temporal leakage: a positive-labeled pair at snapshot t must have
    #    at least one negative event in (end_ts_t, end_ts_next], and candidates
    #    must not have had an existing negative u->v at end_ts_t.
    df_sorted = df.sort_values("ts")
    leakage_ok = True
    checked_t = summary["t"].iloc[-1]  # check the last eligible snapshot (cheap)
    end_ts_t = summary.loc[summary["t"] == checked_t, "end_ts_t"].iloc[0]
    end_ts_next = summary.loc[summary["t"] == checked_t, "end_ts_next"].iloc[0]
    tbl = pd.read_parquet(pairs_path(int(checked_t)))
    pos_tbl = tbl[tbl["y"] == 1]
    if not pos_tbl.empty:
        win_neg = df_sorted[
            (df_sorted["ts"] > end_ts_t)
            & (df_sorted["ts"] <= end_ts_next)
            & (df_sorted["sign"] == -1)
        ]
        win_pairs = set(map(tuple, win_neg[["source", "target"]].itertuples(index=False, name=None)))
        pos_pairs = set(map(tuple, pos_tbl[["u", "v"]].itertuples(index=False, name=None)))
        if not pos_pairs.issubset(win_pairs):
            leakage_ok = False
    # Also confirm no positive-labeled pair had a rating event AT or BEFORE end_ts_t
    # that is inconsistent with the label window.
    snap = io_utils.load_snapshot(int(checked_t))
    existing_neg = cand_mod.existing_negative_directed_edges(snap["G_dir"])
    cand_pairs = set(map(tuple, tbl[["u", "v"]].itertuples(index=False, name=None)))
    if cand_pairs & existing_neg:
        leakage_ok = False
    print(f"  [{'OK' if leakage_ok else 'FAIL'}] no temporal leakage "
          f"(verified on snapshot t={int(checked_t)})")

    # 3. Directed labels: (u,v) and (v,u) treated separately.
    tbl_last = pd.read_parquet(pairs_path(int(summary["t"].iloc[-1])))
    reversed_pairs = set(map(tuple, tbl_last[["v", "u"]].itertuples(index=False, name=None)))
    forward_pairs = set(map(tuple, tbl_last[["u", "v"]].itertuples(index=False, name=None)))
    # It's fine if both (u,v) and (v,u) are candidates — they are *separate*
    # rows, each with its own label. What we check: duplicates of the same
    # ordered pair should not exist.
    assert len(tbl_last) == len(forward_pairs), "duplicate ordered pairs in candidates"
    print(f"  [OK] directed labels (no duplicate ordered pairs); "
          f"reverse-pair overlap = {len(forward_pairs & reversed_pairs)}")

    # 4. No train/val/test overlap by snapshot.
    s_train, s_val, s_test = set(train["t"]), set(val["t"]), set(test["t"])
    overlap = (s_train & s_val) | (s_train & s_test) | (s_val & s_test)
    print(f"  [{'OK' if not overlap else 'FAIL'}] no train/val/test snapshot overlap "
          f"(|train|={len(s_train)}, |val|={len(s_val)}, |test|={len(s_test)})")


if __name__ == "__main__":
    main()
