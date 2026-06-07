# Community-Aware Early Warning of Distrust Emergence in Temporal Bitcoin Trust Networks

Undergraduate course project. Predicts the emergence of **new negative edges** in
the next time window on the Bitcoin-OTC signed trust graph using interpretable
structural, temporal, and community-aware features.

## Dataset
`data/raw/soc-sign-bitcoinotc.csv` — 35,592 rated interactions, 5,881 users,
Nov-2010 to Jan-2016. Columns (no header): `source, target, rating, timestamp`.

## Stage 1 quickstart
```bash
pip install -r requirements.txt
python -m scripts.run_stage1_eda
```
Produces `data/interim/edges_clean.parquet`, ~63 monthly snapshot pickles under
`data/interim/snapshots/`, and EDA figures under `results/figures/`.

See [the approved plan](../../.claude/plans/indexed-noodling-stonebraker.md)
for the full stage-by-stage roadmap.
