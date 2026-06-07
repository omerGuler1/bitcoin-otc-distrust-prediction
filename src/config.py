"""Central configuration: paths, thresholds, random seed.

Keeping every tunable in one place so downstream stages can `from .config import X`
instead of hard-coding values. Directories are created on import to keep scripts
simple (no os.makedirs at every call site).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "soc-sign-bitcoinotc.csv"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results" / "figures"

RANDOM_SEED = 42

# --- Stage 2 placeholders (unused in Stage 1 but defined here for forward-compat) ---
MIN_ACTIVITY = 3      # min total interactions (in+out) for a node to be "core"
RECENT_DAYS = 180     # must be active within last N days of snapshot end
KHOP = 2              # candidate pairs restricted to this graph distance
MAX_CANDIDATES = 200_000

# Split config. t_warmup is the earliest snapshot that can enter the training set.
T_WARMUP = 12
N_TEST_SNAPSHOTS = 2
N_VAL_SNAPSHOTS = 2

# Exclude snapshots from the split pool when the raw next-window negative
# volume is too low to support a reliable evaluation. Candidates and labels
# for excluded snapshots are still generated (so training could pool them if
# needed), but they are not used in the val/test fold selection.
MIN_POS_FOR_SPLIT = 10

# Stage 4: temporal feature knobs.
RECENT_WINDOW_DAYS = 90   # for "recent_*" counts
DECAY_TAU_DAYS = 60.0     # half-life-ish time constant for decayed_* features
DAYS_NEVER = 9999.0       # sentinel for "no prior activity"
# Training-set subsampling (decision A): keep all positives, sample negatives.
TRAIN_NEG_PER_POS = 100

for _d in [
    DATA_INTERIM,
    DATA_PROCESSED,
    FIG_DIR,
    DATA_INTERIM / "snapshots",
    DATA_PROCESSED / "labeled_pairs",
]:
    _d.mkdir(parents=True, exist_ok=True)
