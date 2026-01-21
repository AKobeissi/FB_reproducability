import pandas as pd

# --- config ---
INPUT_CSV = "/u/kobeissa/Documents/thesis/experiments/FB_reproducability/results/metrics_agg2/table__by_question_reasoning.csv"
OUTPUT_CSV = "/u/kobeissa/Documents/thesis/experiments/FB_reproducability/results/metrics_agg2/question_type_filtered.csv"
TIMESTAMP_COL = "timestamp"   # <-- change to your column name

# The timestamps you want to keep
keep_ts_raw = [
    "2026-01-12 17:30:56",
    "2026-01-12 18:31:59",
    "2026-01-14 16:40:17",
    "2026-01-16 15:30:23",
    "2026-01-14 23:14:03",
    "2026-01-14 18:59:08",
    "2026-01-15 0:40:57",
    "2026-01-14 20:22:00",
    "2026-01-15 18:13:47",
    "2026-01-16 12:36:19",
    "2026-01-09 13:46:41",
    "2026-01-09 14:43:48",
    "2026-01-09 15:38:25",
    "2026-01-09 16:32:49",
    "2026-01-17 2:45:38",
    "2026-01-17 1:36:27",
    "2026-01-13 13:30:44",
]

# --- load ---
df = pd.read_csv(INPUT_CSV)

# Parse the CSV timestamp column into datetimes (coerce invalid to NaT)
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")

# Parse the keep-list into datetimes too (handles single-digit hours like "0:40:57")
keep_ts = pd.to_datetime(pd.Series(keep_ts_raw), errors="coerce")

# Drop any NaT from the keep list just in case
keep_set = set(keep_ts.dropna().tolist())

# --- filter ---
filtered = df[df[TIMESTAMP_COL].isin(keep_set)].copy()

# (Optional) sort by timestamp
filtered = filtered.sort_values(TIMESTAMP_COL)

# --- save ---
filtered.to_csv(OUTPUT_CSV, index=False)

print(f"Kept {len(filtered)} / {len(df)} rows -> {OUTPUT_CSV}")
