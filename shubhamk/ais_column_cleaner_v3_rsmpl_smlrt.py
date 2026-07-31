#!/usr/bin/env python3
"""
AIS Column Cleaner v3
=====================
- Exact match cleaning (UN/LOCODE based)
- Similarity score column added
- Optional resampling (for faster testing / review)
- Works on any Parquet file + any column
"""

import re
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION  ←←←  CHANGE THESE
# ============================================================
FILE_PATH       = "data_inst-dd_mp_etl02.parquet"
COLUMN_NAME     = "destination"
MAPPING_FILE    = "destination_mapping.csv"

# Resampling (set to None to process full data)
SAMPLE_SIZE     = None          # e.g. 50000  or  None for full data
RANDOM_SEED     = 42

TOP_N           = 40
SAVE_CLEANED    = True
OUTPUT_DIR      = Path("./output_rsmpl_smlrt")
# ============================================================


# -------------------- Helper functions --------------------
def basic_clean(text) -> str | None:
    if pd.isna(text) or text is None:
        return None
    text = str(text).upper().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Z0-9\s/\->]", "", text)
    text = text.strip()
    return text if text else None


def load_mapping(mapping_path: str) -> dict:
    path = Path(mapping_path)
    if not path.exists():
        print(f"WARNING: Mapping file not found → {path}")
        return {}
    mdf = pd.read_csv(path)
    mapping = dict(zip(
        mdf["dirty"].astype(str).str.upper().str.strip(),
        mdf["clean"].astype(str).str.upper().str.strip()
    ))
    print(f"Loaded {len(mapping)} verified mappings")
    return mapping


def similarity_score(a: str, b: str) -> float:
    """
    Simple but effective similarity (0–100).
    Uses SequenceMatcher (built-in, no extra library needed).
    """
    if a is None or b is None:
        return 0.0
    from difflib import SequenceMatcher
    return round(SequenceMatcher(None, a, b).ratio() * 100, 1)


def clean_and_score(value, mapping: dict):
    """
    Returns: (cleaned_value, similarity_score)
    """
    cleaned = basic_clean(value)
    if cleaned is None:
        return None, 0.0

    # Exact match first
    if cleaned in mapping:
        final = mapping[cleaned]
        score = 100.0 if final == cleaned else similarity_score(cleaned, final)
        return final, score

    # No exact match → keep cleaned original, score = 100 (it is already clean)
    return cleaned, 100.0


# -------------------- Main pipeline --------------------
def run_pipeline():
    file_path = Path(FILE_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AIS COLUMN CLEANER v3 (Exact Match + Similarity Score + Resampling)")
    print(f"File          : {file_path}")
    print(f"Column        : {COLUMN_NAME}")
    print(f"Sample size   : {SAMPLE_SIZE if SAMPLE_SIZE else 'FULL DATA'}")
    print("=" * 70)

    # Load mapping
    mapping = load_mapping(MAPPING_FILE)

    # Check column
    pf = pq.ParquetFile(file_path)
    if COLUMN_NAME not in pf.schema_arrow.names:
        raise ValueError(f"Column '{COLUMN_NAME}' not found.\nAvailable: {pf.schema_arrow.names}")

    # Read data
    extra = [c for c in ["mmsi", "vessel_name", "imo_number"] if c in pf.schema_arrow.names]
    cols = [COLUMN_NAME] + extra
    print(f"\nReading columns: {cols}")
    df = pq.read_table(file_path, columns=cols).to_pandas()
    print(f"Original rows: {len(df):,}")

    # ---------- RESAMPLING ----------
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Resampled to  : {len(df):,} rows")
    else:
        print("Using full dataset (no resampling)")

    # Stats before
    original = df[COLUMN_NAME]
    null_before = original.isna().sum() + (original == "").sum()
    unique_before = original.nunique(dropna=True)
    print(f"\nBEFORE → Nulls: {null_before:,} | Unique: {unique_before:,}")

    # ---------- CLEANING + SIMILARITY SCORE ----------
    print("Cleaning + calculating similarity scores ...")
    results = original.apply(lambda x: clean_and_score(x, mapping))
    df[f"{COLUMN_NAME}_clean"] = results.apply(lambda x: x[0])
    df["similarity_score"] = results.apply(lambda x: x[1])

    # Stats after
    cleaned = df[f"{COLUMN_NAME}_clean"]
    null_after = cleaned.isna().sum()
    unique_after = cleaned.nunique(dropna=True)
    print(f"AFTER  → Nulls: {null_after:,} | Unique: {unique_after:,}")
    print(f"Reduced unique values by: {unique_before - unique_after:,}")

    # Similarity score summary
    print("\nSimilarity Score Summary:")
    print(df["similarity_score"].describe().round(1))

    # Frequency table
    freq = (cleaned.value_counts(dropna=False)
                  .rename_axis(COLUMN_NAME)
                  .reset_index(name="count"))
    freq["pct"] = (freq["count"] / len(df) * 100).round(2)

    print(f"\nTop {TOP_N} values after cleaning:")
    print(freq.head(TOP_N).to_string(index=False))

    # ---------- SAVE ----------
    if SAVE_CLEANED:
        out_parquet = output_dir / f"{file_path.stem}_cleaned_v3.parquet"
        df.to_parquet(out_parquet, index=False)
        print(f"\nCleaned Parquet      → {out_parquet}")

        out_csv = output_dir / f"{file_path.stem}_{COLUMN_NAME}_frequency.csv"
        freq.to_csv(out_csv, index=False)
        print(f"Frequency CSV        → {out_csv}")

        # Also save a sample of low similarity scores for review
        low_sim = df[df["similarity_score"] < 100][[COLUMN_NAME, f"{COLUMN_NAME}_clean", "similarity_score"]]
        if len(low_sim) > 0:
            low_file = output_dir / f"{file_path.stem}_low_similarity_review.csv"
            low_sim.to_csv(low_file, index=False)
            print(f"Low similarity rows  → {low_file} ({len(low_sim):,} rows)")

    # Chart
    plot_df = freq.head(TOP_N)
    plt.figure(figsize=(12, max(6, TOP_N * 0.35)))
    sns.barplot(data=plot_df, y=COLUMN_NAME, x="count", palette="viridis", orient="h")
    plt.title(f"Top {TOP_N} values in '{COLUMN_NAME}' (v3)")
    plt.tight_layout()
    chart_path = output_dir / f"{file_path.stem}_{COLUMN_NAME}_top{TOP_N}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart                → {chart_path}")

    print("\nDone.")
    return df, freq


if __name__ == "__main__":
    run_pipeline()