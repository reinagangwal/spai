"""
preprocess.py — SPENDLY Data Pipeline
======================================
Person 1 responsibility: load, clean, engineer features, encode, and split
the credit card transactions dataset so every other team member gets clean,
consistent CSV files to import.

Output files (written to data/processed/):
    train.csv  — earliest 70 % of rows (chronological)
    val.csv    — next 15 %
    test.csv   — last 15 %

Output columns (contract with teammates):
    city             : raw city string  (e.g. "Delhi, India")
    date             : ISO-8601 date string (YYYY-MM-DD)
    card_type        : raw card type string (Gold / Silver / Platinum / Signature)
    exp_type         : raw expense type — TARGET LABEL for classifier (Person 2)
    gender           : raw gender string (M / F)
    amount_raw       : original INR amount (float) — used by anomaly detector (Person 3)
    amount_norm      : min-max normalised amount in [0, 1] — used by LSTM (Person 4)
    day_of_week      : 0=Monday … 6=Sunday
    month            : 1–12
    week_number      : ISO week number 1–53
    year             : 4-digit year
    card_type_encoded: integer label-encoded card_type
    gender_encoded   : integer label-encoded gender
    city_encoded     : integer label-encoded city
    exp_type_encoded : integer label-encoded exp_type (matches exp_type 1-to-1)

Assumptions (flag to team if these change):
    - Date format in raw file is "DD-Mon-YY" e.g. "29-Oct-14"  (pandas dayfirst parse)
    - Amount is always positive (negatives are dropped as invalid)
    - No missing values in the source file (verified in EDA)
    - No duplicate rows in the source file (verified in EDA)
    - LabelEncoder fit on FULL dataset before splitting → encodings are
      consistent across train/val/test (important for Person 2's model)
    - Chronological split uses sorted date order, NOT the original row order,
      because the CSV is not sorted by date
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------------------------------------------------------------------
# Logging — teammates can see exactly what happened when they import the file
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path constants — change only these if the folder layout ever shifts
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # spendly/
RAW_FILE = ROOT_DIR / "data" / "raw" / "Credit card transactions - India - Simple.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Split ratios — must sum to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  (implicit)


# ===========================================================================
# STEP 1 — LOAD
# ===========================================================================

def load_raw(filepath: Path = RAW_FILE) -> pd.DataFrame:
    """
    Read the raw Kaggle CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original column names preserved.
        Columns: index, City, Date, Card Type, Exp Type, Gender, Amount
    """
    log.info("Loading raw data from %s", filepath)
    df = pd.read_csv(filepath)
    log.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ===========================================================================
# STEP 2 — CLEAN
# ===========================================================================

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove dirty rows and drop the redundant index column.

    Cleaning rules applied (in order):
        1. Drop the 'index' column — it is the original CSV row number,
           not meaningful for modelling.
        2. Drop duplicate rows (full-row exact duplicates).
        3. Drop rows where Amount <= 0 — a credit card charge of zero or
           negative has no financial meaning and would skew normalisation.
        4. Drop rows with any null in the six meaningful columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_raw().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, reset index.
    """
    initial_rows = len(df)

    # 1. Drop the original CSV index column if it exists
    if "index" in df.columns:
        df = df.drop(columns=["index"])
        log.info("Dropped 'index' column.")

    # 2. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    log.info("Duplicate rows removed: %d", dropped)

    # 3. Remove non-positive amounts
    before = len(df)
    df = df[df["Amount"] > 0]
    dropped = before - len(df)
    log.info("Rows with Amount <= 0 removed: %d", dropped)

    # 4. Drop rows with nulls in any column we use
    before = len(df)
    df = df.dropna(subset=["City", "Date", "Card Type", "Exp Type", "Gender", "Amount"])
    dropped = before - len(df)
    log.info("Rows with missing values removed: %d", dropped)

    df = df.reset_index(drop=True)
    log.info("Clean dataset: %d rows (removed %d total)", len(df), initial_rows - len(df))
    return df


# ===========================================================================
# STEP 3 — PARSE DATES & EXTRACT TEMPORAL FEATURES
# ===========================================================================

def parse_dates_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Date column and derive time-based features used by LSTM (Person 4)
    and the classifier (Person 2).

    Raw date format: "DD-Mon-YY"  e.g. "29-Oct-14"
    pandas infers the two-digit year as 20XX (2000s) which is correct here
    because the dataset spans 2013-2015.

    New columns added:
        day_of_week  : int, 0=Monday … 6=Sunday  (sklearn-friendly)
        month        : int, 1–12
        week_number  : int, ISO week 1–53 (useful for seasonality)
        year         : int, 4-digit
        date         : string "YYYY-MM-DD" (replaces raw "Date")

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with date features added and 'Date' replaced by 'date'.
    """
    log.info("Parsing dates …")

    # Parse — dayfirst=True because format is DD-Mon-YY
    parsed = pd.to_datetime(df["Date"], dayfirst=True)

    df["date"]        = parsed.dt.strftime("%Y-%m-%d")   # ISO string for CSV storage
    df["day_of_week"] = parsed.dt.dayofweek.astype(int)  # 0=Mon, 6=Sun
    df["month"]       = parsed.dt.month.astype(int)
    df["week_number"] = parsed.dt.isocalendar().week.astype(int)
    df["year"]        = parsed.dt.year.astype(int)

    # Keep the _parsed_ datetime internally for sorting; drop it later
    df["_date_parsed"] = parsed

    # Drop the original raw 'Date' column
    df = df.drop(columns=["Date"])

    log.info(
        "Date range: %s → %s",
        parsed.min().date(),
        parsed.max().date(),
    )
    return df


# ===========================================================================
# STEP 4 — RENAME COLUMNS (align with output contract)
# ===========================================================================

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw columns to the snake_case names agreed with all teammates.

    Mapping:
        City      → city
        Card Type → card_type
        Exp Type  → exp_type
        Gender    → gender
        Amount    → amount_raw   (raw value kept for anomaly detector, Person 3)
    """
    rename_map = {
        "City":      "city",
        "Card Type": "card_type",
        "Exp Type":  "exp_type",
        "Gender":    "gender",
        "Amount":    "amount_raw",
    }
    df = df.rename(columns=rename_map)
    log.info("Columns renamed.")
    return df


# ===========================================================================
# STEP 5 — NORMALISE AMOUNT
# ===========================================================================

def normalise_amount(df: pd.DataFrame, scaler: MinMaxScaler = None):
    """
    Apply min-max normalisation to amount_raw → amount_norm in [0, 1].

    WHY min-max and not z-score?
    The LSTM (Person 4) works well with bounded inputs.  Z-score normalisation
    produces unbounded outputs for outliers, which can destabilise gradients.
    Min-max keeps everything in [0, 1].

    IMPORTANT: the scaler is FIT on train only and then APPLIED to val/test.
    This function accepts an optional pre-fit scaler so the same scale is
    used consistently across splits.  When scaler=None (first call on train),
    it creates and fits a new one.

    Parameters
    ----------
    df     : pd.DataFrame  with column 'amount_raw'
    scaler : fitted MinMaxScaler or None

    Returns
    -------
    (pd.DataFrame, MinMaxScaler)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[["amount_raw"]])
        log.info(
            "MinMaxScaler fit: min=%.2f, max=%.2f",
            scaler.data_min_[0],
            scaler.data_max_[0],
        )

    df["amount_norm"] = scaler.transform(df[["amount_raw"]]).flatten()
    return df, scaler


# ===========================================================================
# STEP 6 — LABEL-ENCODE CATEGORICAL COLUMNS
# ===========================================================================

def encode_categoricals(df: pd.DataFrame, encoders: dict = None):
    """
    Label-encode the four categorical columns.

    Encoders are FIT on the FULL dataset BEFORE splitting so that the integer
    mapping is identical across train, val, and test.  If Person 2 saves a
    model that expects city_encoded=7 to mean "Kolkata, India", that must be
    true in val and test too — fitting per-split would break this.

    Output columns added:
        card_type_encoded : int (Gold=0, Platinum=1, Silver=2, Signature=3 — alphabetical)
        gender_encoded    : int (F=0, M=1)
        city_encoded      : int (alphabetical order of city string)
        exp_type_encoded  : int (Bills=0, Entertainment=1, Food=2, Fuel=3, Grocery=4, Travel=5)

    Exact mapping depends on what values appear in the data.  Person 2 should
    call  encoder['exp_type'].classes_  to see the mapping.

    Parameters
    ----------
    df       : pd.DataFrame
    encoders : dict of {column_name: fitted LabelEncoder} or None

    Returns
    -------
    (pd.DataFrame, dict of fitted LabelEncoders)
    """
    cols = {
        "card_type": "card_type_encoded",
        "gender":    "gender_encoded",
        "city":      "city_encoded",
        "exp_type":  "exp_type_encoded",
    }

    if encoders is None:
        encoders = {}
        for src_col in cols:
            le = LabelEncoder()
            le.fit(df[src_col])
            encoders[src_col] = le
            log.info(
                "LabelEncoder for '%s': %d classes → %s",
                src_col, len(le.classes_), list(le.classes_),
            )

    for src_col, tgt_col in cols.items():
        df[tgt_col] = encoders[src_col].transform(df[src_col])

    return df, encoders


# ===========================================================================
# STEP 7 — CHRONOLOGICAL SPLIT
# ===========================================================================

def chronological_split(df: pd.DataFrame):
    """
    Sort by date ascending and split into train / val / test with NO shuffling.

    WHY no shuffle?
    This is time-series data.  Shuffling would leak future information into
    the training set (data leakage), making model performance metrics
    unrealistically optimistic.

    Split sizes (based on 26,052 rows after cleaning):
        train : first 70 % ≈ 18,236 rows  (oldest transactions)
        val   : next  15 % ≈  3,908 rows
        test  : last  15 % ≈  3,908 rows  (newest transactions)

    Parameters
    ----------
    df : pd.DataFrame  must contain '_date_parsed' column (added in step 3)

    Returns
    -------
    (train_df, val_df, test_df) — all as pd.DataFrame, index reset
    """
    df = df.sort_values("_date_parsed").reset_index(drop=True)

    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    log.info(
        "Split sizes → train: %d  val: %d  test: %d  (total: %d)",
        len(train), len(val), len(test), n,
    )
    log.info(
        "Train date range  : %s → %s",
        train["_date_parsed"].min().date(),
        train["_date_parsed"].max().date(),
    )
    log.info(
        "Val   date range  : %s → %s",
        val["_date_parsed"].min().date(),
        val["_date_parsed"].max().date(),
    )
    log.info(
        "Test  date range  : %s → %s",
        test["_date_parsed"].min().date(),
        test["_date_parsed"].max().date(),
    )
    return train, val, test


# ===========================================================================
# STEP 8 — FINALISE COLUMNS & SAVE
# ===========================================================================

# Exact column order that every other team member must see in their CSV
OUTPUT_COLUMNS = [
    "city", "date", "card_type", "exp_type", "gender",
    "amount_raw", "amount_norm",
    "day_of_week", "month", "week_number", "year",
    "card_type_encoded", "gender_encoded", "city_encoded", "exp_type_encoded",
]


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Drop the internal helper column, reorder to the agreed schema,
    and write CSVs to data/processed/.

    Parameters
    ----------
    train, val, test : pd.DataFrame — already processed splits
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        # Drop internal column used only for sorting
        split = split.drop(columns=["_date_parsed"], errors="ignore")

        # Enforce exact column order for teammates
        split = split[OUTPUT_COLUMNS]

        out_path = PROCESSED_DIR / f"{name}.csv"
        split.to_csv(out_path, index=False)
        log.info("Saved %s → %s  (%d rows)", name, out_path, len(split))


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_pipeline() -> tuple:
    """
    Execute the full preprocessing pipeline end-to-end.

    Returns
    -------
    (train_df, val_df, test_df, encoders, scaler)
        Returned so teammates can import run_pipeline() and use the
        fitted encoders / scaler without re-reading the CSV.
    """
    log.info("=" * 60)
    log.info("SPENDLY  |  preprocess.py  |  starting pipeline")
    log.info("=" * 60)

    # --- Load ---
    df = load_raw()

    # --- Clean ---
    df = clean(df)

    # --- Parse dates & temporal features ---
    df = parse_dates_and_features(df)

    # --- Rename to agreed schema ---
    df = rename_columns(df)

    # --- Encode categoricals on FULL dataset (before split) ---
    df, encoders = encode_categoricals(df, encoders=None)

    # --- Chronological split ---
    train, val, test = chronological_split(df)

    # --- Normalise amount: fit ONLY on train, transform all splits ---
    train, scaler = normalise_amount(train, scaler=None)
    val,   _      = normalise_amount(val,   scaler=scaler)
    test,  _      = normalise_amount(test,  scaler=scaler)

    # --- Save ---
    save_splits(train, val, test)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)

    return train, val, test, encoders, scaler


# ---------------------------------------------------------------------------
# Run when executed directly:  python src/preprocess.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
