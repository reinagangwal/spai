"""
sanity_check.py — Verify preprocess.py output is correct
=========================================================
Run from the spendly/ root:  python sanity_check.py

Prints PASS / FAIL for each check so you can hand this to your teammates
as proof that the preprocessing contract is met.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED = Path("data/processed")
REQUIRED_COLUMNS = [
    "city", "date", "card_type", "exp_type", "gender",
    "amount_raw", "amount_norm",
    "day_of_week", "month", "week_number", "year",
    "card_type_encoded", "gender_encoded", "city_encoded", "exp_type_encoded",
]

RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

passed = 0
failed = 0

def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {GREEN}PASS{RESET}  {label}")
        passed += 1
    else:
        print(f"  {RED}FAIL{RESET}  {label}" + (f"  → {detail}" if detail else ""))
        failed += 1

print("\n" + "="*60)
print("SPENDLY  |  Preprocessing Sanity Checks")
print("="*60 + "\n")

# --- Load splits ---
try:
    train = pd.read_csv(PROCESSED / "train.csv")
    val   = pd.read_csv(PROCESSED / "val.csv")
    test  = pd.read_csv(PROCESSED / "test.csv")
except FileNotFoundError as e:
    print(f"{RED}Cannot load files: {e}{RESET}")
    print("Run  python src/preprocess.py  first.")
    sys.exit(1)

# 1. Row counts
total = len(train) + len(val) + len(test)
print(f"[1] Row counts  (total={total:,})")
check("train is ~70%", 0.68 <= len(train)/total <= 0.72,
      f"actual {len(train)/total:.2%}")
check("val   is ~15%", 0.13 <= len(val)/total <= 0.17,
      f"actual {len(val)/total:.2%}")
check("test  is ~15%", 0.13 <= len(test)/total <= 0.17,
      f"actual {len(test)/total:.2%}")

# 2. Column names
print(f"\n[2] Column names")
for split_name, split in [("train", train), ("val", val), ("test", test)]:
    missing = set(REQUIRED_COLUMNS) - set(split.columns)
    extra   = set(split.columns) - set(REQUIRED_COLUMNS)
    check(f"{split_name} has all required columns",
          len(missing) == 0, f"missing: {missing}")
    check(f"{split_name} has no extra columns",
          len(extra) == 0, f"extra: {extra}")

# 3. Column order
print(f"\n[3] Column order")
check("train column order matches contract",
      list(train.columns) == REQUIRED_COLUMNS,
      f"got {list(train.columns)}")

# 4. No nulls
print(f"\n[4] No nulls")
for split_name, split in [("train", train), ("val", val), ("test", test)]:
    nulls = split[REQUIRED_COLUMNS].isnull().sum().sum()
    check(f"{split_name} has zero nulls", nulls == 0, f"{nulls} nulls found")

# 5. Chronological order — no date overlap between splits
print(f"\n[5] Chronological split (no data leakage)")
train_max = pd.to_datetime(train["date"]).max()
val_min   = pd.to_datetime(val["date"]).min()
val_max   = pd.to_datetime(val["date"]).max()
test_min  = pd.to_datetime(test["date"]).min()

check("train ends before or at val start",
      train_max <= val_min, f"train_max={train_max.date()}, val_min={val_min.date()}")
check("val ends before or at test start",
      val_max <= test_min, f"val_max={val_max.date()}, test_min={test_min.date()}")

# 6. amount_norm range
print(f"\n[6] amount_norm is in [0, 1]")
for split_name, split in [("train", train), ("val", val), ("test", test)]:
    lo, hi = split["amount_norm"].min(), split["amount_norm"].max()
    check(f"{split_name} amount_norm in [0,1]",
          lo >= 0.0 and hi <= 1.0, f"min={lo:.4f}, max={hi:.4f}")
check("train amount_norm reaches ~0 and ~1 (scaler fit correctly)",
      train["amount_norm"].min() < 0.001 and train["amount_norm"].max() > 0.999)

# 7. amount_raw always positive
print(f"\n[7] amount_raw > 0")
for split_name, split in [("train", train), ("val", val), ("test", test)]:
    bad = (split["amount_raw"] <= 0).sum()
    check(f"{split_name} amount_raw > 0", bad == 0, f"{bad} invalid rows")

# 8. Encoded columns are non-negative integers
print(f"\n[8] Encoded columns are non-negative integers")
enc_cols = ["card_type_encoded", "gender_encoded", "city_encoded", "exp_type_encoded"]
for col in enc_cols:
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        ok = (split[col] >= 0).all() and split[col].dtype in [np.int32, np.int64, int]
        check(f"{split_name}.{col} is non-neg int", ok,
              f"dtype={split[col].dtype}, min={split[col].min()}")

# 9. exp_type ↔ exp_type_encoded mapping is consistent across splits
print(f"\n[9] exp_type ↔ exp_type_encoded consistency")
all_data = pd.concat([train, val, test])
mapping = all_data.groupby("exp_type_encoded")["exp_type"].nunique()
check("each encoded value maps to exactly one exp_type",
      (mapping == 1).all(), f"inconsistent mappings: {mapping[mapping>1]}")

# 10. Temporal feature ranges
print(f"\n[10] Temporal feature sanity")
check("day_of_week in [0,6]",
      train["day_of_week"].between(0, 6).all())
check("month in [1,12]",
      train["month"].between(1, 12).all())
check("week_number in [1,53]",
      train["week_number"].between(1, 53).all())
check("year values are 4-digit",
      train["year"].between(2000, 2100).all())

# --- Summary ---
print("\n" + "="*60)
print(f"Result: {GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}")
print("="*60 + "\n")

if failed > 0:
    sys.exit(1)
