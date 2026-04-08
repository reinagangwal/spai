import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_data() -> pd.DataFrame:
    """
    Load all processed splits (train/val/test) if present.
    Produces a unified dataframe with a year-month `month` key.
    """
    processed_dir = os.path.join("data", "processed")
    candidates: list[str] = []
    if os.path.isdir(processed_dir):
        for name in os.listdir(processed_dir):
            if name.lower().endswith(".csv"):
                candidates.append(os.path.join(processed_dir, name))

    if not candidates:
        raise FileNotFoundError("No CSVs found under data/processed/.")

    frames = [pd.read_csv(p) for p in sorted(candidates)]
    df = pd.concat(frames, ignore_index=True)

    # Ensure month is a proper year-month key for time ordering.
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = dt.dt.to_period("M").astype(str)
    elif {"year", "month"}.issubset(df.columns):
        df["month"] = (
            pd.to_datetime(
                df["year"].astype(str)
                + "-"
                + df["month"].astype(int).astype(str).str.zfill(2)
                + "-01",
                errors="coerce",
            )
            .dt.to_period("M")
            .astype(str)
        )
    else:
        raise ValueError("Expected columns: either `date` or (`year` and `month`).")

    if "exp_type" not in df.columns or "amount_raw" not in df.columns:
        raise ValueError("Expected columns `exp_type` and `amount_raw` in processed data.")

    df["amount_raw"] = pd.to_numeric(df["amount_raw"], errors="coerce")
    df = df.dropna(subset=["month", "exp_type", "amount_raw"])
    return df


def identify_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating by month and exp_type...")
    monthly_spend = df.groupby(["month", "exp_type"])["amount_raw"].sum().reset_index()
    monthly_spend = monthly_spend.sort_values(by=["exp_type", "month"])

    # Use prior months only (shifted) so the current month is compared to history.
    def _rolling_mean_prev(x: pd.Series) -> pd.Series:
        return x.rolling(window=3, min_periods=2).mean().shift(1)

    def _rolling_std_prev(x: pd.Series) -> pd.Series:
        return x.rolling(window=3, min_periods=2).std(ddof=0).shift(1)

    monthly_spend["rolling_mean"] = monthly_spend.groupby("exp_type")["amount_raw"].transform(
        _rolling_mean_prev
    )
    monthly_spend["rolling_std"] = monthly_spend.groupby("exp_type")["amount_raw"].transform(
        _rolling_std_prev
    )

    monthly_spend["z_score"] = np.nan
    valid = (
        monthly_spend["rolling_std"].notna()
        & (monthly_spend["rolling_std"] > 0)
        & monthly_spend["rolling_mean"].notna()
    )
    monthly_spend.loc[valid, "z_score"] = (
        (monthly_spend.loc[valid, "amount_raw"] - monthly_spend.loc[valid, "rolling_mean"])
        / monthly_spend.loc[valid, "rolling_std"]
    )

    # Optional IQR method (secondary check)
    q1 = monthly_spend.groupby("exp_type")["amount_raw"].transform(lambda x: x.quantile(0.25))
    q3 = monthly_spend.groupby("exp_type")["amount_raw"].transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    monthly_spend["is_outlier_iqr"] = (monthly_spend["amount_raw"] > (q3 + 1.5 * iqr)) | (
        monthly_spend["amount_raw"] < (q1 - 1.5 * iqr)
    )

    anomalies_z = monthly_spend[np.abs(monthly_spend["z_score"]) > 3].copy()
    anomalies_z["method"] = "zscore"

    anomalies_iqr = monthly_spend[monthly_spend["is_outlier_iqr"]].copy()
    anomalies_iqr["method"] = "iqr"

    anomalies = pd.concat([anomalies_z, anomalies_iqr]).drop_duplicates(subset=["month", "exp_type"])

    def get_explanation(row: pd.Series) -> str:
        mean = row.get("rolling_mean", np.nan)
        if pd.isna(mean) or mean == 0:
            return f"Flagged as unusual monthly total for {row['exp_type']} (insufficient history)"
        ratio = float(row["amount_raw"]) / float(mean)
        if ratio >= 1:
            return f"{ratio:.1f}x above monthly average for {row['exp_type']}"
        pct_below = (1 - ratio) * 100
        return f"{pct_below:.0f}% below monthly average for {row['exp_type']}"

    anomalies["explanation"] = anomalies.apply(get_explanation, axis=1) if not anomalies.empty else ""

    os.makedirs("outputs", exist_ok=True)
    final_anomalies = anomalies[["month", "exp_type", "amount_raw", "z_score", "explanation", "method"]]
    final_anomalies.to_csv(os.path.join("outputs", "anomaly_df.csv"), index=False)
    print(f"Exported outputs/anomaly_df.csv with {len(final_anomalies)} anomalies found.")

    return monthly_spend


def perform_clustering(df: pd.DataFrame) -> None:
    print("Building K-Means clustering (k=3)...")
    monthly_spend = df.groupby(["month", "exp_type"])["amount_raw"].sum().reset_index()
    pivot_df = monthly_spend.pivot_table(
        index="month", columns="exp_type", values="amount_raw", aggfunc="sum", fill_value=0
    )

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    pivot_df["cluster_id"] = kmeans.fit_predict(scaled_data)

    cluster_means = pivot_df.drop(columns=["cluster_id"]).groupby(pivot_df["cluster_id"]).mean()
    overall_mean_total = pivot_df.drop(columns=["cluster_id"]).sum(axis=1).mean()

    labels: dict[int, str] = {}
    for cid in sorted(cluster_means.index.tolist()):
        row = cluster_means.loc[cid]
        top_cat = row.idxmax()
        total = row.sum()
        if overall_mean_total > 0 and total < 0.6 * overall_mean_total:
            labels[int(cid)] = "Low activity month"
        elif total > 1.4 * overall_mean_total:
            labels[int(cid)] = f"High spend month (especially {top_cat})"
        else:
            labels[int(cid)] = f"{top_cat}-heavy month"

    # Keep labels distinct for display
    seen: set[str] = set()
    for cid in list(labels.keys()):
        base = labels[cid]
        if base in seen:
            labels[cid] = f"{base} (cluster {cid})"
        seen.add(labels[cid])

    pivot_df["cluster_label"] = pivot_df["cluster_id"].map(labels)
    cluster_df = pivot_df[["cluster_id", "cluster_label"]].reset_index()

    os.makedirs("outputs", exist_ok=True)
    cluster_df.to_csv(os.path.join("outputs", "cluster_labels.csv"), index=False)
    print("Exported outputs/cluster_labels.csv.")


def main() -> None:
    df = load_data()
    identify_anomalies(df)
    perform_clustering(df)
    print("P3 tasks completely finished.")


if __name__ == "__main__":
    main()

