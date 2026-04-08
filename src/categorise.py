import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "card_type_encoded",
    "gender_encoded",
    "city_encoded",
    "exp_type_encoded",  # note: kept out of features below (target), included here for validation only
    "amount_norm",
    "day_of_week",
    "month",
    "week_number",
]

RAW_TEXT_COLS = ["city", "date", "card_type", "exp_type", "gender", "amount_raw"]


def _load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "val.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    missing = [p for p in [train_path, val_path, test_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required processed CSVs: {missing}")

    return _load_split(train_path), _load_split(val_path), _load_split(test_path)


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Target = exp_type_encoded (as per spec)
    if "exp_type_encoded" not in df.columns:
        raise ValueError("Column `exp_type_encoded` missing; cannot train classifier.")

    y = df["exp_type_encoded"].astype(int)

    # Features = encoded columns + amount_norm, day_of_week, month, week_number
    feature_cols = [
        "card_type_encoded",
        "gender_encoded",
        "city_encoded",
        "amount_norm",
        "day_of_week",
        "month",
        "week_number",
    ]

    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    return X, y


def train_and_evaluate() -> None:
    train_df, val_df, _test_df = load_data()

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    report = classification_report(y_val, y_pred)

    os.makedirs("outputs", exist_ok=True)

    # Save evaluation summary
    with open(os.path.join("outputs", "rf_val_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"weighted_f1: {f1:.4f}\n\n")
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
    plt.title("Random Forest Confusion Matrix (val)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "rf_confusion_matrix.png"), dpi=200)
    plt.close()

    # Feature importance plot (top 10)
    importances = model.feature_importances_
    feats = np.array(X_train.columns)
    order = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(9, 5))
    sns.barplot(x=importances[order], y=feats[order], orient="h")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "rf_feature_importance.png"), dpi=200)
    plt.close()

    print(f"RandomForest trained | val accuracy={acc:.4f} weighted_f1={f1:.4f}")
    print("Saved outputs/rf_confusion_matrix.png and outputs/rf_feature_importance.png")


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()

