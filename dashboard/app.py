import os

import pandas as pd
import streamlit as st


def _read_csv_if_exists(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


st.set_page_config(page_title="SPENDLY Dashboard", layout="wide")

st.title("SPENDLY — Dashboard")

with st.sidebar:
    st.header("Navigation")
    st.caption("Person 2 + Person 3 outputs viewer")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    st.code(repo_root)

tab_overview, tab_cls, tab_anom, tab_forecast = st.tabs(
    ["Overview", "Classification", "Anomalies", "Forecast"]
)


with tab_overview:
    st.subheader("Overview")
    st.write(
        "This dashboard consumes artifacts produced by the team pipelines: "
        "classification outputs (Person 2), anomaly detection + clustering (Person 3), and forecasting (Person 4)."
    )

    anomaly_path = os.path.join(repo_root, "outputs", "anomaly_df.csv")
    cluster_path = os.path.join(repo_root, "outputs", "cluster_labels.csv")

    st.markdown("- **Artifacts status**")
    st.write(
        {
            "outputs/anomaly_df.csv": os.path.exists(anomaly_path),
            "outputs/cluster_labels.csv": os.path.exists(cluster_path),
        }
    )


with tab_cls:
    st.subheader("Classification (Random Forest)")

    report_path = os.path.join(repo_root, "outputs", "rf_val_report.txt")
    cm_path = os.path.join(repo_root, "outputs", "rf_confusion_matrix.png")
    fi_path = os.path.join(repo_root, "outputs", "rf_feature_importance.png")

    if os.path.exists(report_path):
        st.markdown("**Validation report**")
        st.code(open(report_path, "r", encoding="utf-8").read())
    else:
        st.info("Run `py src/categorise.py` to generate the validation report + plots.")

    cols = st.columns(2)
    with cols[0]:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion matrix (val)", use_container_width=True)
        else:
            st.warning("Missing `outputs/rf_confusion_matrix.png`")

    with cols[1]:
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Top feature importances", use_container_width=True)
        else:
            st.warning("Missing `outputs/rf_feature_importance.png`")


with tab_anom:
    st.subheader("Anomalies (z-score / IQR) + Monthly Clusters")

    anomaly_df = _read_csv_if_exists(os.path.join(repo_root, "outputs", "anomaly_df.csv"))
    cluster_df = _read_csv_if_exists(os.path.join(repo_root, "outputs", "cluster_labels.csv"))

    if anomaly_df is None:
        st.info("Run `py src/patterns.py` to generate `outputs/anomaly_df.csv`.")
    else:
        st.markdown("**Flagged anomalies**")
        st.dataframe(anomaly_df, use_container_width=True)

    if cluster_df is None:
        st.info("Run `py src/patterns.py` to generate `outputs/cluster_labels.csv`.")
    else:
        st.markdown("**Monthly cluster labels**")
        st.dataframe(cluster_df, use_container_width=True)


with tab_forecast:
    st.subheader("Forecast (Person 4)")
    st.info("Forecasting outputs will appear here once Person 4’s model and CSVs are pushed.")

