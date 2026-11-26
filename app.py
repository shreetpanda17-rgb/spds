import pandas as pd
import streamlit as st
from core.engine import SPDSEngine

st.set_page_config(page_title="SPDS", page_icon="📊", layout="wide")
st.title("SPDS — Strategic & Professional Data Science")
st.caption("Autonomous data analysis and decision-support workspace")


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import io

    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


@st.cache_resource(show_spinner=False)
def get_engine(df: pd.DataFrame) -> SPDSEngine:
    return SPDSEngine(df)


uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Upload a dataset to begin.")
    st.markdown(
        """
### Included
- Automatic profiling and data-quality checks
- Exploratory summaries and correlations
- IQR-based anomaly screening
- Baseline Random Forest prediction
- Automated business findings and next-step suggestions

This is a decision-support prototype; important decisions should be validated with current data and domain expertise.
"""
    )
    st.stop()

try:
    df = load_dataset(uploaded.getvalue(), uploaded.name)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

engine = get_engine(df)
p = engine.profile()
st.success(f"Loaded {p['rows']:,} rows × {p['columns']:,} columns")

a, b, c, d = st.columns(4)
a.metric("Rows", f"{p['rows']:,}")
b.metric("Columns", f"{p['columns']:,}")
c.metric("Missing cells", f"{p['missing_cells']:,}")
d.metric("Duplicate rows", f"{p['duplicate_rows']:,}")

t1, t2, t3, t4 = st.tabs(["Profile", "Quality & EDA", "Prediction", "Business Intelligence"])

with t1:
    st.dataframe(engine.column_profile(), use_container_width=True)

with t2:
    st.subheader("Quality")
    st.dataframe(engine.quality_report(), use_container_width=True)

    st.subheader("Numeric summary")
    ns = engine.numeric_summary()
    if not ns.empty:
        st.dataframe(ns, use_container_width=True)
    else:
        st.info("No numeric columns.")

    st.subheader("Correlations")
    cr = engine.correlation_matrix()
    if not cr.empty:
        st.dataframe(cr, use_container_width=True)
    else:
        st.info("Not enough numeric columns.")

    st.subheader("Potential anomalies")
    ar = engine.anomaly_report()
    if not ar.empty:
        st.dataframe(ar, use_container_width=True)
    else:
        st.info("No anomaly candidates detected.")

with t3:
    target = st.selectbox("Target column", ["— none —"] + list(df.columns))
    if target != "— none —":
        r = engine.predict(target)
        if r["ok"]:
            st.metric("Model", r["model"])
            st.metric("Validation score", r["score_text"])
            st.write(r["explanation"])
            st.dataframe(r["feature_importance"], use_container_width=True)
        else:
            st.warning(r["message"])

with t4:
    st.subheader("Automated findings")
    findings = engine.business_findings()
    if findings:
        for x in findings:
            st.markdown(f"- {x}")
    else:
        st.info("No automated findings generated.")

    st.subheader("Recommended next steps")
    steps = engine.next_steps()
    if steps:
        for x in steps:
            st.markdown(f"- {x}")
    else:
        st.info("No next steps generated.")
