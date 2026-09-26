import io, os, json, time
import pandas as pd
import plotly.express as px
import streamlit as st
from core.engine import SPDSEngine

st.set_page_config(page_title="SPDS Intelligence OS", page_icon="◈", layout="wide")
st.title("◈ SPDS — Enterprise Intelligence Operating System")
st.caption("Autonomous data science • forecasting • scenario intelligence • decision support")

with st.sidebar:
    st.header("Workspace")
    uploaded=st.file_uploader("Upload company data",type=["csv","xlsx","xls","json","parquet"])
    target=st.text_input("Target column (optional)")
    date_col=st.text_input("Date column (optional)")
    horizon=st.slider("Forecast horizon",1,52,12)
    run=st.button("⚡ RUN AUTONOMOUS ANALYSIS",type="primary",use_container_width=True)
    st.divider()
    st.info("No API key is required for the analytical engine. An LLM can be connected later for richer reasoning.")

if uploaded:
    raw=uploaded.getvalue(); name=uploaded.name
    if st.session_state.get("dataset_name")!=name:
        ext=os.path.splitext(name)[1].lower()
        if ext==".csv": df=pd.read_csv(io.BytesIO(raw))
        elif ext in (".xlsx",".xls"): df=pd.read_excel(io.BytesIO(raw))
        elif ext==".json": df=pd.read_json(io.BytesIO(raw))
        else: df=pd.read_parquet(io.BytesIO(raw))
        st.session_state.dataset=df; st.session_state.dataset_name=name; st.session_state.results=None
    df=st.session_state.dataset
    st.success(f"Loaded {len(df):,} rows × {len(df.columns):,} columns — {name}")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Rows",f"{len(df):,}"); m2.metric("Columns",f"{len(df.columns):,}"); m3.metric("Missing cells",f"{int(df.isna().sum().sum()):,}"); m4.metric("Duplicates",f"{int(df.duplicated().sum()):,}")
    if run:
        with st.spinner("SPDS is executing independent analytical jobs in parallel…"):
            engine=SPDSEngine(df)
            st.session_state.results=engine.run_autonomous(target or None,date_col or None,horizon)
        st.success(f"Analysis complete in {st.session_state.results['_meta'].data['total_seconds']:.2f}s")

if "results" in st.session_state and st.session_state.results:
    r=st.session_state.results
    tabs=st.tabs(["Executive","Data Quality","EDA","Models","Forecast","Scenarios","Anomalies","Raw Results"])
    with tabs[0]:
        st.subheader("Executive intelligence")
        for x in r["insights"].data: st.write("• "+x)
        st.subheader("Job status")
        rows=[]
        for k,v in r.items():
            if k.startswith("_"): continue
            rows.append({"job":k,"status":v.status,"seconds":round(v.duration_s,3),"summary":v.summary})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    with tabs[1]:
        st.json(r["_meta"].data["profile"])
        st.dataframe(df.head(100),use_container_width=True)
    with tabs[2]:
        d=r["eda"].data
        if not d["describe"].empty: st.dataframe(d["describe"],use_container_width=True)
        if not d["correlation"].empty: st.plotly_chart(px.imshow(d["correlation"],text_auto=True,title="Correlation matrix"),use_container_width=True)
    with tabs[3]:
        if r["automl"].status=="ok": st.dataframe(r["automl"].data["leaderboard"],use_container_width=True,hide_index=True)
        else: st.warning(r["automl"].summary)
    with tabs[4]:
        if r["forecast"].status=="ok":
            fc=r["forecast"].data["forecast"]; fig=px.line(fc,x="date",y=["forecast","lower_95","upper_95"],title="Forecast with uncertainty band"); st.plotly_chart(fig,use_container_width=True); st.dataframe(fc,use_container_width=True,hide_index=True)
        else: st.warning(r["forecast"].summary)
    with tabs[5]:
        if r["scenarios"].status=="ok":
            sc=r["scenarios"].data; st.dataframe(sc,use_container_width=True,hide_index=True); st.plotly_chart(px.bar(sc,x="scenario",y="estimated_value",title="Scenario values"),use_container_width=True)
    with tabs[6]:
        if r["anomalies"].status=="ok": st.dataframe(r["anomalies"].data,use_container_width=True,hide_index=True)
        else: st.warning(r["anomalies"].summary)
    with tabs[7]:
        st.json({k:{"status":v.status,"duration_s":v.duration_s,"summary":v.summary,"error":v.error} for k,v in r.items() if k!="_meta"})
else:
    st.markdown("### Give SPDS a dataset and it does the workflow automatically")
    st.markdown("**Profile → clean → explore → detect anomalies → compare models → forecast → simulate scenarios → produce evidence-backed insights.**")
    st.code("Example request: upload sales data, select revenue as target, and press RUN AUTONOMOUS ANALYSIS.")
