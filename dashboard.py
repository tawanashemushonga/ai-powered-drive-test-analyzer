"""
dashboard.py
============
Streamlit web dashboard for the AI Drive Test Analyser.

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from generate_sample_data import generate_drive_test_data
from analyser import run_analysis
from ai_engine import run_ai_pipeline
from map_visualiser import (make_rsrp_map, make_event_map,
                            make_throughput_map, make_anomaly_map)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Drive Test Analyser",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Sleek Enterprise Light Theme) ─────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f8f9fa; color: #212529; }
  [data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid #dee2e6; }
  
  /* Metric Cards */
  .metric-card {
    background: #ffffff; 
    border-radius: 6px; 
    padding: 20px; 
    text-align: left;
    border: 1px solid #e9ecef;
    border-left: 4px solid #6c757d;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .metric-value { font-size: 1.8rem; font-weight: 600; font-family: monospace; line-height: 1.2; color: #212529; }
  .metric-label { font-size: 0.85rem; color: #495057; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  
  /* Status Colors */
  .kpi-good  { color: #2ea043; border-left-color: #2ea043 !important; }
  .kpi-warn  { color: #a67c1e; border-left-color: #d29922 !important; }
  .kpi-bad   { color: #d9383a; border-left-color: #f85149 !important; }
  
  h1, h2, h3 { color: #212529 !important; font-weight: 500 !important; }
  
  /* Primary Button Styling */
  .stButton > button {
    background: #0d6efd; color: white; border: 1px solid #0d6efd;
    border-radius: 4px; padding: 6px 16px; font-weight: 500;
  }
  .stButton > button:hover { background: #0b5ed7; color: white; border-color: #0a58ca; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AI Drive Test Analyser")
    st.markdown("---")

    st.markdown("### Data Source")
    data_source = st.radio("", ["Generate Sample Data", "Upload CSV"], label_visibility="collapsed")

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload TEMS/NEMO CSV", type=["csv"],
            help="CSV must have: Latitude, Longitude, RSRP, RSRQ, SINR, Throughput"
        )

    st.markdown("---")
    st.markdown("### AI Engine Parameters")
    contamination = st.slider(
        "Anomaly Sensitivity", 0.01, 0.20, 0.08,
        help="Higher values flag more data points as anomalies."
    )
    n_clusters = st.slider("Performance Zones (KMeans)", 2, 6, 4)

    st.markdown("---")
    st.markdown("### Map Visualization Settings")
    map_type = st.selectbox(
        "Layer Select",
        ["RSRP Coverage", "Fault Events", "Throughput", "AI Anomalies"]
    )

    run_btn = st.button("Execute Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#495057; font-size:11px; font-family: monospace; line-height: 1.5;">
    <b>ARCHITECTURE FRAMEWORK:</b><br>
    Python · Pandas · Scikit-learn<br>
    Isolation Forest · KMeans<br>
    Folium · Plotly · Streamlit<br>
    Ollama LLM Integration
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# AI-Powered Drive Test Analytics Engine")
st.markdown("<p style='color:#495057; margin-top:-15px;'>Telecom Network Performance Diagnostics & Anomaly Detection Framework</p>", unsafe_allow_html=True)
st.markdown("---")


@st.cache_data(show_spinner=False)
def load_and_analyse(source, file_content=None, contam=0.08, n_clust=4):
    """Cached execution of the data pipeline."""
    if source == "Generate Sample Data":
        df_raw = generate_drive_test_data("drive_test_data.csv", num_points=450)
    else:
        import io
        df_raw = pd.read_csv(io.BytesIO(file_content))
        df_raw.to_csv("drive_test_data.csv", index=False)

    result          = run_analysis("drive_test_data.csv")
    df_ai, report   = run_ai_pipeline(
        result["df"], result["stats"], result["worst"])
    result["df"]    = df_ai
    result["report"]= report
    return result


if run_btn or "result" not in st.session_state:
    with st.spinner("Executing core AI data pipeline analysis..."):
        file_bytes = uploaded_file.read() if uploaded_file else None
        result = load_and_analyse(
            data_source, file_bytes, contamination, n_clusters)
        st.session_state["result"] = result
    st.success("Pipeline execution completed successfully.")

result = st.session_state.get("result")
if result is None:
    st.info("Action Required: Click 'Execute Analysis' in the sidebar configuration to initialize dashboard data.")
    st.stop()

df    = result["df"]
stats = result["stats"]
worst = result["worst"]


# ── KPI Headline Metrics ───────────────────────────────────────────────────────
st.markdown("## Core Network KPIs")

def color_class(val, good, warn):
    if val >= good: return "kpi-good"
    if val >= warn: return "kpi-warn"
    return "kpi-bad"

anomaly_count = int((df["AI_Anomaly"] == -1).sum()) if "AI_Anomaly" in df.columns else 0

cols = st.columns(6)
metrics = [
    ("RSRP Mean", f"{stats['rsrp_mean']} dBm",
     color_class(stats['rsrp_mean'], -85, -100)),
    ("SINR Mean",  f"{stats['sinr_mean']} dB",
     color_class(stats['sinr_mean'], 15, 5)),
    ("Avg Throughput", f"{stats['tput_mean_mbps']} Mbps",
     color_class(stats['tput_mean_mbps'], 30, 10)),
    ("Coverage > -110dBm", f"{stats['coverage_above_minus110']}%",
     color_class(stats['coverage_above_minus110'], 95, 80)),
    ("Hardware / Call Faults",
     f"{stats['call_drops'] + stats['ho_failures']}",
     "kpi-bad" if (stats['call_drops'] + stats['ho_failures']) > 5 else "kpi-warn"),
    ("Model Anomalies", str(anomaly_count),
     "kpi-bad" if anomaly_count > 30 else "kpi-warn"),
]

for col, (label, value, cls) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card {cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "KPI Performance Charts", "Spatial Coverage Maps",
    "Machine Learning Anomalies", "Degraded Cells", "Analytical Optimization Report"
])


# ── Tab 1: KPI Charts ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Metric Analytics Over Time")
    fig_ts = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=("Reference Signal Received Power (RSRP)", "Reference Signal Received Quality (RSRQ)",
                                           "Signal-to-Interference-plus-Noise Ratio (SINR)", "Data Throughput"))
    x = list(range(len(df)))
    kpis = [("RSRP", "#2ea043"), ("RSRQ", "#a67c1e"),
            ("SINR", "#0d6efd"), ("Throughput", "#8a3ffc")]
    for i, (kpi, color) in enumerate(kpis, start=1):
        fig_ts.add_trace(go.Scatter(
            x=x, y=df[kpi], mode="lines",
            line=dict(color=color, width=1.5),
            name=kpi), row=i, col=1)
        fig_ts.add_hline(
            y={"RSRP": -110, "RSRQ": -15, "SINR": 0, "Throughput": 10}[kpi],
            line_dash="dash", line_color="#d9383a", opacity=0.7, row=i, col=1)

    fig_ts.update_layout(height=700, paper_bgcolor="#ffffff",
                         plot_bgcolor="#f8f9fa", font_color="#212529",
                         showlegend=False)
    fig_ts.update_xaxes(gridcolor="#dee2e6")
    fig_ts.update_yaxes(gridcolor="#dee2e6")
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### RSRP Distribution Spread")
        dist = result["dist"]
        colors_bar = ["#2ea043","#72c240","#d29922","#ff6d00","#f85149"]
        fig_bar = px.bar(dist, x="Category", y="Percentage",
                         color="Category",
                         color_discrete_sequence=colors_bar,
                         text="Percentage")
        fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_bar.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
                               font_color="#212529", showlegend=False,
                               xaxis_tickangle=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("### SINR vs Throughput Regression Plot")
        fig_sc = px.scatter(df, x="SINR", y="Throughput",
                            color="RSRP", color_continuous_scale="RdYlGn",
                            opacity=0.7, labels={"Throughput": "Throughput (Mbps)"})
        fig_sc.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
                              font_color="#212529")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("### Aggregate Event Categorization")
    event_counts = df["Event"].value_counts().reset_index()
    event_counts.columns = ["Event", "Count"]
    colors_evt = {"Normal":"#2ea043","Call Drop":"#f85149",
                  "HO Failure":"#d29922","Coverage Hole":"#ff6d00",
                  "Interference":"#8a3ffc"}
    fig_evt = px.pie(event_counts, names="Event", values="Count",
                     color="Event", color_discrete_map=colors_evt,
                     hole=0.5)
    fig_evt.update_layout(paper_bgcolor="#ffffff", font_color="#212529")
    st.plotly_chart(fig_evt, use_container_width=True)


# ── Tab 2: Maps ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"### Geographic Layer: {map_type}")
    st.markdown("<p style='color:#495057; font-size: 13px; margin-top:-10px;'>Map Interaction: Select plot markers to query exact structural coordinate points. Layer controllers are located in the map header.</p>", unsafe_allow_html=True)

    with st.spinner("Rendering geospatial layer..."):
        if map_type == "RSRP Coverage":
            map_file = make_rsrp_map(df, "map_rsrp.html")
        elif map_type == "Fault Events":
            map_file = make_event_map(df, "map_events.html")
        elif map_type == "Throughput":
            map_file = make_throughput_map(df, "map_throughput.html")
        else:
            map_file = make_anomaly_map(df, output="map_anomalies.html")

    with open(map_file, "r", encoding="utf-8") as f:
        map_html = f.read()
    st.components.v1.html(map_html, height=600, scrolling=False)


# ── Tab 3: AI Anomalies ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### Unsupervised Isolation Forest Anomaly Matrix")
    st.markdown("""
    The Isolation Forest engine identifies statistical multi-dimensional outliers across the composite telemetry metrics 
    (RSRP, RSRQ, SINR, and Data Throughput). Points requiring shorter partition pathways are calculated and isolated as 
    anomalous operation windows differing significantly from baseline network trends.
    """)

    if "AI_Anomaly" in df.columns:
        anomaly_df = df[df["AI_Anomaly"] == -1].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Identified Anomalies", anomaly_count,
                  f"{anomaly_count/len(df)*100:.1f}% total footprint Data", delta_color="off")
        real_events = df[df["Event"] != "Normal"]
        caught = real_events[real_events["AI_Anomaly"] == -1]
        c2.metric("True Positives Identified", f"{len(caught)}/{len(real_events)}",
                  f"{len(caught)/max(len(real_events),1)*100:.1f}% Detection Sensitivity", delta_color="off")
        c3.metric("Normal Baseline Crossings",
                  f"{(anomaly_df['Event']=='Normal').sum()}",
                  "Flagged False Positives", delta_color="off")

        st.markdown("### Mathematical Anomaly Score Density Distribution")
        fig_score = px.histogram(df, x="Anomaly_Score",
                                 color="AI_Anomaly",
                                 color_discrete_map={1: "#2ea043", -1: "#f85149"},
                                 nbins=50,
                                 labels={"AI_Anomaly": "Model Classification Status"})
        fig_score.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
                                font_color="#212529")
        st.plotly_chart(fig_score, use_container_width=True)

        st.markdown("### Isolated Outlier Logs")
        show_cols = ["Timestamp","Latitude","Longitude","RSRP","RSRQ",
                     "SINR","Throughput","Event","Anomaly_Score"]
        show_cols = [c for c in show_cols if c in anomaly_df.columns]
        st.dataframe(anomaly_df[show_cols].sort_values("Anomaly_Score").head(30),
                     use_container_width=True)

    if "Cluster_Label" in df.columns:
        st.markdown("### Performance Zone Vector Clustering (KMeans)")
        cluster_summary = (
            df.groupby("Cluster_Label")
            .agg(Points=("RSRP","count"), RSRP=("RSRP","mean"),
                 SINR=("SINR","mean"), Tput=("Throughput","mean"))
            .round(1).reset_index()
        )
        fig_cl = px.bar(cluster_summary, x="Cluster_Label",
                        y=["RSRP","SINR","Tput"], barmode="group",
                        color_discrete_sequence=["#2ea043","#0d6efd","#8a3ffc"])
        fig_cl.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
                              font_color="#212529", xaxis_tickangle=0)
        st.plotly_chart(fig_cl, use_container_width=True)


# ── Tab 4: Degraded Cells ─────────────────────────────────────────────────────
with tab4:
    st.markdown("### Lowest Performing Cells (Worst Cell Identification)")
    if not worst.empty:
        def color_rsrp(val):
            if val >= -90: return "background-color: #e8f5e9; color: #1b5e20"
            if val >= -100: return "background-color: #fffde7; color: #f57f17"
            return "background-color: #ffebee; color: #b71c1c"

        styled = worst.style.map(color_rsrp, subset=["Mean_RSRP"])
        st.dataframe(styled, use_container_width=True)

        fig_worst = px.bar(worst, x="Cell_Name", y="Mean_RSRP",
                           color="Mean_RSRP",
                           color_continuous_scale="RdYlGn",
                           range_color=[-130, -70],
                           text="Mean_RSRP",
                           title="Mean RSRP Evaluation by Cell Sector")
        fig_worst.update_traces(texttemplate="%{text} dBm", textposition="outside")
        fig_worst.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
                                font_color="#212529")
        st.plotly_chart(fig_worst, use_container_width=True)

    st.markdown("### Complete Telemetry Statistical Overview")
    stats_df = pd.DataFrame([
        {"KPI Parameter": "RSRP Mean", "Value": f"{stats['rsrp_mean']} dBm"},
        {"KPI Parameter": "RSRP 5th Percentile", "Value": f"{stats['rsrp_p5']} dBm"},
        {"KPI Parameter": "RSRQ Mean", "Value": f"{stats['rsrq_mean']} dB"},
        {"KPI Parameter": "SINR Mean", "Value": f"{stats['sinr_mean']} dB"},
        {"KPI Parameter": "Throughput Mean", "Value": f"{stats['tput_mean_mbps']} Mbps"},
        {"KPI Parameter": "Throughput 95th Percentile", "Value": f"{stats['tput_p95_mbps']} Mbps"},
        {"KPI Parameter": "Coverage Integrity Threshold > -110 dBm", "Value": f"{stats['coverage_above_minus110']}%"},
        {"KPI Parameter": "Coverage Integrity Threshold > -100 dBm", "Value": f"{stats['coverage_above_minus100']}%"},
        {"KPI Parameter": "Call Drop Totals", "Value": str(stats['call_drops'])},
        {"KPI Parameter": "Handover Failure Totals", "Value": str(stats['ho_failures'])},
        {"KPI Parameter": "Aggregate Population Samples", "Value": str(stats['total_samples'])},
    ])
    st.dataframe(stats_df, use_container_width=True)


# ── Tab 5: AI Report ──────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Automated Network Optimization Report")
    st.markdown("""
    This operational analysis artifact is constructed automatically from drive test KPI telemetry input data. 
    The logic engine handles contextual formatting dynamically based on running infrastructure (local LLM Llama3 state 
    pipelines, or a rigid engineering heuristic framework fallback matrix).
    """)

    report = result.get("report", "No report generated.")
    st.text_area("Analytical Output Viewer", report, height=600, label_visibility="collapsed")

    # Data Export Operations
    col1, col2 = st.columns(2)
    col1.download_button(
        "Export Optimization Report (.txt)",
        data=report,
        file_name="drive_test_optimisation_report.txt",
        mime="text/plain",
        use_container_width=True
    )
    col2.download_button(
        "Export Analyzed Datatable (.csv)",
        data=df.to_csv(index=False),
        file_name="drive_test_analysed.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#495057; font-size:11px; font-family: monospace;">
AI Drive Test Analyser · Corporate Analytics Core System Framework
</div>
""", unsafe_allow_html=True)