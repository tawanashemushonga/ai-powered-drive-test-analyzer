"""
ai_engine.py
============
AI/ML layer for the drive test analyser. Three components:

1. Isolation Forest — unsupervised anomaly detection on KPI features
2. KMeans Clustering — segment the drive route into performance zones
3. LLM Report Generator — calls Ollama (local, free) to write a
   professional optimisation report in plain English.
   Falls back to a rule-based report if Ollama is not installed.
"""

import pandas as pd
import numpy as np
import json
import requests
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── 1. Anomaly Detection ──────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame,
                     features=("RSRP", "RSRQ", "SINR", "Throughput"),
                     contamination=0.08) -> pd.DataFrame:
    """
    Isolation Forest anomaly detection.
    Adds columns: AI_Anomaly (-1=anomaly, 1=normal), Anomaly_Score.
    """
    print("🤖 Running Isolation Forest anomaly detection...")

    X = df[list(features)].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    df = df.copy()
    df["AI_Anomaly"]     = model.fit_predict(X_scaled)
    df["Anomaly_Score"]  = model.score_samples(X_scaled).round(4)

    n_anomalies = (df["AI_Anomaly"] == -1).sum()
    total       = len(df)

    # Precision check: how many real events did the AI catch?
    if "Event" in df.columns:
        real_events  = df[df["Event"] != "Normal"]
        ai_caught    = real_events[real_events["AI_Anomaly"] == -1]
        detection_rt = round(len(ai_caught) / max(len(real_events), 1) * 100, 1)
        print(f"   Anomalies detected : {n_anomalies}/{total} ({n_anomalies/total*100:.1f}%)")
        print(f"   Event detection rate: {detection_rt}%")
    else:
        print(f"   Anomalies detected : {n_anomalies}/{total}")

    return df


# ── 2. KPI Clustering ─────────────────────────────────────────────────────────

CLUSTER_LABELS = {
    0: "Excellent Coverage Zone",
    1: "Good Coverage Zone",
    2: "Fair Coverage Zone",
    3: "Poor Coverage Zone",
    4: "Critical Zone",
}


def cluster_drive_route(df: pd.DataFrame, n_clusters=4) -> pd.DataFrame:
    """
    KMeans clustering to segment the drive into performance zones.
    Adds: Cluster (int), Cluster_Label (str).
    """
    print(f"🤖 Running KMeans clustering (k={n_clusters})...")

    features = ["RSRP", "SINR", "Throughput"]
    X = df[features].fillna(df[features].median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Sort clusters by mean RSRP so label 0 = best, n-1 = worst
    cluster_means = (df.groupby("Cluster")["RSRP"]
                       .mean()
                       .sort_values(ascending=False))
    rank_map  = {old: new for new, old in enumerate(cluster_means.index)}
    df["Cluster"] = df["Cluster"].map(rank_map)
    df["Cluster_Label"] = df["Cluster"].map(
        {i: CLUSTER_LABELS.get(i, f"Zone {i}") for i in range(n_clusters)})

    summary = df.groupby("Cluster_Label")[["RSRP", "SINR", "Throughput"]].mean().round(1)
    print("   Cluster summary:")
    print(summary.to_string())
    return df


# ── 3. LLM Report Generator ───────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"


def _call_ollama(prompt: str, timeout=120) -> str:
    """Call local Ollama instance. Returns empty string on failure."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"   ⚠️  Ollama not available ({e}). Using rule-based report.")
    return ""


def _rule_based_report(stats: dict, worst_cells: pd.DataFrame) -> str:
    """Fallback professional report when Ollama is not running."""
    s = stats
    worst_cell_txt = ""
    if not worst_cells.empty:
        top = worst_cells.iloc[0]
        worst_cell_txt = (f"The worst performing cell is {top['Cell_Name']} "
                          f"with a mean RSRP of {top['Mean_RSRP']:.1f} dBm "
                          f"and {top['Events']} fault events.")

    coverage_assessment = (
        "Excellent" if s["coverage_above_minus110"] >= 95 else
        "Good"      if s["coverage_above_minus110"] >= 85 else
        "Fair"      if s["coverage_above_minus110"] >= 70 else "Poor"
    )

    recommendations = []
    if s["rsrp_mean"] < -95:
        recommendations.append(
            "Consider reducing mechanical downtilt by 2° on underperforming sectors "
            "to improve edge coverage.")
    if s["sinr_mean"] < 10:
        recommendations.append(
            "High interference detected. Review neighbour lists and consider "
            "power reduction on overshooting cells.")
    if s["ho_failures"] > 5:
        recommendations.append(
            f"{s['ho_failures']} handover failures detected. Review A3 offset "
            "and hysteresis parameters on affected cell borders.")
    if s["call_drops"] > 3:
        recommendations.append(
            f"{s['call_drops']} call drops logged. Investigate RLF (Radio Link "
            "Failure) root cause — likely poor SINR at cell edge.")
    if s["tput_mean_mbps"] < 20:
        recommendations.append(
            "Average throughput below target. Check scheduler configuration "
            "and consider carrier aggregation where supported.")
    if not recommendations:
        recommendations.append(
            "Network performance is within acceptable thresholds. Continue "
            "regular monitoring and quarterly drive test campaigns.")

    rec_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(recommendations))

    return f"""
DRIVE TEST OPTIMISATION REPORT
Generated by AI Drive Test Analyser
{'='*60}

EXECUTIVE SUMMARY
-----------------
The drive test campaign recorded {s['total_samples']} measurement samples
across the network. Overall coverage quality is assessed as {coverage_assessment}.

KEY PERFORMANCE INDICATORS
--------------------------
  RSRP  — Mean: {s['rsrp_mean']} dBm  |  5th pct: {s['rsrp_p5']} dBm
  RSRQ  — Mean: {s['rsrq_mean']} dB
  SINR  — Mean: {s['sinr_mean']} dB   |  5th pct: {s['sinr_p5']} dB
  Throughput — Mean: {s['tput_mean_mbps']} Mbps | 95th pct: {s['tput_p95_mbps']} Mbps

COVERAGE ANALYSIS
-----------------
  Coverage above -110 dBm : {s['coverage_above_minus110']}%
  Coverage above -100 dBm : {s['coverage_above_minus100']}%
  RSRP Excellent (≥-80)   : {s['rsrp_excellent']}%
  RSRP Good (-90 to -80)  : {s['rsrp_good']}%
  RSRP Fair (-100 to -90) : {s['rsrp_fair']}%
  RSRP Poor (-110 to -100): {s['rsrp_poor']}%
  RSRP Bad (<-110 dBm)    : {s['rsrp_bad']}%

FAULT ANALYSIS
--------------
  Call Drops     : {s['call_drops']}
  HO Failures    : {s['ho_failures']}
  Coverage Holes : {s['coverage_holes']} measurement points
  Interference   : {s['interference_pts']} measurement points

WORST PERFORMING CELL
---------------------
{worst_cell_txt}

RECOMMENDATIONS
---------------
{rec_text}

CONCLUSION
----------
Immediate action is recommended on the identified coverage holes and
handover failure zones. A follow-up drive test should be conducted
within 30 days of parameter changes to validate improvements.
{'='*60}
"""


def generate_ai_report(stats: dict, worst_cells: pd.DataFrame,
                        df: pd.DataFrame) -> str:
    """
    Generate optimisation report.
    Tries Ollama first; falls back to rule-based if unavailable.
    """
    print("\n📝 Generating AI optimisation report...")

    # Build a concise KPI summary for the LLM prompt
    anomaly_count = int((df.get("AI_Anomaly", pd.Series(1)) == -1).sum())

    prompt = f"""You are a senior telecoms RF optimisation engineer.
Analyse the following drive test KPI summary and write a professional
optimisation report with executive summary, findings, and concrete
parameter-change recommendations.

KPI SUMMARY:
- Total samples: {stats['total_samples']}
- RSRP mean: {stats['rsrp_mean']} dBm (5th pct: {stats['rsrp_p5']} dBm)
- RSRQ mean: {stats['rsrq_mean']} dB
- SINR mean: {stats['sinr_mean']} dB (5th pct: {stats['sinr_p5']} dB)
- Mean throughput: {stats['tput_mean_mbps']} Mbps
- Coverage > -110 dBm: {stats['coverage_above_minus110']}%
- Coverage > -100 dBm: {stats['coverage_above_minus100']}%
- RSRP Excellent: {stats['rsrp_excellent']}%, Good: {stats['rsrp_good']}%,
  Fair: {stats['rsrp_fair']}%, Poor: {stats['rsrp_poor']}%, Bad: {stats['rsrp_bad']}%
- Call drops: {stats['call_drops']}
- Handover failures: {stats['ho_failures']}
- Coverage hole points: {stats['coverage_holes']}
- Interference points: {stats['interference_pts']}
- AI-detected anomalies (Isolation Forest): {anomaly_count}

Write a 400-500 word professional report with sections:
1. Executive Summary
2. Coverage Analysis
3. Interference & Quality Assessment
4. Fault Analysis
5. Recommendations (numbered, specific parameter changes)
6. Conclusion
"""

    llm_report = _call_ollama(prompt)
    if llm_report:
        print("✅ LLM report generated via Ollama.")
        return llm_report
    else:
        print("✅ Rule-based report generated (install Ollama for LLM reports).")
        return _rule_based_report(stats, worst_cells)


# ── Full AI pipeline ──────────────────────────────────────────────────────────

def run_ai_pipeline(df: pd.DataFrame, stats: dict,
                    worst_cells: pd.DataFrame) -> tuple:
    """Run full AI pipeline and return (enriched_df, report_text)."""
    print("\n🧠 Running AI pipeline...")
    df     = detect_anomalies(df)
    df     = cluster_drive_route(df)
    report = generate_ai_report(stats, worst_cells, df)
    return df, report


if __name__ == "__main__":
    from analyser import run_analysis
    result          = run_analysis("drive_test_data.csv")
    df, report      = run_ai_pipeline(
        result["df"], result["stats"], result["worst"])
    print(report)
