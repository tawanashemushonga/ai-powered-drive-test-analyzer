"""
analyser.py
===========
Parses drive test CSV (real TEMS/NEMO export or synthetic) and computes
all standard telecoms KPIs with thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── 3GPP / Industry KPI Thresholds ────────────────────────────────────────────
THRESHOLDS = {
    "RSRP": {
        "Excellent": (-80,  0),
        "Good":      (-90,  -80),
        "Fair":      (-100, -90),
        "Poor":      (-110, -100),
        "Bad":       (-140, -110),
    },
    "RSRQ": {
        "Excellent": (-10, 0),
        "Good":      (-13, -10),
        "Fair":      (-15, -13),
        "Poor":      (-20, -15),
    },
    "SINR": {
        "Excellent": (20,  100),
        "Good":      (13,  20),
        "Fair":      (0,   13),
        "Poor":      (-10, 0),
    },
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate drive test CSV."""
    df = pd.read_csv(filepath, parse_dates=["Timestamp"])
    required = ["Latitude", "Longitude", "RSRP", "RSRQ", "SINR", "Throughput"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["Latitude", "Longitude", "RSRP"])
    print(f"📂 Loaded {len(df)} samples from {filepath}")
    return df


def classify_kpi(value, thresholds):
    """Classify a single KPI value against threshold dict."""
    for label, (low, high) in thresholds.items():
        if low <= value < high:
            return label
    return "Unknown"


def add_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Add classification columns for each KPI."""
    df["RSRP_Class"] = df["RSRP"].apply(
        lambda x: classify_kpi(x, THRESHOLDS["RSRP"]))
    df["RSRQ_Class"] = df["RSRQ"].apply(
        lambda x: classify_kpi(x, THRESHOLDS["RSRQ"]))
    df["SINR_Class"] = df["SINR"].apply(
        lambda x: classify_kpi(x, THRESHOLDS["SINR"]))
    return df


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute all headline KPI statistics."""
    n = len(df)

    def pct(mask): return round(mask.sum() / n * 100, 2)

    stats = {
        # ── RSRP ──────────────────────────────────────────────────────────────
        "rsrp_mean":       round(df["RSRP"].mean(), 2),
        "rsrp_median":     round(df["RSRP"].median(), 2),
        "rsrp_std":        round(df["RSRP"].std(), 2),
        "rsrp_p5":         round(df["RSRP"].quantile(0.05), 2),
        "rsrp_p95":        round(df["RSRP"].quantile(0.95), 2),
        "rsrp_excellent":  pct(df["RSRP"] >= -80),
        "rsrp_good":       pct((df["RSRP"] >= -90) & (df["RSRP"] < -80)),
        "rsrp_fair":       pct((df["RSRP"] >= -100) & (df["RSRP"] < -90)),
        "rsrp_poor":       pct((df["RSRP"] >= -110) & (df["RSRP"] < -100)),
        "rsrp_bad":        pct(df["RSRP"] < -110),
        # ── RSRQ ──────────────────────────────────────────────────────────────
        "rsrq_mean":       round(df["RSRQ"].mean(), 2),
        "rsrq_p5":         round(df["RSRQ"].quantile(0.05), 2),
        # ── SINR ──────────────────────────────────────────────────────────────
        "sinr_mean":       round(df["SINR"].mean(), 2),
        "sinr_median":     round(df["SINR"].median(), 2),
        "sinr_p5":         round(df["SINR"].quantile(0.05), 2),
        # ── Throughput ────────────────────────────────────────────────────────
        "tput_mean_mbps":  round(df["Throughput"].mean(), 2),
        "tput_median_mbps":round(df["Throughput"].median(), 2),
        "tput_p5_mbps":    round(df["Throughput"].quantile(0.05), 2),
        "tput_p95_mbps":   round(df["Throughput"].quantile(0.95), 2),
        # ── Events ────────────────────────────────────────────────────────────
        "total_samples":   n,
        "call_drops":      int((df["Event"] == "Call Drop").sum()),
        "ho_failures":     int((df["Event"] == "HO Failure").sum()),
        "coverage_holes":  int((df["Event"] == "Coverage Hole").sum()),
        "interference_pts":int((df["Event"] == "Interference").sum()),
        "normal_pts":      int((df["Event"] == "Normal").sum()),
        # ── Coverage KPIs (industry standard targets) ─────────────────────────
        "coverage_above_minus110": pct(df["RSRP"] >= -110),
        "coverage_above_minus100": pct(df["RSRP"] >= -100),
        "sinr_above_zero":         pct(df["SINR"] >= 0),
    }
    return stats


def worst_cells(df: pd.DataFrame, top_n=5) -> pd.DataFrame:
    """Identify worst performing cells."""
    if "Cell_Name" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("Cell_Name")
        .agg(
            Samples=("RSRP", "count"),
            Mean_RSRP=("RSRP", "mean"),
            Mean_RSRQ=("RSRQ", "mean"),
            Mean_SINR=("SINR", "mean"),
            Mean_Tput=("Throughput", "mean"),
            Events=("Event", lambda x: (x != "Normal").sum()),
        )
        .round(2)
        .sort_values("Mean_RSRP")
        .head(top_n)
        .reset_index()
    )


def kpi_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """RSRP distribution breakdown (for bar charts)."""
    labels = ["Excellent (≥-80)", "Good (-90 to -80)", "Fair (-100 to -90)",
              "Poor (-110 to -100)", "Bad (<-110)"]
    counts = [
        (df["RSRP"] >= -80).sum(),
        ((df["RSRP"] >= -90) & (df["RSRP"] < -80)).sum(),
        ((df["RSRP"] >= -100) & (df["RSRP"] < -90)).sum(),
        ((df["RSRP"] >= -110) & (df["RSRP"] < -100)).sum(),
        (df["RSRP"] < -110).sum(),
    ]
    pcts = [round(c / len(df) * 100, 1) for c in counts]
    return pd.DataFrame({"Category": labels, "Count": counts, "Percentage": pcts})


def run_analysis(filepath: str) -> dict:
    """Full pipeline: load → classify → stats → return all results."""
    df    = load_data(filepath)
    df    = add_classifications(df)
    stats = compute_summary_stats(df)
    worst = worst_cells(df)
    dist  = kpi_distribution(df)

    print("\n📊 KPI Summary")
    print(f"   RSRP mean      : {stats['rsrp_mean']} dBm")
    print(f"   SINR mean      : {stats['sinr_mean']} dB")
    print(f"   Throughput mean: {stats['tput_mean_mbps']} Mbps")
    print(f"   Coverage >-110 : {stats['coverage_above_minus110']}%")
    print(f"   Call Drops     : {stats['call_drops']}")
    print(f"   HO Failures    : {stats['ho_failures']}")

    return {
        "df":    df,
        "stats": stats,
        "worst": worst,
        "dist":  dist,
    }


if __name__ == "__main__":
    result = run_analysis("drive_test_data.csv")
    print("\n🏚️ Worst Cells:")
    print(result["worst"].to_string(index=False))
    print("\n📶 RSRP Distribution:")
    print(result["dist"].to_string(index=False))
