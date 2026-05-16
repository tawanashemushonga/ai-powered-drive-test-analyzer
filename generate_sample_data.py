"""
generate_sample_data.py
=======================
Generates realistic synthetic drive test data mimicking TEMS/NEMO CSV exports.
Uses Harare, Zimbabwe as the drive route area.
Run this first if you don't have real logs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ── Harare city drive route waypoints (lat, lon) ──────────────────────────────
ROUTE_WAYPOINTS = [
    (-17.8252, 31.0335),  # CBD
    (-17.8200, 31.0450),  # Avenues
    (-17.8150, 31.0520),  # Belgravia
    (-17.8080, 31.0610),  # Borrowdale
    (-17.8000, 31.0700),  # Highlands
    (-17.8100, 31.0580),  # Greendale
    (-17.8300, 31.0480),  # Mabelreign
    (-17.8400, 31.0350),  # Hatfield
    (-17.8350, 31.0200),  # Southerton
    (-17.8252, 31.0335),  # Back to CBD
]

# ── Cell site definitions (name, lat, lon, frequency, power) ──────────────────
CELL_SITES = [
    {"name": "HRE_CBD_1",      "lat": -17.8252, "lon": 31.0335, "freq": 1800, "power": 43},
    {"name": "HRE_AVENUES_1",  "lat": -17.8180, "lon": 31.0470, "freq": 2100, "power": 40},
    {"name": "HRE_BORROW_1",   "lat": -17.8060, "lon": 31.0650, "freq": 800,  "power": 46},
    {"name": "HRE_SOUTH_1",    "lat": -17.8420, "lon": 31.0250, "freq": 1800, "power": 43},
    {"name": "HRE_GREEN_1",    "lat": -17.8120, "lon": 31.0560, "freq": 2600, "power": 38},
]

TECHNOLOGIES = ["LTE", "LTE", "LTE", "UMTS", "LTE"]  # Mostly LTE
OPERATORS    = ["NetOne", "Econet", "Telecel"]


def interpolate_route(waypoints, points_per_segment=50):
    """Smoothly interpolate GPS points along the route."""
    lats, lons = [], []
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        for j in range(points_per_segment):
            t = j / points_per_segment
            lats.append(lat1 + t * (lat2 - lat1) + np.random.normal(0, 0.0002))
            lons.append(lon1 + t * (lon2 - lon1) + np.random.normal(0, 0.0002))
    return np.array(lats), np.array(lons)


def calc_rsrp(lat, lon, cell, shadowing_std=8):
    """Simplified RSRP using free-space path loss + log-normal shadowing."""
    dist_km = np.sqrt((lat - cell["lat"])**2 + (lon - cell["lon"])**2) * 111
    dist_m  = max(dist_km * 1000, 10)
    freq_hz = cell["freq"] * 1e6
    fspl_db = 20 * np.log10(dist_m) + 20 * np.log10(freq_hz) - 147.55
    rsrp    = cell["power"] - fspl_db + np.random.normal(0, shadowing_std)
    return float(np.clip(rsrp - 50, -140, -44))  # Convert to RSRP range


def rsrp_to_rsrq(rsrp):
    """Derive RSRQ from RSRP with noise."""
    rsrq = (rsrp + 100) / 5 - 20 + np.random.normal(0, 1.5)
    return float(np.clip(rsrq, -20, -3))


def rsrp_to_sinr(rsrp):
    """Derive SINR from RSRP with noise."""
    sinr = (rsrp + 120) / 4 + np.random.normal(0, 2)
    return float(np.clip(sinr, -10, 30))


def sinr_to_throughput(sinr):
    """Shannon-inspired throughput from SINR (Mbps)."""
    bw_hz   = 20e6  # 20 MHz LTE
    tput    = (bw_hz / 1e6) * np.log2(1 + 10 ** (sinr / 10)) * 0.6
    tput   += np.random.normal(0, 2)
    return float(np.clip(tput, 0, 150))


def inject_problems(df):
    """Inject realistic network problems into the data."""
    n = len(df)
    
    # Coverage hole — segment with very weak signal
    hole_start = int(n * 0.25)
    hole_end   = int(n * 0.32)
    df.loc[hole_start:hole_end, "RSRP"]       -= 25
    df.loc[hole_start:hole_end, "RSRQ"]       -= 5
    df.loc[hole_start:hole_end, "SINR"]       -= 10
    df.loc[hole_start:hole_end, "Throughput"] *= 0.1
    df.loc[hole_start:hole_end, "Event"]       = "Coverage Hole"

    # Interference zone
    interf_start = int(n * 0.55)
    interf_end   = int(n * 0.62)
    df.loc[interf_start:interf_end, "SINR"]       -= 15
    df.loc[interf_start:interf_end, "Throughput"] *= 0.3
    df.loc[interf_start:interf_end, "Event"]       = "Interference"

    # Handover failures
    ho_indices = random.sample(range(n), 15)
    for idx in ho_indices:
        df.loc[idx, "Event"] = "HO Failure"

    # Call drops
    drop_indices = random.sample(range(n), 8)
    for idx in drop_indices:
        df.loc[idx, "Event"] = "Call Drop"
        df.loc[idx, "RSRP"]  = np.random.uniform(-130, -120)

    # Clip values to valid ranges after injection
    df["RSRP"]       = df["RSRP"].clip(-140, -44)
    df["RSRQ"]       = df["RSRQ"].clip(-20, -3)
    df["SINR"]       = df["SINR"].clip(-10, 30)
    df["Throughput"] = df["Throughput"].clip(0, 150)
    return df


def generate_drive_test_data(output_file="drive_test_data.csv", num_points=450):
    """Main generator function."""
    print("🚗 Generating synthetic drive test data...")

    lats, lons = interpolate_route(ROUTE_WAYPOINTS)

    # Trim/extend to desired length
    idx  = np.linspace(0, len(lats) - 1, num_points).astype(int)
    lats = lats[idx]
    lons = lons[idx]

    start_time = datetime(2024, 6, 15, 8, 0, 0)
    records    = []

    for i in range(num_points):
        lat = lats[i]
        lon = lons[i]
        ts  = start_time + timedelta(seconds=i * 8)

        # Find serving cell (strongest signal)
        rsrps = [calc_rsrp(lat, lon, cell) for cell in CELL_SITES]
        best_cell_idx = int(np.argmax(rsrps))
        best_cell     = CELL_SITES[best_cell_idx]
        rsrp          = rsrps[best_cell_idx]
        rsrq          = rsrp_to_rsrq(rsrp)
        sinr          = rsrp_to_sinr(rsrp)
        throughput    = sinr_to_throughput(sinr)

        records.append({
            "Timestamp":    ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Latitude":     round(lat, 6),
            "Longitude":    round(lon, 6),
            "Operator":     random.choices(OPERATORS, weights=[40, 45, 15])[0],
            "Technology":   TECHNOLOGIES[best_cell_idx],
            "Cell_Name":    best_cell["name"],
            "Band":         best_cell["freq"],
            "RSRP":         round(rsrp, 2),
            "RSRQ":         round(rsrq, 2),
            "SINR":         round(sinr, 2),
            "Throughput":   round(throughput, 2),
            "RSRP_Neighbor1": round(rsrps[(best_cell_idx + 1) % len(CELL_SITES)], 2),
            "RSRP_Neighbor2": round(rsrps[(best_cell_idx + 2) % len(CELL_SITES)], 2),
            "Speed_kmh":    round(abs(np.random.normal(40, 15)), 1),
            "Event":        "Normal",
        })

    df = pd.DataFrame(records)
    df = inject_problems(df)
    df.to_csv(output_file, index=False)

    print(f"✅ Generated {num_points} samples → {output_file}")
    print(f"   RSRP range : {df['RSRP'].min():.1f} to {df['RSRP'].max():.1f} dBm")
    print(f"   SINR range : {df['SINR'].min():.1f} to {df['SINR'].max():.1f} dB")
    print(f"   Events     : {df['Event'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    generate_drive_test_data()
