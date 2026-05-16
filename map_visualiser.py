"""
map_visualiser.py
=================
Generates interactive Folium HTML maps from drive test data:
  1. RSRP Heatmap
  2. Event/Fault Map (call drops, HO failures, coverage holes)
  3. Throughput Map
  4. AI Anomaly Map (from ai_engine output)
"""

import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
import pandas as pd
import numpy as np


# ── Colour helpers ────────────────────────────────────────────────────────────
def rsrp_to_color(rsrp):
    if rsrp >= -80:  return "#00e676"  # Excellent — green
    if rsrp >= -90:  return "#aeea00"  # Good — yellow-green
    if rsrp >= -100: return "#ffea00"  # Fair — yellow
    if rsrp >= -110: return "#ff6d00"  # Poor — orange
    return "#d50000"                   # Bad  — red


def sinr_to_color(sinr):
    if sinr >= 20:  return "#00e676"
    if sinr >= 13:  return "#aeea00"
    if sinr >= 0:   return "#ffea00"
    return "#d50000"


EVENT_COLORS = {
    "Call Drop":    "red",
    "HO Failure":   "orange",
    "Coverage Hole":"darkred",
    "Interference": "purple",
    "Normal":       "green",
}

EVENT_ICONS = {
    "Call Drop":    "phone-slash",
    "HO Failure":   "exchange-alt",
    "Coverage Hole":"signal",
    "Interference": "broadcast-tower",
    "Normal":       "check",
}


def make_rsrp_map(df: pd.DataFrame, output="map_rsrp.html"):
    """Interactive RSRP coverage map with coloured circle markers + heatmap."""
    center = [df["Latitude"].mean(), df["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB dark_matter")

    # Heatmap layer
    heat_data = [[row.Latitude, row.Longitude, max(row.RSRP + 140, 0)]
                 for row in df.itertuples()]
    HeatMap(heat_data, radius=18, blur=20, min_opacity=0.4,
            gradient={"0.2": "blue", "0.5": "lime", "1.0": "red"},
            name="RSRP Heatmap").add_to(m)

    # Individual sample markers
    marker_group = folium.FeatureGroup(name="Samples", show=False)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color=rsrp_to_color(row["RSRP"]),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>RSRP:</b> {row['RSRP']} dBm<br>"
                f"<b>RSRQ:</b> {row['RSRQ']} dB<br>"
                f"<b>SINR:</b> {row['SINR']} dB<br>"
                f"<b>Tput:</b> {row['Throughput']:.1f} Mbps<br>"
                f"<b>Cell:</b> {row.get('Cell_Name','N/A')}<br>"
                f"<b>Time:</b> {row.get('Timestamp','N/A')}",
                max_width=200
            ),
        ).add_to(marker_group)
    marker_group.add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.8);padding:12px 16px;border-radius:8px;
                color:white;font-family:monospace;font-size:12px;line-height:1.8">
        <b>RSRP Legend</b><br>
        <span style="color:#00e676">●</span> Excellent (≥ -80 dBm)<br>
        <span style="color:#aeea00">●</span> Good (-90 to -80)<br>
        <span style="color:#ffea00">●</span> Fair (-100 to -90)<br>
        <span style="color:#ff6d00">●</span> Poor (-110 to -100)<br>
        <span style="color:#d50000">●</span> Bad (< -110 dBm)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(output)
    print(f"✅ RSRP map saved → {output}")
    return output


def make_event_map(df: pd.DataFrame, output="map_events.html"):
    """Map showing faults: call drops, HO failures, coverage holes."""
    center = [df["Latitude"].mean(), df["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")

    # Normal route line
    route_coords = list(zip(df["Latitude"], df["Longitude"]))
    folium.PolyLine(route_coords, color="#2196F3", weight=2,
                    opacity=0.5, tooltip="Drive Route").add_to(m)

    # Event markers (cluster non-normal events)
    cluster = MarkerCluster(name="Fault Events").add_to(m)
    event_df = df[df["Event"] != "Normal"]

    for _, row in event_df.iterrows():
        color = EVENT_COLORS.get(row["Event"], "gray")
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            icon=folium.Icon(color=color, icon="exclamation-triangle",
                             prefix="fa"),
            popup=folium.Popup(
                f"<b>⚠️ {row['Event']}</b><br>"
                f"RSRP: {row['RSRP']} dBm<br>"
                f"SINR: {row['SINR']} dB<br>"
                f"Tput: {row['Throughput']:.1f} Mbps<br>"
                f"Cell: {row.get('Cell_Name','N/A')}<br>"
                f"Time: {row.get('Timestamp','N/A')}",
                max_width=200
            ),
            tooltip=row["Event"],
        ).add_to(cluster)

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(output)
    print(f"✅ Event map saved → {output}")
    return output


def make_throughput_map(df: pd.DataFrame, output="map_throughput.html"):
    """Throughput coverage map."""
    center = [df["Latitude"].mean(), df["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB dark_matter")

    max_tput = df["Throughput"].max()
    heat_data = [[row.Latitude, row.Longitude,
                  row.Throughput / max_tput]
                 for row in df.itertuples()]
    HeatMap(heat_data, radius=20, blur=25, min_opacity=0.5,
            gradient={"0.0": "red", "0.4": "orange",
                      "0.7": "yellow", "1.0": "green"},
            name="Throughput Heatmap").add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.8);padding:12px 16px;border-radius:8px;
                color:white;font-family:monospace;font-size:12px;line-height:1.8">
        <b>Throughput</b><br>
        <span style="color:#00ff00">●</span> High (> 50 Mbps)<br>
        <span style="color:#ffff00">●</span> Medium (20-50 Mbps)<br>
        <span style="color:#ff6600">●</span> Low (5-20 Mbps)<br>
        <span style="color:#ff0000">●</span> Very Low (< 5 Mbps)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    MiniMap(toggle_display=True).add_to(m)
    m.save(output)
    print(f"✅ Throughput map saved → {output}")
    return output


def make_anomaly_map(df: pd.DataFrame, anomaly_col="AI_Anomaly",
                     output="map_anomalies.html"):
    """Map highlighting AI-detected anomalies."""
    center = [df["Latitude"].mean(), df["Longitude"].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB dark_matter")

    route_coords = list(zip(df["Latitude"], df["Longitude"]))
    folium.PolyLine(route_coords, color="#607D8B", weight=2,
                    opacity=0.4).add_to(m)

    normal_group  = folium.FeatureGroup(name="Normal", show=True)
    anomaly_group = folium.FeatureGroup(name="AI Anomalies", show=True)

    for _, row in df.iterrows():
        is_anomaly = row.get(anomaly_col, 0) == -1
        if is_anomaly:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=8,
                color="#FF1744",
                fill=True,
                fill_color="#FF1744",
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>🤖 AI Anomaly Detected</b><br>"
                    f"Score: {row.get('Anomaly_Score', 'N/A')}<br>"
                    f"RSRP: {row['RSRP']} dBm<br>"
                    f"SINR: {row['SINR']} dB<br>"
                    f"Tput: {row['Throughput']:.1f} Mbps<br>"
                    f"Actual Event: {row.get('Event','N/A')}",
                    max_width=220
                ),
                tooltip="⚠️ AI Anomaly",
            ).add_to(anomaly_group)
        else:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=3,
                color="#00BCD4",
                fill=True,
                fill_opacity=0.4,
            ).add_to(normal_group)

    normal_group.add_to(m)
    anomaly_group.add_to(m)

    anomaly_count = (df.get(anomaly_col, pd.Series()) == -1).sum()
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.85);padding:14px 18px;border-radius:8px;
                color:white;font-family:monospace;font-size:12px;line-height:2">
        <b>🤖 AI Anomaly Detection</b><br>
        <span style="color:#FF1744">●</span> Anomaly detected ({anomaly_count} pts)<br>
        <span style="color:#00BCD4">●</span> Normal ({len(df) - anomaly_count} pts)<br>
        <br><i>Model: Isolation Forest</i>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)
    m.save(output)
    print(f"✅ Anomaly map saved → {output}")
    return output


def generate_all_maps(df: pd.DataFrame):
    """Generate all 4 maps at once."""
    print("\n🗺️  Generating maps...")
    make_rsrp_map(df)
    make_event_map(df)
    make_throughput_map(df)
    if "AI_Anomaly" in df.columns:
        make_anomaly_map(df)
    print("✅ All maps generated.")


if __name__ == "__main__":
    from analyser import run_analysis
    result = run_analysis("drive_test_data.csv")
    generate_all_maps(result["df"])
