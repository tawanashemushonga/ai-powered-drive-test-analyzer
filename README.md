# 📡 AI-Powered Drive Test Analyser

> A complete telecoms drive test analysis pipeline with ML anomaly detection,
> interactive coverage maps, and AI-generated optimisation reports.
> **100% free tools. No paid software required.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Scikit-learn](https://img.shields.io/badge/ML-Isolation%20Forest-orange)
![Folium](https://img.shields.io/badge/Maps-Folium-green)

---

## 🎯 What This Does

| Stage | What Happens |
|-------|-------------|
| **Data** | Load real TEMS/NEMO CSV logs OR generate realistic synthetic data |
| **Analysis** | Parse RSRP, RSRQ, SINR, Throughput · classify against 3GPP thresholds |
| **AI Detection** | Isolation Forest flags anomalous KPI measurements automatically |
| **AI Clustering** | KMeans segments the drive route into performance zones |
| **Mapping** | Folium generates interactive RSRP/event/throughput/anomaly maps |
| **AI Report** | LLM (Ollama) or rule-based engine writes a professional optimisation report |

---

## 📁 Project Structure

```
drive_test_ai/
├── generate_sample_data.py   # Synthetic TEMS/NEMO data generator
├── analyser.py               # KPI parsing and statistics engine
├── ai_engine.py              # Isolation Forest + KMeans + LLM report
├── map_visualiser.py         # Folium interactive map generator
├── dashboard.py              # Streamlit web dashboard (main app)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run dashboard.py
```

### 3. Or run the pipeline from CLI
```bash
# Generate synthetic data
python generate_sample_data.py

# Run analysis
python analyser.py

# Run AI pipeline
python ai_engine.py

# Generate all maps
python map_visualiser.py
```

---

## 🤖 AI Components

### Isolation Forest (Anomaly Detection)
- **Unsupervised** — no labelled data needed
- Detects abnormal combinations of RSRP, RSRQ, SINR, Throughput
- Configurable sensitivity (contamination parameter)
- Outputs anomaly score per measurement point

### KMeans Clustering (Zone Segmentation)
- Segments the drive route into performance zones (Excellent → Critical)
- Uses RSRP, SINR, Throughput as features
- Helps identify geographic clusters of poor performance

### LLM Report Generation (Ollama — Optional)
Install Ollama and pull Llama 3 for AI-written reports:
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3
```
Without Ollama, the tool uses a rule-based report engine that produces
equally professional output based on KPI thresholds.

---

## 📊 KPI Thresholds Used

| KPI | Excellent | Good | Fair | Poor | Bad |
|-----|-----------|------|------|------|-----|
| RSRP | ≥ -80 dBm | -90 to -80 | -100 to -90 | -110 to -100 | < -110 |
| RSRQ | ≥ -10 dB | -13 to -10 | -15 to -13 | < -15 | — |
| SINR | ≥ 20 dB | 13 to 20 | 0 to 13 | < 0 | — |

---

## 🗺️ Maps Generated

- **RSRP Coverage Map** — heatmap + coloured markers per threshold
- **Fault Events Map** — call drops, HO failures, coverage holes clustered
- **Throughput Map** — download speed heatmap
- **AI Anomaly Map** — Isolation Forest detections overlaid on route

---

## 📂 Real Data Sources (Free)

| Source | What You Get |
|--------|-------------|
| [OpenCelliD](https://opencellid.org) | Real cell tower coordinates |
| [Kaggle Telecoms Datasets](https://kaggle.com) | Drive test KPI datasets |
| Your operator | Export from TEMS Investigation or NEMO Outdoor |
| Network Cell Info (Android) | Free drive test app — exports CSV |

The CSV must contain: `Latitude, Longitude, RSRP, RSRQ, SINR, Throughput`

---

## 🛠️ Tech Stack (All Free)

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas + NumPy | Data analysis |
| Scikit-learn | ML (Isolation Forest, KMeans) |
| Folium | Interactive maps |
| Plotly | Charts and visualisations |
| Streamlit | Web dashboard |
| Ollama + Llama 3 | Local LLM for report generation |

---

## 📸 Dashboard Preview

The dashboard includes:
- KPI headline metrics with colour-coded RAG status
- Time series charts for all KPIs with threshold lines
- RSRP distribution bar chart
- SINR vs Throughput scatter plot
- Event/fault pie chart
- Interactive Folium maps (embedded)
- AI anomaly detection results and score distribution
- KMeans cluster analysis
- AI-generated optimisation report with download button

---

## 💼 CV / Portfolio Notes

This project demonstrates:
- **Drive Testing** — KPI parsing, threshold classification, coverage analysis
- **RF Optimisation** — worst cell identification, fault detection, recommendations
- **Data Engineering** — pandas pipelines, data cleaning, feature engineering
- **Machine Learning** — unsupervised anomaly detection, clustering
- **Cloud-Ready** — easily deployable on Oracle Cloud (free) or Streamlit Cloud (free)
- **AI Integration** — LLM for natural language report generation

---

## 📄 Licence

MIT — free to use, modify, and include in your portfolio.
