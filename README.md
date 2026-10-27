# Kerala Tourism Outlier Analysis & Data Mining

A comprehensive data mining and anomaly detection framework designed to analyze tourist arrivals, hotel occupancies, and tourism revenue dynamics across all 14 districts of Kerala, India.

---

## Project Structure

```text
kerala-tourism-outlier-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── outlier_detection.py
│   ├── evaluation.py
│   └── visualization.py
│
├── results/
│   └── README.md
│
├── reports/
│   └── README.md
│
└── main.py
```

---

## Domain Overview

Kerala ("God's Own Country") exhibits distinct tourism patterns influenced by monsoons, hill station climates, backwaters, and vibrant cultural events. Outlier detection in this context isolates critical signals:

1. **Natural Disasters & Crisis Drops**:
   - **August 2018 Deluge**: Catastrophic floods causing sudden footfall collapse (-85% to -95%) in Idukki, Ernakulam, Alappuzha, and Wayanad.
   - **2020-2021 COVID-19 Pandemic**: Near-zero international arrivals and severe domestic restrictions.
2. **Festival & Super-Season Spikes**:
   - **Onam Carnival (August - September)**: Statewide domestic surge.
   - **Thrissur Pooram (April/May in Thrissur)**: Cultural festival demand multiplying baseline footfalls up to 2.4x.
   - **Winter / Biennale Surge (December - January)**: High-season peaks in Fort Kochi, Munnar, Kovalam, and Varkala.
3. **Data Quality Irregularities**:
   - Under-reporting, reporting lags, or duplicated batch records at local DTPC counters.

---

## Implemented Outlier Detection Algorithms

| Paradigm | Algorithm | Description |
| :--- | :--- | :--- |
| **Statistical** | **Z-Score** | Standard deviation boundary ($\|Z\| > 3.0$) |
| **Statistical** | **Modified Z-Score (MAD)** | Median Absolute Deviation, robust against masking effects |
| **Statistical** | **Tukey's IQR Fences** | Interquartile Range fences ($Q_1 - 1.5 \times IQR$, $Q_3 + 1.5 \times IQR$) evaluated per district |
| **Machine Learning** | **Isolation Forest** | Tree-partition depth isolating anomalous observations with fewer cuts |
| **Machine Learning** | **Local Outlier Factor (LOF)** | Density-based anomaly detection comparing local density against $k$-nearest neighbors |
| **Machine Learning** | **One-Class SVM** | Non-linear boundary estimation using RBF kernel |
| **Machine Learning** | **Elliptic Envelope (Fast-MCD)** | Robust Mahalanobis distance under Gaussian covariance assumption |
| **Machine Learning** | **DBSCAN** | Density-based spatial clustering identifying isolated noise points |
| **Ensemble** | **Consensus Voting** | Aggregates individual model predictions with continuous confidence scoring |

---

## Getting Started

### 1. Installation

Ensure Python 3.9+ is installed, then clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/kerala-tourism-outlier-analysis.git
cd kerala-tourism-outlier-analysis
pip install -r requirements.txt
```

### 2. Running the End-to-End Pipeline

Run `main.py` directly. If no dataset is found in `data/`, it automatically synthesizes a realistic, calibrated historical Kerala dataset (2017–2024):

```bash
python main.py --generate-sample
```

#### CLI Options:
```bash
python main.py \
  --data data/kerala_tourism_data.csv \
  --contamination 0.06 \
  --consensus 2 \
  --results-dir results \
  --reports-dir reports
```

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data` | `data/kerala_tourism_data.csv` | Path to CSV/Excel input file |
| `--generate-sample` | `False` | Synthesizes historical Kerala tourism dataset |
| `--contamination` | `0.06` | Expected anomaly proportion for ML algorithms |
| `--consensus` | `2` | Minimum votes required for ensemble outlier flag |
| `--results-dir` | `results` | Folder for saving tables and plots |
| `--reports-dir` | `reports` | Folder for executive summary report |

---

## Output Artifacts

Upon running `main.py`, the following artifacts are generated:

### Tabular Data (`results/`)
- **`detected_anomalies.csv`**: Full dataset annotated with model predictions, anomaly scores, and categorizations (`Severe Drop / Crisis Shock`, `Domestic Festival Surge`, `International Peak Surge`, etc.).
- **`district_anomaly_summary.csv`**: District-level vulnerability rankings, anomaly rates, and average deviations.
- **`model_agreement_matrix.csv`**: Pairwise Jaccard similarity coefficients between all algorithms.

### Visualizations (`results/`)
- **`timeseries_anomalies.png`**: Multi-district longitudinal arrival trajectories with anomaly points.
- **`multivariate_outliers_scatter.png`**: Domestic vs. foreign tourist volume and occupancy vs. revenue scatter plots.
- **`district_boxplots.png`**: Arrival distributions across all 14 Kerala districts displaying statistical fences.
- **`model_agreement_heatmap.png`**: Heatmap displaying inter-algorithm consensus.
- **`seasonal_anomaly_matrix.png`**: Month vs. District vulnerability heatmap.

### Executive Report (`reports/`)
- **`outlier_analysis_summary.md`**: Executive briefing for Department of Tourism decision-makers with strategic policy and capacity recommendations.

---

## Modular Programmatic Usage

You can also import and use the pipeline components in your own Python scripts or Jupyter Notebooks:

```python
from src.preprocessing import generate_kerala_tourism_dataset, preprocess_tourism_data
from src.outlier_detection import run_all_detectors
from src.evaluation import summarize_district_anomalies
from src.visualization import generate_all_visualizations

# 1. Load or generate data
df = generate_kerala_tourism_dataset(start_year=2018, end_year=2024)

# 2. Preprocess and engineer features
processed_df, scaled_features, scaler = preprocess_tourism_data(df)

# 3. Detect anomalies with all models
results_df, models = run_all_detectors(processed_df, scaled_features, contamination=0.05)

# 4. View district vulnerability
summary = summarize_district_anomalies(results_df)
print(summary.head())
```

---

## License
MIT License. Open-source for academic, governmental, and commercial data mining research.
