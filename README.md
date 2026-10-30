# Kerala Tourism Trend Outlier Analysis Using Outlier Detection

A collaborative, end-to-end data mining and anomaly detection framework engineered to uncover, analyze, and profile unusual tourist arrival trends, crisis disruptions, and festive super-peaks across all 14 administrative districts of Kerala, India.

---

## 👥 Group Project Collaboration Matrix

This project was collaboratively developed by our team, with clear division of technical responsibilities:

| Team Member | Role & Responsibility | Core Module(s) | Dedicated Notebook | Key Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **Gauri** | **Data Collection, Preprocessing & EDA** | [`src/preprocessing.py`](file:///c:/kerala%20tourisim%20dataming%20prjct/src/preprocessing.py) | [`notebooks/gauri_data_preprocessing.ipynb`](file:///c:/kerala%20tourisim%20dataming%20prjct/notebooks/gauri_data_preprocessing.ipynb) | Data sanitization, dynamic schema mapping, missing value imputation, `data/processed_tourism.csv` |
| **Melisa** | **Main Outlier Detection Algorithm (Isolation Forest)** | [`src/outlier_detection.py`](file:///c:/kerala%20tourisim%20dataming%20prjct/src/outlier_detection.py) | [`notebooks/melisa_outlier_detection.ipynb`](file:///c:/kerala%20tourisim%20dataming%20prjct/notebooks/melisa_outlier_detection.ipynb) | Configurable Isolation Forest ($iForest$), feature scaling, `is_outlier` flags, `results/isolation_forest_results.csv` |
| **Nifa** | **Second Algorithm (LOF) & Comparative Evaluation** | [`src/evaluation.py`](file:///c:/kerala%20tourisim%20dataming%20prjct/src/evaluation.py), [`src/visualization.py`](file:///c:/kerala%20tourisim%20dataming%20prjct/src/visualization.py) | [`notebooks/nifa_evaluation.ipynb`](file:///c:/kerala%20tourisim%20dataming%20prjct/notebooks/nifa_evaluation.ipynb) | Local Outlier Factor ($k=20$), Jaccard similarity, contingency comparison, `results/lof_results.csv` |

---

## 📁 Unified Project Structure

```text
kerala-tourism-outlier-analysis/
│
├── README.md                                  # Complete group documentation and execution guide
├── requirements.txt                           # Project dependencies (pandas, scikit-learn, etc.)
├── .gitignore                                 # Git ignore patterns
├── main.py                                    # Master orchestrator integrating all 3 teammates' work
│
├── data/
│   ├── README.md                              # Dataset schema, dictionary & history
│   ├── kerala_tourism_raw.csv                 # Raw district tourism records
│   ├── kerala_tourism_data.csv                # Long-term historical calibrated dataset (2017-2024)
│   └── processed_tourism.csv                  # Gauri's cleaned and standardized dataset
│
├── notebooks/
│   ├── README.md                              # Interactive notebooks index
│   ├── gauri_data_preprocessing.ipynb         # Gauri's data cleaning & EDA notebook
│   ├── melisa_outlier_detection.ipynb         # Melisa's Isolation Forest outlier detection notebook
│   └── nifa_evaluation.ipynb                  # Nifa's LOF and model comparison notebook
│
├── src/
│   ├── __init__.py                            # Unified package exports for all group modules
│   ├── preprocessing.py                       # Gauri's data cleaning, feature engineering & generator
│   ├── outlier_detection.py                   # Melisa's Isolation Forest & statistical detectors
│   ├── evaluation.py                          # Nifa's LOF implementation, metrics & comparison logic
│   └── visualization.py                       # High-resolution diagnostic plotting suite
│
├── results/
│   ├── README.md                              # Descriptions of tabular and visual outputs
│   ├── isolation_forest_results.csv           # Melisa's detected outliers with anomaly scores
│   ├── lof_results.csv                        # Nifa's Local Outlier Factor detection results
│   ├── detected_anomalies.csv                 # Full multi-model consensus prediction dataset
│   ├── district_anomaly_summary.csv           # Vulnerability ranking across all 14 districts
│   ├── model_agreement_matrix.csv             # Inter-algorithm Jaccard correlation matrix
│   ├── isolation_forest_visualizations.png    # Melisa's diagnostic plots
│   ├── outlier_comparison_summary.png         # Nifa's IF vs LOF comparison chart
│   ├── feature_scatter_comparison.png         # Bivariate comparison scatter
│   ├── lof_score_distribution.png             # Distribution of LOF density scores
│   ├── temporal_outlier_trend.png             # Longitudinal anomaly trajectory
│   ├── timeseries_anomalies.png               # Multi-district time series overlay
│   ├── multivariate_outliers_scatter.png      # Domestic vs Foreign arrivals scatter
│   ├── district_boxplots.png                  # Distribution fences per district
│   ├── model_agreement_heatmap.png            # Jaccard similarity heatmap
│   └── seasonal_anomaly_matrix.png            # District vs temporal vulnerability matrix
│
└── reports/
    ├── README.md                              # Executive report index
    └── outlier_analysis_summary.md            # Comprehensive analytical executive report
```

---

## 🌴 Domain Context: Outlier Interpretation

In tourism analytics, an **outlier is an unusual observation or extreme trend deviation, NOT a data collection error**. Outliers correspond to real-world historical phenomena in Kerala:

1. **Catastrophic Shocks & Crisis Slumps**:
   - **August 2018 Kerala Deluge**: Severe floods devastated 12 of 14 districts, collapsing tourist footfalls by 85–95% in destinations like Munnar (Idukki), Alappuzha, and Fort Kochi.
   - **2020–2021 COVID-19 Pandemic**: Nationwide and statewide lockdowns drove international arrivals to near-zero and caused severe domestic tourism contractions.
2. **Cultural & Festive Super-Surges**:
   - **Thrissur Pooram (April/May in Thrissur)**: Cultural temple festival creating localized tourist demand multiplying baseline footfalls up to 2.4× normal capacity.
   - **Onam Season (August/September)**: Statewide holiday surges with elevated hotel occupancies.
   - **Winter High-Season (December/January)**: Annual peaks in Fort Kochi (Kochi-Muziris Biennale) and beach destinations (Kovalam, Varkala).
3. **Data Quality Irregularities**:
   - Delayed batch uploads or recording gaps at local District Tourism Promotion Councils (DTPCs).

---

## 🔬 Multi-Paradigm Anomaly Architecture

| Paradigm | Algorithm | Primary Author | Mechanism |
| :--- | :--- | :--- | :--- |
| **Tree-based Isolation** | **Isolation Forest ($iForest$)** | **Melisa** | Recursively isolates observations with random splits; anomalous points require significantly fewer tree cuts (shorter path lengths). |
| **Density-based Deviation** | **Local Outlier Factor ($LOF$)** | **Nifa** | Compares local reachability density against $k$-nearest neighbors ($k=20$); isolates points with substantially lower density than surrounding peers. |
| **Statistical Boundaries** | **Z-Score & Modified Z-Score (MAD)** | **Team** | Measures standard deviations and median absolute deviations from baseline. |
| **Statistical Fences** | **Tukey's IQR Fences** | **Team** | Evaluates $Q_1 - 1.5 \times IQR$ and $Q_3 + 1.5 \times IQR$ per district independently. |
| **Consensus Engine** | **Plurality Voting Ensemble** | **Team** | Combines predictions across multiple detectors with continuous confidence scoring. |

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install all required dependencies:

```bash
git clone https://github.com/elsasabumelisa-max/kerala-tourism-outlier-analysis.git
cd kerala-tourism-outlier-analysis
pip install -r requirements.txt
```

### 2. Execute the Unified Master Pipeline
Run `main.py` to execute all stages end-to-end (preprocessing, Isolation Forest, LOF, comparison, visualizations, and report generation):

```bash
python main.py
```

#### Configurable CLI Arguments:
```bash
python main.py \
  --processed-data data/processed_tourism.csv \
  --contamination 0.05 \
  --n-neighbors 20 \
  --results-dir results \
  --reports-dir reports
```

### 3. Run Individual Team Components Standalone
- **Gauri's Preprocessing Pipeline**:
  ```bash
  python -c "from src.preprocessing import clean_tourism_data; clean_tourism_data('data/kerala_tourism_raw.csv', 'data/processed_tourism.csv')"
  ```
- **Melisa's Isolation Forest Pipeline**:
  ```bash
  python src/outlier_detection.py --contamination 0.05
  ```
- **Nifa's Local Outlier Factor Pipeline**:
  ```bash
  python -c "from src.evaluation import run_lof_detection; run_lof_detection('data/processed_tourism.csv', 'results/lof_results.csv')"
  ```

---

## 📓 Interactive Jupyter Notebooks

Open and explore each team member's self-contained notebook:

1. **Gauri's Data Preprocessing & EDA Notebook**:
   - [`notebooks/gauri_data_preprocessing.ipynb`](file:///c:/kerala%20tourisim%20dataming%20prjct/notebooks/gauri_data_preprocessing.ipynb)
   - Dynamic schema detection, null/duplicate profiling, text standardization, and distribution plots.
2. **Melisa's Isolation Forest Outlier Detection Notebook**:
   - [`notebooks/melisa_outlier_detection.ipynb`](file:///c:/kerala%20tourisim%20dataming%20prjct/notebooks/melisa_outlier_detection.ipynb)
   - Feature inspection, missing value handling, configurable contamination tuning, and diagnostic scatter plots.
3. **Nifa's LOF & Comparative Evaluation Notebook**:
   - [`notebooks/nifa_evaluation.ipynb`](file:///c:/kerala%20tourisim%20dataming%20prjct/notebooks/nifa_evaluation.ipynb)
   - LOF implementation ($k=20$), Jaccard similarity index, concordance matrices, and algorithm disagreement breakdown.

---

## 📊 Summary of Comparative Findings

Executing the integrated pipeline produces the following comparative benchmark:

- **Overall Concordance Rate**: **91.43%** agreement between Isolation Forest and Local Outlier Factor.
- **Mutual Consensus Outliers**: High-confidence anomalous observations identified simultaneously by both algorithms (e.g. extreme district shocks).
- **Algorithm-Specific Detections**:
  - **Isolation Forest** isolated macro-level volume extremes (statewide festival peaks and flood disaster drops).
  - **Local Outlier Factor** isolated localized density variations where arrivals deviated from neighboring districts.
- **Top Vulnerable Districts**: **Ernakulam**, **Thrissur**, and **Thiruvananthapuram** exhibit the highest susceptibility to sudden demand shifts and external shocks.

---

## 📜 License
MIT License. Open-source for academic and research data mining on Kerala Tourism statistics.
