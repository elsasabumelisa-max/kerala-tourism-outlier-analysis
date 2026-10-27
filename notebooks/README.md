# Jupyter Notebooks Directory

This folder contains experimental, exploratory, and analytical Jupyter Notebooks for interactive data mining on Kerala tourism statistics.

## Recommended Notebook Workflow

1. **`01_exploratory_data_analysis.ipynb`**:
   - Inspect data distribution across all 14 Kerala districts.
   - Analyze seasonality indices, domestic vs. foreign tourist proportions.
   - Detect trends over years (monsoon dips, winter highs, post-COVID rebound).

2. **`02_outlier_detection_experiments.ipynb`**:
   - Interactive testing of statistical algorithms (IQR, Z-Score, Modified Z-score with MAD).
   - Machine learning anomaly detection with Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, and Elliptic Envelope.
   - Hyperparameter sensitivity tuning (contamination parameter, $k$-neighbors, kernel choice).

3. **`03_anomaly_interpretation_and_insights.ipynb`**:
   - Cluster anomalies by causal factor (natural disasters like the 2018 floods, pandemic lockdowns, festive surges like Onam, and data reporting irregularities).
   - Policy recommendations and capacity planning for the Department of Tourism.

## How to Run Notebooks
Make sure the environment dependencies are installed:
```bash
pip install -r requirements.txt
pip install jupyterlab
jupyter lab notebooks/
```
