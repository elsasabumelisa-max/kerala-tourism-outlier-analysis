"""
Kerala Tourism Trend Outlier Analysis Using Outlier Detection
=============================================================
Unified Data Mining & Anomaly Detection Package

Group Project Contributors & Module Roles:
------------------------------------------
1. Gauri:
   - Data Collection, Preprocessing & Exploratory Data Analysis
   - Modules: `src/preprocessing.py`, `notebooks/gauri_data_preprocessing.ipynb`
   - Inputs/Outputs: `data/kerala_tourism_raw.csv` -> `data/processed_tourism.csv`

2. Melisa:
   - Main Outlier Detection Algorithm (Isolation Forest)
   - Modules: `src/outlier_detection.py`, `notebooks/melisa_outlier_detection.ipynb`
   - Inputs/Outputs: `data/processed_tourism.csv` -> `results/isolation_forest_results.csv`

3. Nifa:
   - Comparative Model (Local Outlier Factor / LOF) & Evaluation Methodology
   - Modules: `src/evaluation.py`, `notebooks/nifa_evaluation.ipynb`, `src/visualization.py`
   - Inputs/Outputs: `data/processed_tourism.csv` + `results/isolation_forest_results.csv` -> `results/lof_results.csv`
"""

# Gauri's Data Preprocessing Module Exports
from .preprocessing import (
    DISTRICTS,
    CANONICAL_COLUMNS,
    FEATURE_COLUMNS,
    load_tourism_data,
    clean_tourism_data,
    preprocess_tourism_data,
    generate_kerala_tourism_dataset,
)

# Melisa's Isolation Forest Outlier Detection Module Exports
from .outlier_detection import (
    TourismIsolationForest,
    detect_outliers_isolation_forest,
    run_outlier_detection_pipeline,
    plot_isolation_forest_visualizations,
    StatisticalOutlierDetector,
    MLOutlierDetector,
    EnsembleOutlierDetector,
    run_all_detectors,
)

# Nifa's Evaluation, LOF & Comparison Module Exports
from .evaluation import (
    load_processed_tourism_data,
    inspect_and_select_features,
    preprocess_features_for_lof,
    fit_local_outlier_factor,
    run_lof_detection,
    load_isolation_forest_results,
    compare_outlier_algorithms,
    compute_model_agreement,
    categorize_anomalies,
    summarize_district_anomalies,
    evaluate_outliers,
)

# Unified Visualizations
from .visualization import (
    set_plot_style,
    plot_timeseries_outliers,
    plot_multivariate_scatter,
    plot_district_boxplots,
    plot_model_agreement,
    plot_seasonal_heatmap,
    generate_all_visualizations,
    plot_lof_score_distribution,
    plot_comparison_summary,
    plot_feature_scatter_comparison,
    plot_temporal_outlier_trend,
    plot_score_correlation,
)

__all__ = [
    # Gauri
    "DISTRICTS",
    "CANONICAL_COLUMNS",
    "FEATURE_COLUMNS",
    "load_tourism_data",
    "clean_tourism_data",
    "preprocess_tourism_data",
    "generate_kerala_tourism_dataset",
    # Melisa
    "TourismIsolationForest",
    "detect_outliers_isolation_forest",
    "run_outlier_detection_pipeline",
    "plot_isolation_forest_visualizations",
    "StatisticalOutlierDetector",
    "MLOutlierDetector",
    "EnsembleOutlierDetector",
    "run_all_detectors",
    # Nifa
    "load_processed_tourism_data",
    "inspect_and_select_features",
    "preprocess_features_for_lof",
    "fit_local_outlier_factor",
    "run_lof_detection",
    "load_isolation_forest_results",
    "compare_outlier_algorithms",
    "compute_model_agreement",
    "categorize_anomalies",
    "summarize_district_anomalies",
    "evaluate_outliers",
    # Visualizations
    "set_plot_style",
    "plot_timeseries_outliers",
    "plot_multivariate_scatter",
    "plot_district_boxplots",
    "plot_model_agreement",
    "plot_seasonal_heatmap",
    "generate_all_visualizations",
    "plot_lof_score_distribution",
    "plot_comparison_summary",
    "plot_feature_scatter_comparison",
    "plot_temporal_outlier_trend",
    "plot_score_correlation",
]
