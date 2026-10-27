"""
Kerala Tourism Outlier Analysis - Data Mining Package
======================================================
A modular framework for data preprocessing, multi-model outlier detection,
evaluation, and visualization tailored for Kerala Tourism arrival and revenue datasets.
"""

from .preprocessing import (
    load_tourism_data,
    preprocess_tourism_data,
    generate_kerala_tourism_dataset,
    DISTRICTS,
    FEATURE_COLUMNS,
)

from .outlier_detection import (
    StatisticalOutlierDetector,
    MLOutlierDetector,
    EnsembleOutlierDetector,
    run_all_detectors,
)

from .evaluation import (
    evaluate_outliers,
    summarize_district_anomalies,
    compute_model_agreement,
    categorize_anomalies,
)

from .visualization import (
    plot_timeseries_outliers,
    plot_multivariate_scatter,
    plot_district_boxplots,
    plot_model_agreement,
    plot_seasonal_heatmap,
    generate_all_visualizations,
)

__all__ = [
    "load_tourism_data",
    "preprocess_tourism_data",
    "generate_kerala_tourism_dataset",
    "DISTRICTS",
    "FEATURE_COLUMNS",
    "StatisticalOutlierDetector",
    "MLOutlierDetector",
    "EnsembleOutlierDetector",
    "run_all_detectors",
    "evaluate_outliers",
    "summarize_district_anomalies",
    "compute_model_agreement",
    "categorize_anomalies",
    "plot_timeseries_outliers",
    "plot_multivariate_scatter",
    "plot_district_boxplots",
    "plot_model_agreement",
    "plot_seasonal_heatmap",
    "generate_all_visualizations",
]
