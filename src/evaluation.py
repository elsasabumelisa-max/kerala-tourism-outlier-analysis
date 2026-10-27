"""
Evaluation Metrics & Anomaly Profiling for Kerala Tourism Data Mining
======================================================================
Provides comparative model evaluation, inter-algorithm consensus analysis,
district-level vulnerability profiling, and domain-informed anomaly categorization.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def compute_model_agreement(
    df: pd.DataFrame,
    model_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes pairwise Jaccard Similarity coefficients across outlier detection models.
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    if model_cols is None:
        model_cols = [
            "Outlier_ZScore",
            "Outlier_ModZScore",
            "Outlier_IQR_District",
            "Outlier_IsolationForest",
            "Outlier_LOF",
            "Outlier_OneClassSVM",
            "Outlier_EllipticEnvelope",
            "Ensemble_Outlier",
        ]

    available_cols = [col for col in model_cols if col in df.columns]
    n_models = len(available_cols)
    similarity_matrix = pd.DataFrame(index=available_cols, columns=available_cols, dtype=float)

    for i in range(n_models):
        for j in range(n_models):
            col_a = available_cols[i]
            col_b = available_cols[j]
            set_a = df[col_a] == 1
            set_b = df[col_b] == 1

            intersection = (set_a & set_b).sum()
            union = (set_a | set_b).sum()

            jaccard = (intersection / union) if union > 0 else 1.0
            similarity_matrix.loc[col_a, col_b] = round(jaccard, 3)

    return similarity_matrix


def categorize_anomalies(
    df: pd.DataFrame,
    outlier_col: str = "Ensemble_Outlier",
) -> pd.DataFrame:
    """
    Classifies detected anomalies into intuitive domain categories based on
    footfall deviations, foreign visitor shares, and occupancy shifts:
      - Severe Drop / Disaster Shock (e.g. 2018 Floods, Lockdowns)
      - Peak Surge / Festival Season (e.g. Onam, Thrissur Pooram)
      - Foreign Tourist Spike (e.g. Fort Kochi Biennale)
      - Reporting Discrepancy
    """
    data = df.copy()
    categories = []

    # Calculate district historical baseline median
    district_medians = data.groupby("District")["Total_Tourists"].transform("median")

    for idx, row in data.iterrows():
        if row.get(outlier_col, 0) == 0:
            categories.append("Normal")
            continue

        total = row.get("Total_Tourists", 0)
        median = district_medians.iloc[idx]
        foreign_share = row.get("Foreign_Share", 0)

        if total <= 10 or (median > 0 and total < 0.20 * median):
            categories.append("Severe Drop / Crisis Shock")
        elif median > 0 and total > 1.70 * median:
            if foreign_share > 0.15:
                categories.append("International Peak Surge")
            else:
                categories.append("Domestic Festival / Holiday Surge")
        elif row.get("Hotel_Occupancy_Rate", 50) < 20 and total > median:
            categories.append("Occupancy-Arrival Mismatch")
        else:
            categories.append("Moderate Deviation")

    data["Anomaly_Category"] = categories
    return data


def summarize_district_anomalies(
    df: pd.DataFrame,
    outlier_col: str = "Ensemble_Outlier",
) -> pd.DataFrame:
    """
    Summarizes anomaly statistics per Kerala district:
      - Total records analyzed
      - Total detected anomalies
      - Anomaly prevalence rate (%)
      - Frequency of severe drops vs peak surges
    """
    working_df = df.copy()
    if "Anomaly_Category" not in working_df.columns:
        working_df = categorize_anomalies(working_df, outlier_col=outlier_col)

    records = []
    for district, group in working_df.groupby("District"):
        total_records = len(group)
        anomalies = group[group[outlier_col] == 1]
        n_anomalies = len(anomalies)
        rate = round((n_anomalies / total_records) * 100, 2) if total_records > 0 else 0.0

        drops = (anomalies["Anomaly_Category"] == "Severe Drop / Crisis Shock").sum()
        surges = anomalies["Anomaly_Category"].str.contains("Surge", case=False).sum()

        avg_loss_or_gain = round(
            anomalies["Total_Tourists"].mean() - group["Total_Tourists"].median(), 2
        ) if n_anomalies > 0 else 0.0

        records.append(
            {
                "District": district,
                "Total_Records": total_records,
                "Detected_Anomalies": n_anomalies,
                "Anomaly_Rate_Pct": rate,
                "Drop_Shocks": drops,
                "Surge_Peaks": surges,
                "Avg_Volume_Deviation": avg_loss_or_gain,
            }
        )

    summary_df = pd.DataFrame(records).sort_values(by="Anomaly_Rate_Pct", ascending=False)
    return summary_df.reset_index(drop=True)


def evaluate_outliers(
    df: pd.DataFrame,
    ground_truth_col: Optional[str] = "Ground_Truth_Anomaly",
    model_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates detection performance:
      - Precision, Recall, F1, Balanced Accuracy (if ground truth labels available)
      - Profiling statistics for Inliers vs Outliers across key indicators
    """
    if model_cols is None:
        model_cols = [
            "Outlier_ZScore",
            "Outlier_ModZScore",
            "Outlier_IQR_District",
            "Outlier_IsolationForest",
            "Outlier_LOF",
            "Outlier_OneClassSVM",
            "Outlier_EllipticEnvelope",
            "Ensemble_Outlier",
        ]

    available_models = [m for m in model_cols if m in df.columns]
    results: Dict[str, Any] = {"Model_Metrics": {}, "Profile_Comparison": {}}

    # Benchmark metrics against ground truth if present
    if ground_truth_col and ground_truth_col in df.columns:
        y_true = df[ground_truth_col].values
        metrics_list = []

        for model in available_models:
            y_pred = df[model].values
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            bacc = balanced_accuracy_score(y_true, y_pred)

            metrics_list.append(
                {
                    "Model": model.replace("Outlier_", ""),
                    "Precision": round(p, 4),
                    "Recall": round(r, 4),
                    "F1_Score": round(f1, 4),
                    "Balanced_Accuracy": round(bacc, 4),
                    "Total_Flagged": int(y_pred.sum()),
                }
            )

        results["Model_Metrics"] = pd.DataFrame(metrics_list).sort_values(
            by="F1_Score", ascending=False
        ).reset_index(drop=True)

    # Inlier vs Outlier Statistical Profile
    target_outlier_col = "Ensemble_Outlier" if "Ensemble_Outlier" in df.columns else available_models[0]
    inliers = df[df[target_outlier_col] == 0]
    outliers = df[df[target_outlier_col] == 1]

    profile_features = [
        "Domestic_Tourists",
        "Foreign_Tourists",
        "Total_Tourists",
        "Hotel_Occupancy_Rate",
        "Tourism_Revenue_Cr",
        "Average_Stay_Days",
    ]
    profile_features = [f for f in profile_features if f in df.columns]

    comparison = []
    for feat in profile_features:
        in_mean = inliers[feat].mean()
        out_mean = outliers[feat].mean()
        in_std = inliers[feat].std()
        out_std = outliers[feat].std()
        diff_pct = ((out_mean - in_mean) / in_mean * 100) if in_mean != 0 else 0.0

        comparison.append(
            {
                "Feature": feat,
                "Inlier_Mean": round(in_mean, 2),
                "Inlier_Std": round(in_std, 2),
                "Outlier_Mean": round(out_mean, 2),
                "Outlier_Std": round(out_std, 2),
                "Mean_Shift_Pct": round(diff_pct, 2),
            }
        )

    results["Profile_Comparison"] = pd.DataFrame(comparison)
    return results
