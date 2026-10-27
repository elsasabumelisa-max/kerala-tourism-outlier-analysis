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


from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler


"""
src/evaluation.py
-----------------
Kerala Tourism Trend Outlier Analysis Using Outlier Detection
Role / Responsibility: Nifa (Second-Algorithm Implementation, Evaluation, Comparison)

This module implements:
1. Data loading and feature inspection for Kerala Tourism processed datasets.
2. Distance-aware preprocessing (missing value imputation and feature scaling)
   essential for Local Outlier Factor (LOF).
3. Configurable Local Outlier Factor (LOF) algorithm implementation.
4. Result serialization to `results/lof_results.csv`.
5. Comparison logic between LOF and Isolation Forest (Melisa's output) with
   contingency summaries, Jaccard similarity, and overlap quantification.
6. Clear error handling and messaging when partner dependencies are missing.

IMPORTANT METHODOLOGICAL NOTE:
Because this is an unsupervised anomaly detection task without verified ground-truth
labels, classification metrics such as Accuracy, Precision, Recall, and F1-score
MUST NOT be reported. Treating one unsupervised algorithm as ground truth for another
is methodologically invalid. Agreement and disagreement between LOF and Isolation Forest
reflect differing algorithmic definitions of "anomaly" (local density vs. tree partition
isolation), not error.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler


# =====================================================================
# 1. DATA LOADING & FEATURE INSPECTION
# =====================================================================

def load_processed_tourism_data(
    filepath: str = "data/processed_tourism.csv",
) -> pd.DataFrame:
    """
    Loads the cleaned tourism dataset generated by Gauri's preprocessing module.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the cleaned CSV file.
        Defaults to 'data/processed_tourism.csv'.

    Returns
    -------
    pd.DataFrame
        The loaded tourism dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist, providing explicit guidance for team integration.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[PREREQUISITE MISSING] Cleaned tourism dataset not found at: '{filepath}'\n"
            "Gauri is responsible for data preprocessing. Please ensure that:\n"
            "  1. Gauri has run 'preprocessing.py' or generated the cleaned dataset.\n"
            "  2. The cleaned file is saved at 'data/processed_tourism.csv'.\n"
            "  3. If stored elsewhere, provide the path via the 'filepath' argument.\n"
        )

    df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded cleaned tourism dataset from: {filepath}")
    print(f"[INFO] Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def inspect_and_select_features(
    df: pd.DataFrame,
    candidate_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Inspects columns in the dataset and selects suitable numerical features
    for density-based outlier detection (LOF).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    candidate_cols : list of str, optional
        Explicit list of columns to select if present. If None, all suitable
        numerical columns are detected automatically.
    exclude_cols : list of str, optional
        Columns to explicitly exclude from feature selection (e.g., ID columns,
        year/month index columns, raw text, or target columns).

    Returns
    -------
    list of str
        List of selected numerical column names.
    """
    if exclude_cols is None:
        exclude_cols = [
            "id", "index", "unnamed: 0", "sr_no", "s_no", "district", "month",
            "state", "destination", "category"
        ]
    exclude_lower = {c.strip().lower() for c in exclude_cols}

    if candidate_cols is not None:
        # User specified columns: verify existence
        missing = [c for c in candidate_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Specified candidate columns not found in DataFrame: {missing}")
        selected = candidate_cols
    else:
        # Automatically detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected = [c for c in numeric_cols if c.strip().lower() not in exclude_lower]

    if not selected:
        raise ValueError(
            "No suitable numerical features found in dataset for LOF outlier detection. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Selected {len(selected)} numerical features for LOF analysis:")
    for col in selected:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        print(f"       - '{col}' (dtype: {dtype}, missing: {nulls})")

    return selected


# =====================================================================
# 2. PREPROCESSING FOR LOCAL OUTLIER FACTOR (LOF)
# =====================================================================

def preprocess_features_for_lof(
    df: pd.DataFrame,
    feature_cols: List[str],
    handle_missing: str = "median",
    scaler_type: str = "standard",
) -> Tuple[np.ndarray, SimpleImputer, Union[StandardScaler, RobustScaler]]:
    """
    Prepares numerical features specifically for Local Outlier Factor (LOF).

    Why this preprocessing is necessary for LOF:
    1. Distance Sensitivity: LOF relies on Euclidean or Minkowski distance between
       data points and their k-nearest neighbors. If one feature (e.g., total tourist count
       in hundreds of thousands) has a much larger variance/scale than another (e.g.,
       domestic-to-foreign tourist ratio or monthly growth rate in single digits),
       the unscaled distance will be dominated entirely by the larger-magnitude feature.
       Scaling (StandardScaler or RobustScaler) brings features to an equal footing.
    2. Missing Values: Distance calculations in scikit-learn LOF will raise a ValueError
       if NaN values are present. Missing values must be systematically imputed.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    feature_cols : list of str
        The numerical columns to use.
    handle_missing : str, default='median'
        Strategy for missing value imputation: 'median', 'mean', or 'most_frequent'.
    scaler_type : str, default='standard'
        Type of scaling: 'standard' (StandardScaler) or 'robust' (RobustScaler).
        RobustScaler is less sensitive to extreme outliers when calculating center and scale.

    Returns
    -------
    X_scaled : np.ndarray
        Preprocessed and scaled 2D numpy array ready for LOF.
    imputer : SimpleImputer
        The fitted imputer instance.
    scaler : StandardScaler or RobustScaler
        The fitted scaler instance.
    """
    X_raw = df[feature_cols].copy()

    # Handle missing values
    imputer = SimpleImputer(strategy=handle_missing)
    X_imputed = imputer.fit_transform(X_raw)

    # Scale features
    if scaler_type.lower() == "robust":
        scaler = RobustScaler()
    elif scaler_type.lower() == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'standard' or 'robust'.")

    X_scaled = scaler.fit_transform(X_imputed)

    print(f"[INFO] Preprocessing complete for LOF:")
    print(f"       - Missing value imputation: strategy='{handle_missing}'")
    print(f"       - Feature scaling: '{scaler_type}' scaler applied across {len(feature_cols)} features.")
    print(f"       - Matrix shape: {X_scaled.shape}")

    return X_scaled, imputer, scaler


# =====================================================================
# 3. LOCAL OUTLIER FACTOR (LOF) IMPLEMENTATION
# =====================================================================

def fit_local_outlier_factor(
    X_scaled: np.ndarray,
    n_neighbors: int = 20,
    contamination: Union[float, str] = 0.05,
    metric: str = "minkowski",
    p: int = 2,
    **kwargs,
) -> Tuple[LocalOutlierFactor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits Local Outlier Factor (LOF) on the scaled feature matrix.

    Mathematical Basis:
    LOF compares the local density of an observation to that of its k-nearest neighbors:
    - k-distance of p: distance to its k-th nearest neighbor.
    - reachability distance(p, o) = max(k-distance(o), d(p, o)).
    - local reachability density (lrd) = inverse of average reachability distance.
    - LOF(p) = average ratio of lrd(neighbors) to lrd(p).
    If LOF ~ 1, density is comparable to neighbors (inlier).
    If LOF > 1 (e.g., > 1.5), density is significantly lower than neighbors (outlier).

    Parameters
    ----------
    X_scaled : np.ndarray
        2D numpy array of preprocessed features.
    n_neighbors : int, default=20
        Number of neighbors to consider for local density estimation.
        Standard recommendation: >= 10, typically 20 for stable local densities.
    contamination : float or 'auto', default=0.05
        The expected proportion of outliers in the data set (range (0, 0.5]).
        Used to determine the threshold on negative outlier factor.
    metric : str, default='minkowski'
        Distance metric to use.
    p : int, default=2
        Power parameter for the Minkowski metric (p=2 corresponds to Euclidean distance).
    **kwargs :
        Additional keyword arguments passed to sklearn's LocalOutlierFactor.

    Returns
    -------
    model : LocalOutlierFactor
        The fitted LOF model.
    is_outlier : np.ndarray
        Binary array where 1 = outlier, 0 = inlier.
    raw_prediction : np.ndarray
        Scikit-learn native predictions: -1 for outlier, 1 for inlier.
    lof_score : np.ndarray
        Intuitive positive LOF score (score > 1 means lower local density than neighbors).
        Computed as `-negative_outlier_factor_`.
    """
    if isinstance(contamination, float) and not (0.0 < contamination <= 0.5):
        raise ValueError("contamination must be in (0, 0.5] or 'auto'")

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric=metric,
        p=p,
        novelty=False,  # Unsupervised outlier detection on the training sample
        **kwargs,
    )

    # In unsupervised mode, fit_predict returns: 1 for inliers, -1 for outliers
    raw_pred = lof.fit_predict(X_scaled)
    is_outlier = (raw_pred == -1).astype(int)

    # Scikit-learn's negative_outlier_factor_ is negative (larger negative = more abnormal).
    # Positive LOF score: -negative_outlier_factor_ (values ~ 1 are normal, >> 1 are outliers).
    lof_score = -lof.negative_outlier_factor_

    n_outliers = int(is_outlier.sum())
    total = len(is_outlier)
    pct_outliers = (n_outliers / total) * 100

    print(f"[INFO] Fitted Local Outlier Factor (LOF):")
    print(f"       - Hyperparameters: n_neighbors={n_neighbors}, contamination={contamination}, metric='{metric}'")
    print(f"       - Detected {n_outliers} outliers out of {total} records ({pct_outliers:.2f}%).")
    print(f"       - LOF score range: min={lof_score.min():.3f}, median={np.median(lof_score):.3f}, max={lof_score.max():.3f}")

    return lof, is_outlier, raw_pred, lof_score


def run_lof_detection(
    data_path: str = "data/processed_tourism.csv",
    output_path: str = "results/lof_results.csv",
    feature_cols: Optional[List[str]] = None,
    n_neighbors: int = 20,
    contamination: Union[float, str] = 0.05,
    scaler_type: str = "standard",
) -> pd.DataFrame:
    """
    High-level end-to-end execution function for LOF anomaly detection.
    Reads data, inspects features, preprocesses, trains LOF, and saves results.

    Parameters
    ----------
    data_path : str
        Path to processed dataset from Gauri.
    output_path : str
        Path where LOF results CSV will be saved.
    feature_cols : list of str, optional
        Specific features to use; if None, suitable features are detected.
    n_neighbors : int, default=20
        Neighborhood size parameter for LOF.
    contamination : float, default=0.05
        Contamination rate parameter for LOF.
    scaler_type : str, default='standard'
        'standard' or 'robust'.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus LOF detection columns:
        'lof_prediction', 'is_lof_outlier', 'negative_outlier_factor', 'lof_score'.
    """
    df = load_processed_tourism_data(data_path)
    selected_features = inspect_and_select_features(df, candidate_cols=feature_cols)

    X_scaled, _, _ = preprocess_features_for_lof(
        df, selected_features, scaler_type=scaler_type
    )

    lof_model, is_outlier, raw_pred, lof_score = fit_local_outlier_factor(
        X_scaled, n_neighbors=n_neighbors, contamination=contamination
    )

    results_df = df.copy()
    results_df["lof_prediction"] = raw_pred
    results_df["is_lof_outlier"] = is_outlier
    results_df["negative_outlier_factor"] = lof_model.negative_outlier_factor_
    results_df["lof_score"] = lof_score

    # Save results
    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"[INFO] LOF anomaly detection results saved to: {output_path}")

    return results_df


# =====================================================================
# 4. ISOLATION FOREST RESULTS INGESTION & PREREQUISITE VALIDATION
# =====================================================================

def load_isolation_forest_results(
    filepath: str = "results/isolation_forest_results.csv",
) -> pd.DataFrame:
    """
    Loads Isolation Forest anomaly detection results generated by Melisa.

    If Melisa's output file is not found, provides a crystal-clear, informative
    message explaining which file is missing rather than fabricating data.

    Parameters
    ----------
    filepath : str
        Path to Isolation Forest results CSV.
        Defaults to 'results/isolation_forest_results.csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Isolation Forest detections.

    Raises
    ------
    FileNotFoundError
        When the file is not present, detailing missing prerequisite.
    """
    if not os.path.exists(filepath):
        separator = "=" * 70
        raise FileNotFoundError(
            f"\n{separator}\n"
            "[MISSING PREREQUISITE: MELISA'S ISOLATION FOREST RESULTS]\n"
            f"The Isolation Forest results file was not found at:\n"
            f"  --> '{filepath}'\n\n"
            "Melisa is responsible for implementing Isolation Forest in\n"
            "'src/outlier_detection.py' and generating 'results/isolation_forest_results.csv'.\n\n"
            "Action required:\n"
            "  1. Ensure Melisa runs her Isolation Forest detection script.\n"
            "  2. Confirm that 'results/isolation_forest_results.csv' is saved.\n"
            "  3. Once the file is present, re-run this comparison step.\n"
            "Per project guidelines, comparison cannot be performed without Melisa's real output.\n"
            f"{separator}\n"
        )

    if_df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded Isolation Forest results from: {filepath}")
    print(f"[INFO] Isolation Forest results shape: {if_df.shape[0]} rows, {if_df.shape[1]} columns.")
    return if_df


def identify_if_outlier_column(if_df: pd.DataFrame) -> str:
    """
    Automatically detects the binary outlier column in Melisa's Isolation Forest results.

    Checks common naming patterns: 'is_if_outlier', 'is_outlier', 'anomaly',
    'outlier', 'if_outlier', 'prediction', etc.
    """
    candidates = [
        "is_if_outlier", "is_outlier", "anomaly", "outlier", "if_outlier",
        "is_anomaly", "isolation_forest_outlier", "iforest_outlier"
    ]
    for col in candidates:
        if col in if_df.columns:
            return col

    # Check for prediction column with -1 / 1
    if "if_prediction" in if_df.columns:
        return "if_prediction"
    if "prediction" in if_df.columns:
        return "prediction"

    # Fallback to column search
    matching = [c for c in if_df.columns if "outlier" in c.lower() or "anomaly" in c.lower()]
    if matching:
        return matching[0]

    raise KeyError(
        f"Could not identify outlier indicator column in Melisa's results. "
        f"Available columns: {list(if_df.columns)}. "
        f"Expected one of: {candidates}"
    )


# =====================================================================
# 5. ALGORITHM COMPARISON & EVALUATION
# =====================================================================

def compare_outlier_algorithms(
    lof_df: pd.DataFrame,
    if_df: pd.DataFrame,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str, dict]]]:
    """
    Performs comparative evaluation between Local Outlier Factor (LOF) and
    Isolation Forest (IF) anomaly detections.

    METHODOLOGICAL NOTICE:
    - This function does NOT calculate Accuracy, Precision, Recall, or F1-score
      because no verified ground-truth anomaly labels exist.
    - Algorithm disagreement does NOT mean either algorithm is incorrect.
      * Isolation Forest detects global outliers by recursive tree partitioning;
        points in sparse regions require fewer random splits to isolate.
      * Local Outlier Factor detects density-based local outliers;
        points that deviate significantly from the density of their immediate
        neighborhood are flagged, even if they lie within a moderately populated region.
      * Points detected by BOTH represent high-confidence, consensus anomalies.
      * Points detected by ONLY ONE algorithm represent algorithm-specific anomaly types.

    Parameters
    ----------
    lof_df : pd.DataFrame
        DataFrame with LOF results (must include 'is_lof_outlier').
    if_df : pd.DataFrame
        DataFrame with Isolation Forest results.
    id_col : str, optional
        Unique row identifier to align datasets on. If None, aligns by index.

    Returns
    -------
    comparison_df : pd.DataFrame
        Merged DataFrame containing original features, both model detections,
        and a 'comparison_status' column ('Both', 'LOF Only', 'Isolation Forest Only', 'Inlier').
    summary : dict
        Dictionary containing counts, percentages, overlap metrics, and Jaccard similarity.
    """
    if "is_lof_outlier" not in lof_df.columns:
        raise KeyError("lof_df is missing required column 'is_lof_outlier'.")

    if_col = identify_if_outlier_column(if_df)

    # Standardize IF outlier flag to binary 0/1
    if_flag = if_df[if_col].copy()
    if set(if_flag.unique()).issubset({-1, 1}):
        # Scikit-learn raw: -1 is outlier, 1 is inlier
        if_binary = (if_flag == -1).astype(int)
    else:
        # Standard boolean or 0/1
        if_binary = if_flag.astype(int)

    # Align DataFrames
    if id_col is not None and id_col in lof_df.columns and id_col in if_df.columns:
        cols_to_add = [c for c in if_df.columns if c not in lof_df.columns]
        merged_df = lof_df.merge(if_df[[id_col] + cols_to_add], on=id_col, how="inner")
        merged_df["is_if_outlier"] = (
            (merged_df[if_col] == -1).astype(int)
            if set(merged_df[if_col].unique()).issubset({-1, 1})
            else merged_df[if_col].astype(int)
        )
    else:
        if len(lof_df) != len(if_df):
            raise ValueError(
                f"Row count mismatch between LOF results ({len(lof_df)}) and "
                f"Isolation Forest results ({len(if_df)}). Specify 'id_col' to align."
            )
        merged_df = lof_df.copy()
        for col in if_df.columns:
            if col not in merged_df.columns:
                merged_df[col] = if_df[col].values
        merged_df["is_if_outlier"] = if_binary.values

    # Categorize consensus status
    lof_flags = merged_df["is_lof_outlier"].astype(int)
    if_flags = merged_df["is_if_outlier"].astype(int)

    conditions = [
        (lof_flags == 1) & (if_flags == 1),
        (lof_flags == 1) & (if_flags == 0),
        (lof_flags == 0) & (if_flags == 1),
        (lof_flags == 0) & (if_flags == 0),
    ]
    choices = ["Both", "LOF Only", "Isolation Forest Only", "Inlier"]
    merged_df["comparison_status"] = np.select(conditions, choices, default="Unknown")

    # Metrics computation
    total_records = len(merged_df)
    n_lof = int((lof_flags == 1).sum())
    n_if = int((if_flags == 1).sum())
    n_both = int(((lof_flags == 1) & (if_flags == 1)).sum())
    n_lof_only = int(((lof_flags == 1) & (if_flags == 0)).sum())
    n_if_only = int(((lof_flags == 0) & (if_flags == 1)).sum())
    n_inliers = int(((lof_flags == 0) & (if_flags == 0)).sum())
    n_union = n_both + n_lof_only + n_if_only

    jaccard_similarity = (n_both / n_union) if n_union > 0 else 0.0
    overall_concordance = ((n_both + n_inliers) / total_records) * 100

    summary = {
        "total_records": total_records,
        "lof_outliers_count": n_lof,
        "lof_outliers_pct": round((n_lof / total_records) * 100, 2),
        "if_outliers_count": n_if,
        "if_outliers_pct": round((n_if / total_records) * 100, 2),
        "both_outliers_count": n_both,
        "both_outliers_pct": round((n_both / total_records) * 100, 2),
        "lof_only_count": n_lof_only,
        "lof_only_pct": round((n_lof_only / total_records) * 100, 2),
        "if_only_count": n_if_only,
        "if_only_pct": round((n_if_only / total_records) * 100, 2),
        "inliers_count": n_inliers,
        "inliers_pct": round((n_inliers / total_records) * 100, 2),
        "jaccard_similarity": round(jaccard_similarity, 4),
        "concordance_rate_pct": round(overall_concordance, 2),
        "overlap_percentage_of_lof": round((n_both / n_lof * 100), 2) if n_lof > 0 else 0.0,
        "overlap_percentage_of_if": round((n_both / n_if * 100), 2) if n_if > 0 else 0.0,
        "contingency_matrix": {
            "LOF_Outlier_and_IF_Outlier": n_both,
            "LOF_Outlier_and_IF_Inlier": n_lof_only,
            "LOF_Inlier_and_IF_Outlier": n_if_only,
            "LOF_Inlier_and_IF_Inlier": n_inliers,
        },
        "evaluation_notes": (
            "Ground-truth anomaly labels are unavailable (unsupervised task). "
            "Accuracy, Precision, Recall, and F1-score are mathematically and "
            "methodologically invalid in this setting. Disagreement reflects "
            "differing algorithmic mechanisms: LOF captures local density deviations, "
            "while Isolation Forest captures global tree-path isolation."
        ),
    }

    print("\n" + "=" * 65)
    print("      OUTLIER DETECTION COMPARISON SUMMARY (LOF vs. IF)")
    print("=" * 65)
    print(f"Total Records Analyzed:                {total_records}")
    print(f"LOF Outliers Detected:                 {n_lof} ({summary['lof_outliers_pct']}%)")
    print(f"Isolation Forest Outliers Detected:    {n_if} ({summary['if_outliers_pct']}%)")
    print("-" * 65)
    print(f"Mutual Consensus Outliers (Both):      {n_both} ({summary['both_outliers_pct']}%)")
    print(f"Detected by LOF Only:                  {n_lof_only} ({summary['lof_only_pct']}%)")
    print(f"Detected by Isolation Forest Only:     {n_if_only} ({summary['if_only_pct']}%)")
    print(f"Agreed Inliers (Neither):              {n_inliers} ({summary['inliers_pct']}%)")
    print("-" * 65)
    print(f"Jaccard Similarity Index:              {summary['jaccard_similarity']}")
    print(f"Overall Concordance Rate:              {summary['concordance_rate_pct']}%")
    print(f"LOF Outliers also flagged by IF:       {summary['overlap_percentage_of_lof']}%")
    print(f"IF Outliers also flagged by LOF:       {summary['overlap_percentage_of_if']}%")
    print("=" * 65 + "\n")

    return merged_df, summary


from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler


"""
src/evaluation.py
-----------------
Kerala Tourism Trend Outlier Analysis Using Outlier Detection
Role / Responsibility: Nifa (Second-Algorithm Implementation, Evaluation, Comparison)

This module implements:
1. Data loading and feature inspection for Kerala Tourism processed datasets.
2. Distance-aware preprocessing (missing value imputation and feature scaling)
   essential for Local Outlier Factor (LOF).
3. Configurable Local Outlier Factor (LOF) algorithm implementation.
4. Result serialization to `results/lof_results.csv`.
5. Comparison logic between LOF and Isolation Forest (Melisa's output) with
   contingency summaries, Jaccard similarity, and overlap quantification.
6. Clear error handling and messaging when partner dependencies are missing.

IMPORTANT METHODOLOGICAL NOTE:
Because this is an unsupervised anomaly detection task without verified ground-truth
labels, classification metrics such as Accuracy, Precision, Recall, and F1-score
MUST NOT be reported. Treating one unsupervised algorithm as ground truth for another
is methodologically invalid. Agreement and disagreement between LOF and Isolation Forest
reflect differing algorithmic definitions of "anomaly" (local density vs. tree partition
isolation), not error.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler


# =====================================================================
# 1. DATA LOADING & FEATURE INSPECTION
# =====================================================================

def load_processed_tourism_data(
    filepath: str = "data/processed_tourism.csv",
) -> pd.DataFrame:
    """
    Loads the cleaned tourism dataset generated by Gauri's preprocessing module.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to the cleaned CSV file.
        Defaults to 'data/processed_tourism.csv'.

    Returns
    -------
    pd.DataFrame
        The loaded tourism dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist, providing explicit guidance for team integration.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[PREREQUISITE MISSING] Cleaned tourism dataset not found at: '{filepath}'\n"
            "Gauri is responsible for data preprocessing. Please ensure that:\n"
            "  1. Gauri has run 'preprocessing.py' or generated the cleaned dataset.\n"
            "  2. The cleaned file is saved at 'data/processed_tourism.csv'.\n"
            "  3. If stored elsewhere, provide the path via the 'filepath' argument.\n"
        )

    df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded cleaned tourism dataset from: {filepath}")
    print(f"[INFO] Dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def inspect_and_select_features(
    df: pd.DataFrame,
    candidate_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Inspects columns in the dataset and selects suitable numerical features
    for density-based outlier detection (LOF).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    candidate_cols : list of str, optional
        Explicit list of columns to select if present. If None, all suitable
        numerical columns are detected automatically.
    exclude_cols : list of str, optional
        Columns to explicitly exclude from feature selection (e.g., ID columns,
        year/month index columns, raw text, or target columns).

    Returns
    -------
    list of str
        List of selected numerical column names.
    """
    if exclude_cols is None:
        exclude_cols = [
            "id", "index", "unnamed: 0", "sr_no", "s_no", "district", "month",
            "year", "date", "ground_truth_anomaly", "anomaly_reason",
            "is_outlier", "anomaly_score", "state", "destination", "category"
        ]
    exclude_lower = {c.strip().lower() for c in exclude_cols}

    if candidate_cols is not None:
        # User specified columns: verify existence
        missing = [c for c in candidate_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Specified candidate columns not found in DataFrame: {missing}")
        selected = candidate_cols
    else:
        # Automatically detect numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected = [c for c in numeric_cols if c.strip().lower() not in exclude_lower]

    if not selected:
        raise ValueError(
            "No suitable numerical features found in dataset for LOF outlier detection. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Selected {len(selected)} numerical features for LOF analysis:")
    for col in selected:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        print(f"       - '{col}' (dtype: {dtype}, missing: {nulls})")

    return selected


# =====================================================================
# 2. PREPROCESSING FOR LOCAL OUTLIER FACTOR (LOF)
# =====================================================================

def preprocess_features_for_lof(
    df: pd.DataFrame,
    feature_cols: List[str],
    handle_missing: str = "median",
    scaler_type: str = "standard",
) -> Tuple[np.ndarray, SimpleImputer, Union[StandardScaler, RobustScaler]]:
    """
    Prepares numerical features specifically for Local Outlier Factor (LOF).

    Why this preprocessing is necessary for LOF:
    1. Distance Sensitivity: LOF relies on Euclidean or Minkowski distance between
       data points and their k-nearest neighbors. If one feature (e.g., total tourist count
       in hundreds of thousands) has a much larger variance/scale than another (e.g.,
       domestic-to-foreign tourist ratio or monthly growth rate in single digits),
       the unscaled distance will be dominated entirely by the larger-magnitude feature.
       Scaling (StandardScaler or RobustScaler) brings features to an equal footing.
    2. Missing Values: Distance calculations in scikit-learn LOF will raise a ValueError
       if NaN values are present. Missing values must be systematically imputed.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    feature_cols : list of str
        The numerical columns to use.
    handle_missing : str, default='median'
        Strategy for missing value imputation: 'median', 'mean', or 'most_frequent'.
    scaler_type : str, default='standard'
        Type of scaling: 'standard' (StandardScaler) or 'robust' (RobustScaler).
        RobustScaler is less sensitive to extreme outliers when calculating center and scale.

    Returns
    -------
    X_scaled : np.ndarray
        Preprocessed and scaled 2D numpy array ready for LOF.
    imputer : SimpleImputer
        The fitted imputer instance.
    scaler : StandardScaler or RobustScaler
        The fitted scaler instance.
    """
    X_raw = df[feature_cols].copy()

    # Handle missing values
    imputer = SimpleImputer(strategy=handle_missing)
    X_imputed = imputer.fit_transform(X_raw)

    # Scale features
    if scaler_type.lower() == "robust":
        scaler = RobustScaler()
    elif scaler_type.lower() == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'standard' or 'robust'.")

    X_scaled = scaler.fit_transform(X_imputed)

    print(f"[INFO] Preprocessing complete for LOF:")
    print(f"       - Missing value imputation: strategy='{handle_missing}'")
    print(f"       - Feature scaling: '{scaler_type}' scaler applied across {len(feature_cols)} features.")
    print(f"       - Matrix shape: {X_scaled.shape}")

    return X_scaled, imputer, scaler


# =====================================================================
# 3. LOCAL OUTLIER FACTOR (LOF) IMPLEMENTATION
# =====================================================================

def fit_local_outlier_factor(
    X_scaled: np.ndarray,
    n_neighbors: int = 20,
    contamination: Union[float, str] = 0.05,
    metric: str = "minkowski",
    p: int = 2,
    **kwargs,
) -> Tuple[LocalOutlierFactor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits Local Outlier Factor (LOF) on the scaled feature matrix.

    Mathematical Basis:
    LOF compares the local density of an observation to that of its k-nearest neighbors:
    - k-distance of p: distance to its k-th nearest neighbor.
    - reachability distance(p, o) = max(k-distance(o), d(p, o)).
    - local reachability density (lrd) = inverse of average reachability distance.
    - LOF(p) = average ratio of lrd(neighbors) to lrd(p).
    If LOF ~ 1, density is comparable to neighbors (inlier).
    If LOF > 1 (e.g., > 1.5), density is significantly lower than neighbors (outlier).

    Parameters
    ----------
    X_scaled : np.ndarray
        2D numpy array of preprocessed features.
    n_neighbors : int, default=20
        Number of neighbors to consider for local density estimation.
        Standard recommendation: >= 10, typically 20 for stable local densities.
    contamination : float or 'auto', default=0.05
        The expected proportion of outliers in the data set (range (0, 0.5]).
        Used to determine the threshold on negative outlier factor.
    metric : str, default='minkowski'
        Distance metric to use.
    p : int, default=2
        Power parameter for the Minkowski metric (p=2 corresponds to Euclidean distance).
    **kwargs :
        Additional keyword arguments passed to sklearn's LocalOutlierFactor.

    Returns
    -------
    model : LocalOutlierFactor
        The fitted LOF model.
    is_outlier : np.ndarray
        Binary array where 1 = outlier, 0 = inlier.
    raw_prediction : np.ndarray
        Scikit-learn native predictions: -1 for outlier, 1 for inlier.
    lof_score : np.ndarray
        Intuitive positive LOF score (score > 1 means lower local density than neighbors).
        Computed as `-negative_outlier_factor_`.
    """
    if isinstance(contamination, float) and not (0.0 < contamination <= 0.5):
        raise ValueError("contamination must be in (0, 0.5] or 'auto'")

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        metric=metric,
        p=p,
        novelty=False,  # Unsupervised outlier detection on the training sample
        **kwargs,
    )

    # In unsupervised mode, fit_predict returns: 1 for inliers, -1 for outliers
    raw_pred = lof.fit_predict(X_scaled)
    is_outlier = (raw_pred == -1).astype(int)

    # Scikit-learn's negative_outlier_factor_ is negative (larger negative = more abnormal).
    # Positive LOF score: -negative_outlier_factor_ (values ~ 1 are normal, >> 1 are outliers).
    lof_score = -lof.negative_outlier_factor_

    n_outliers = int(is_outlier.sum())
    total = len(is_outlier)
    pct_outliers = (n_outliers / total) * 100

    print(f"[INFO] Fitted Local Outlier Factor (LOF):")
    print(f"       - Hyperparameters: n_neighbors={n_neighbors}, contamination={contamination}, metric='{metric}'")
    print(f"       - Detected {n_outliers} outliers out of {total} records ({pct_outliers:.2f}%).")
    print(f"       - LOF score range: min={lof_score.min():.3f}, median={np.median(lof_score):.3f}, max={lof_score.max():.3f}")

    return lof, is_outlier, raw_pred, lof_score


def run_lof_detection(
    data_path: str = "data/processed_tourism.csv",
    output_path: str = "results/lof_results.csv",
    feature_cols: Optional[List[str]] = None,
    n_neighbors: int = 20,
    contamination: Union[float, str] = 0.05,
    scaler_type: str = "standard",
) -> pd.DataFrame:
    """
    High-level end-to-end execution function for LOF anomaly detection.
    Reads data, inspects features, preprocesses, trains LOF, and saves results.

    Parameters
    ----------
    data_path : str
        Path to processed dataset from Gauri.
    output_path : str
        Path where LOF results CSV will be saved.
    feature_cols : list of str, optional
        Specific features to use; if None, suitable features are detected.
    n_neighbors : int, default=20
        Neighborhood size parameter for LOF.
    contamination : float, default=0.05
        Contamination rate parameter for LOF.
    scaler_type : str, default='standard'
        'standard' or 'robust'.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus LOF detection columns:
        'lof_prediction', 'is_lof_outlier', 'negative_outlier_factor', 'lof_score'.
    """
    df = load_processed_tourism_data(data_path)
    selected_features = inspect_and_select_features(df, candidate_cols=feature_cols)

    X_scaled, _, _ = preprocess_features_for_lof(
        df, selected_features, scaler_type=scaler_type
    )

    lof_model, is_outlier, raw_pred, lof_score = fit_local_outlier_factor(
        X_scaled, n_neighbors=n_neighbors, contamination=contamination
    )

    results_df = df.copy()
    results_df["lof_prediction"] = raw_pred
    results_df["is_lof_outlier"] = is_outlier
    results_df["negative_outlier_factor"] = lof_model.negative_outlier_factor_
    results_df["lof_score"] = lof_score

    # Save results
    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"[INFO] LOF anomaly detection results saved to: {output_path}")

    return results_df


# =====================================================================
# 4. ISOLATION FOREST RESULTS INGESTION & PREREQUISITE VALIDATION
# =====================================================================

def load_isolation_forest_results(
    filepath: str = "results/isolation_forest_results.csv",
) -> pd.DataFrame:
    """
    Loads Isolation Forest anomaly detection results generated by Melisa.

    If Melisa's output file is not found, provides a crystal-clear, informative
    message explaining which file is missing rather than fabricating data.

    Parameters
    ----------
    filepath : str
        Path to Isolation Forest results CSV.
        Defaults to 'results/isolation_forest_results.csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Isolation Forest detections.

    Raises
    ------
    FileNotFoundError
        When the file is not present, detailing missing prerequisite.
    """
    if not os.path.exists(filepath):
        separator = "=" * 70
        raise FileNotFoundError(
            f"\n{separator}\n"
            "[MISSING PREREQUISITE: MELISA'S ISOLATION FOREST RESULTS]\n"
            f"The Isolation Forest results file was not found at:\n"
            f"  --> '{filepath}'\n\n"
            "Melisa is responsible for implementing Isolation Forest in\n"
            "'src/outlier_detection.py' and generating 'results/isolation_forest_results.csv'.\n\n"
            "Action required:\n"
            "  1. Ensure Melisa runs her Isolation Forest detection script.\n"
            "  2. Confirm that 'results/isolation_forest_results.csv' is saved.\n"
            "  3. Once the file is present, re-run this comparison step.\n"
            "Per project guidelines, comparison cannot be performed without Melisa's real output.\n"
            f"{separator}\n"
        )

    if_df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded Isolation Forest results from: {filepath}")
    print(f"[INFO] Isolation Forest results shape: {if_df.shape[0]} rows, {if_df.shape[1]} columns.")
    return if_df


def identify_if_outlier_column(if_df: pd.DataFrame) -> str:
    """
    Automatically detects the binary outlier column in Melisa's Isolation Forest results.

    Checks common naming patterns: 'is_if_outlier', 'is_outlier', 'anomaly',
    'outlier', 'if_outlier', 'prediction', etc.
    """
    candidates = [
        "is_if_outlier", "is_outlier", "anomaly", "outlier", "if_outlier",
        "is_anomaly", "isolation_forest_outlier", "iforest_outlier"
    ]
    for col in candidates:
        if col in if_df.columns:
            return col

    # Check for prediction column with -1 / 1
    if "if_prediction" in if_df.columns:
        return "if_prediction"
    if "prediction" in if_df.columns:
        return "prediction"

    # Fallback to column search
    matching = [c for c in if_df.columns if "outlier" in c.lower() or "anomaly" in c.lower()]
    if matching:
        return matching[0]

    raise KeyError(
        f"Could not identify outlier indicator column in Melisa's results. "
        f"Available columns: {list(if_df.columns)}. "
        f"Expected one of: {candidates}"
    )


# =====================================================================
# 5. ALGORITHM COMPARISON & EVALUATION
# =====================================================================

def compare_outlier_algorithms(
    lof_df: pd.DataFrame,
    if_df: pd.DataFrame,
    id_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float, str, dict]]]:
    """
    Performs comparative evaluation between Local Outlier Factor (LOF) and
    Isolation Forest (IF) anomaly detections.

    METHODOLOGICAL NOTICE:
    - This function does NOT calculate Accuracy, Precision, Recall, or F1-score
      because no verified ground-truth anomaly labels exist.
    - Algorithm disagreement does NOT mean either algorithm is incorrect.
      * Isolation Forest detects global outliers by recursive tree partitioning;
        points in sparse regions require fewer random splits to isolate.
      * Local Outlier Factor detects density-based local outliers;
        points that deviate significantly from the density of their immediate
        neighborhood are flagged, even if they lie within a moderately populated region.
      * Points detected by BOTH represent high-confidence, consensus anomalies.
      * Points detected by ONLY ONE algorithm represent algorithm-specific anomaly types.

    Parameters
    ----------
    lof_df : pd.DataFrame
        DataFrame with LOF results (must include 'is_lof_outlier').
    if_df : pd.DataFrame
        DataFrame with Isolation Forest results.
    id_col : str, optional
        Unique row identifier to align datasets on. If None, aligns by index.

    Returns
    -------
    comparison_df : pd.DataFrame
        Merged DataFrame containing original features, both model detections,
        and a 'comparison_status' column ('Both', 'LOF Only', 'Isolation Forest Only', 'Inlier').
    summary : dict
        Dictionary containing counts, percentages, overlap metrics, and Jaccard similarity.
    """
    if "is_lof_outlier" not in lof_df.columns:
        raise KeyError("lof_df is missing required column 'is_lof_outlier'.")

    if_col = identify_if_outlier_column(if_df)

    # Standardize IF outlier flag to binary 0/1
    if_flag = if_df[if_col].copy()
    if set(if_flag.unique()).issubset({-1, 1}):
        # Scikit-learn raw: -1 is outlier, 1 is inlier
        if_binary = (if_flag == -1).astype(int)
    else:
        # Standard boolean or 0/1
        if_binary = if_flag.astype(int)

    # Align DataFrames
    if id_col is not None and id_col in lof_df.columns and id_col in if_df.columns:
        cols_to_add = [c for c in if_df.columns if c not in lof_df.columns]
        merged_df = lof_df.merge(if_df[[id_col] + cols_to_add], on=id_col, how="inner")
        merged_df["is_if_outlier"] = (
            (merged_df[if_col] == -1).astype(int)
            if set(merged_df[if_col].unique()).issubset({-1, 1})
            else merged_df[if_col].astype(int)
        )
    else:
        if len(lof_df) != len(if_df):
            raise ValueError(
                f"Row count mismatch between LOF results ({len(lof_df)}) and "
                f"Isolation Forest results ({len(if_df)}). Specify 'id_col' to align."
            )
        merged_df = lof_df.copy()
        for col in if_df.columns:
            if col not in merged_df.columns:
                merged_df[col] = if_df[col].values
        merged_df["is_if_outlier"] = if_binary.values

    # Categorize consensus status
    lof_flags = merged_df["is_lof_outlier"].astype(int)
    if_flags = merged_df["is_if_outlier"].astype(int)

    conditions = [
        (lof_flags == 1) & (if_flags == 1),
        (lof_flags == 1) & (if_flags == 0),
        (lof_flags == 0) & (if_flags == 1),
        (lof_flags == 0) & (if_flags == 0),
    ]
    choices = ["Both", "LOF Only", "Isolation Forest Only", "Inlier"]
    merged_df["comparison_status"] = np.select(conditions, choices, default="Unknown")

    # Metrics computation
    total_records = len(merged_df)
    n_lof = int((lof_flags == 1).sum())
    n_if = int((if_flags == 1).sum())
    n_both = int(((lof_flags == 1) & (if_flags == 1)).sum())
    n_lof_only = int(((lof_flags == 1) & (if_flags == 0)).sum())
    n_if_only = int(((lof_flags == 0) & (if_flags == 1)).sum())
    n_inliers = int(((lof_flags == 0) & (if_flags == 0)).sum())
    n_union = n_both + n_lof_only + n_if_only

    jaccard_similarity = (n_both / n_union) if n_union > 0 else 0.0
    overall_concordance = ((n_both + n_inliers) / total_records) * 100

    summary = {
        "total_records": total_records,
        "lof_outliers_count": n_lof,
        "lof_outliers_pct": round((n_lof / total_records) * 100, 2),
        "if_outliers_count": n_if,
        "if_outliers_pct": round((n_if / total_records) * 100, 2),
        "both_outliers_count": n_both,
        "both_outliers_pct": round((n_both / total_records) * 100, 2),
        "lof_only_count": n_lof_only,
        "lof_only_pct": round((n_lof_only / total_records) * 100, 2),
        "if_only_count": n_if_only,
        "if_only_pct": round((n_if_only / total_records) * 100, 2),
        "inliers_count": n_inliers,
        "inliers_pct": round((n_inliers / total_records) * 100, 2),
        "jaccard_similarity": round(jaccard_similarity, 4),
        "concordance_rate_pct": round(overall_concordance, 2),
        "overlap_percentage_of_lof": round((n_both / n_lof * 100), 2) if n_lof > 0 else 0.0,
        "overlap_percentage_of_if": round((n_both / n_if * 100), 2) if n_if > 0 else 0.0,
        "contingency_matrix": {
            "LOF_Outlier_and_IF_Outlier": n_both,
            "LOF_Outlier_and_IF_Inlier": n_lof_only,
            "LOF_Inlier_and_IF_Outlier": n_if_only,
            "LOF_Inlier_and_IF_Inlier": n_inliers,
        },
        "evaluation_notes": (
            "Ground-truth anomaly labels are unavailable (unsupervised task). "
            "Accuracy, Precision, Recall, and F1-score are mathematically and "
            "methodologically invalid in this setting. Disagreement reflects "
            "differing algorithmic mechanisms: LOF captures local density deviations, "
            "while Isolation Forest captures global tree-path isolation."
        ),
    }

    print("\n" + "=" * 65)
    print("      OUTLIER DETECTION COMPARISON SUMMARY (LOF vs. IF)")
    print("=" * 65)
    print(f"Total Records Analyzed:                {total_records}")
    print(f"LOF Outliers Detected:                 {n_lof} ({summary['lof_outliers_pct']}%)")
    print(f"Isolation Forest Outliers Detected:    {n_if} ({summary['if_outliers_pct']}%)")
    print("-" * 65)
    print(f"Mutual Consensus Outliers (Both):      {n_both} ({summary['both_outliers_pct']}%)")
    print(f"Detected by LOF Only:                  {n_lof_only} ({summary['lof_only_pct']}%)")
    print(f"Detected by Isolation Forest Only:     {n_if_only} ({summary['if_only_pct']}%)")
    print(f"Agreed Inliers (Neither):              {n_inliers} ({summary['inliers_pct']}%)")
    print("-" * 65)
    print(f"Jaccard Similarity Index:              {summary['jaccard_similarity']}")
    print(f"Overall Concordance Rate:              {summary['concordance_rate_pct']}%")
    print(f"LOF Outliers also flagged by IF:       {summary['overlap_percentage_of_lof']}%")
    print(f"IF Outliers also flagged by LOF:       {summary['overlap_percentage_of_if']}%")
    print("=" * 65 + "\n")

    return merged_df, summary
