"""
Outlier Detection Algorithms for Kerala Tourism Data Mining
============================================================
Implements statistical, distance-based, density-based, and tree-based
anomaly detection models alongside an ensemble voting mechanism.
"""

import warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN


class StatisticalOutlierDetector:
    """
    Univariate and group-wise statistical outlier detectors:
      - Z-Score Thresholding
      - Modified Z-Score (Median Absolute Deviation - MAD)
      - Interquartile Range (IQR / Tukey's Fences)
    """

    @staticmethod
    def detect_zscore(
        series: pd.Series,
        threshold: float = 3.0,
    ) -> pd.Series:
        """
        Detects outliers where |Z| > threshold.
        Z = (X - mean) / std
        """
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        z_scores = (series - mean).abs() / std
        return (z_scores > threshold).astype(int)

    @staticmethod
    def detect_modified_zscore(
        series: pd.Series,
        threshold: float = 3.5,
    ) -> pd.Series:
        """
        Detects outliers using Median Absolute Deviation (MAD):
        ModZ = 0.6745 * |X - median| / MAD
        More robust than standard Z-score against extreme skew and disaster drops.
        """
        med = series.median()
        mad = np.median(np.abs(series - med))
        if mad == 0:
            return pd.Series(0, index=series.index)
        mod_z = 0.6745 * (series - med).abs() / mad
        return (mod_z > threshold).astype(int)

    @staticmethod
    def detect_iqr(
        series: pd.Series,
        multiplier: float = 1.5,
    ) -> pd.Series:
        """
        Tukey's Fences: Outliers lie outside [Q1 - k*IQR, Q3 + k*IQR].
        """
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - multiplier * iqr
        upper_bound = q75 + multiplier * iqr
        return ((series < lower_bound) | (series > upper_bound)).astype(int)

    @classmethod
    def apply_district_group_iqr(
        cls,
        df: pd.DataFrame,
        target_col: str = "Total_Tourists",
        group_col: str = "District",
        multiplier: float = 1.5,
    ) -> pd.Series:
        """
        Computes IQR fences per district independently to account for
        heterogeneous visitor capacities across districts (e.g. Ernakulam vs Kasaragod).
        """
        return df.groupby(group_col)[target_col].transform(
            lambda s: cls.detect_iqr(s, multiplier=multiplier)
        ).astype(int)


class MLOutlierDetector:
    """
    Multivariate Machine Learning Outlier Detectors:
      - Isolation Forest (iForest)
      - Local Outlier Factor (LOF)
      - One-Class Support Vector Machine (OC-SVM)
      - Robust Covariance / Minimum Covariance Determinant (Elliptic Envelope)
      - DBSCAN Density Clustering
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

    def fit_isolation_forest(
        self,
        X: pd.DataFrame,
        n_estimators: int = 150,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Isolation Forest isolates observations by randomly selecting a feature
        and splitting value. Anomalies require fewer splits.
        Returns (predictions_binary, anomaly_scores).
        """
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        preds = model.fit_predict(X)
        self.models["Isolation_Forest"] = model
        # scikit-learn returns -1 for outlier, 1 for inlier
        binary_outliers = (preds == -1).astype(int)
        scores = -model.score_samples(X)  # Higher score = more anomalous
        return binary_outliers, scores

    def fit_lof(
        self,
        X: pd.DataFrame,
        n_neighbors: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Local Outlier Factor measures the local deviation of density
        with respect to its k-nearest neighbors.
        """
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )
        preds = model.fit_predict(X)
        self.models["LOF"] = model
        binary_outliers = (preds == -1).astype(int)
        scores = -model.negative_outlier_factor_  # Higher factor = more anomalous
        return binary_outliers, scores

    def fit_one_class_svm(
        self,
        X: pd.DataFrame,
        kernel: str = "rbf",
        gamma: str = "scale",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-Class SVM fits a hyper-plane separating data points from the origin.
        """
        model = OneClassSVM(
            nu=self.contamination,
            kernel=kernel,
            gamma=gamma,
        )
        preds = model.fit_predict(X)
        self.models["OneClass_SVM"] = model
        binary_outliers = (preds == -1).astype(int)
        scores = -model.score_samples(X)
        return binary_outliers, scores

    def fit_elliptic_envelope(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Elliptic Envelope assumes data is Gaussian distributed and fits
        an ellipse based on Fast-MCD (Minimum Covariance Determinant).
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                model = EllipticEnvelope(
                    contamination=self.contamination,
                    random_state=self.random_state,
                    support_fraction=None,
                )
                preds = model.fit_predict(X)
                self.models["Elliptic_Envelope"] = model
                binary_outliers = (preds == -1).astype(int)
                scores = model.mahalanobis(X)
                return binary_outliers, scores
        except Exception:
            # Fallback if covariance matrix is singular
            return np.zeros(len(X), dtype=int), np.zeros(len(X))

    def fit_dbscan(
        self,
        X: pd.DataFrame,
        eps: float = 1.8,
        min_samples: int = 5,
    ) -> np.ndarray:
        """
        DBSCAN density clustering: Points labeled as -1 are treated as noise/outliers.
        """
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X)
        self.models["DBSCAN"] = model
        binary_outliers = (labels == -1).astype(int)
        return binary_outliers


class EnsembleOutlierDetector:
    """
    Combines predictions from multiple individual outlier detection models
    using consensus voting and continuous anomaly confidence scoring.
    """

    def __init__(self, min_consensus_votes: int = 2):
        self.min_consensus_votes = min_consensus_votes

    def vote(
        self,
        pred_dict: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates consensus predictions and anomaly score.
        Args:
            pred_dict: Dict of model_name -> binary outlier array (1 = outlier, 0 = normal)

        Returns:
            consensus_outliers: Binary array where sum of votes >= min_consensus_votes
            anomaly_score: Average vote fraction (0.0 to 1.0)
        """
        vote_matrix = np.column_stack(list(pred_dict.values()))
        total_votes = np.sum(vote_matrix, axis=1)
        model_count = vote_matrix.shape[1]

        anomaly_score = total_votes / max(1, model_count)
        consensus_outliers = (total_votes >= self.min_consensus_votes).astype(int)

        return consensus_outliers, anomaly_score


def run_all_detectors(
    df: pd.DataFrame,
    scaled_features: pd.DataFrame,
    contamination: float = 0.05,
    min_consensus: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes all statistical and ML outlier detectors on Kerala tourism data.
    Attaches prediction columns and returns the enriched dataframe and model objects.
    """
    results_df = df.copy()

    # 1. Statistical Detectors
    stat = StatisticalOutlierDetector()
    results_df["Outlier_ZScore"] = stat.detect_zscore(results_df["Total_Tourists"])
    results_df["Outlier_ModZScore"] = stat.detect_modified_zscore(results_df["Total_Tourists"])
    results_df["Outlier_IQR_Global"] = stat.detect_iqr(results_df["Total_Tourists"])
    results_df["Outlier_IQR_District"] = stat.apply_district_group_iqr(
        results_df, target_col="Total_Tourists", group_col="District"
    )

    # 2. Machine Learning Detectors
    ml = MLOutlierDetector(contamination=contamination)

    iso_pred, iso_scores = ml.fit_isolation_forest(scaled_features)
    lof_pred, lof_scores = ml.fit_lof(scaled_features)
    ocsvm_pred, ocsvm_scores = ml.fit_one_class_svm(scaled_features)
    ee_pred, ee_scores = ml.fit_elliptic_envelope(scaled_features)
    dbscan_pred = ml.fit_dbscan(scaled_features)

    results_df["Outlier_IsolationForest"] = iso_pred
    results_df["Score_IsolationForest"] = iso_scores
    results_df["Outlier_LOF"] = lof_pred
    results_df["Score_LOF"] = lof_scores
    results_df["Outlier_OneClassSVM"] = ocsvm_pred
    results_df["Score_OneClassSVM"] = ocsvm_scores
    results_df["Outlier_EllipticEnvelope"] = ee_pred
    results_df["Score_EllipticEnvelope"] = ee_scores
    results_df["Outlier_DBSCAN"] = dbscan_pred

    # 3. Ensemble Consensus Voting
    candidate_models = {
        "IsolationForest": iso_pred,
        "LOF": lof_pred,
        "OneClassSVM": ocsvm_pred,
        "IQR_District": results_df["Outlier_IQR_District"].values,
        "ModZScore": results_df["Outlier_ModZScore"].values,
    }

    ensemble = EnsembleOutlierDetector(min_consensus_votes=min_consensus)
    consensus_flags, anomaly_scores = ensemble.vote(candidate_models)

    results_df["Ensemble_Outlier"] = consensus_flags
    results_df["Ensemble_Anomaly_Score"] = np.round(anomaly_scores, 3)

    return results_df, ml.models
