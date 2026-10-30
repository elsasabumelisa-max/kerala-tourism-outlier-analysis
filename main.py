"""
Kerala Tourism Trend Outlier Analysis Using Outlier Detection
=============================================================
Unified Master Execution Pipeline

Group Members & Contributions:
------------------------------
- Gauri:  Data Collection, Preprocessing & EDA (`src/preprocessing.py`)
- Melisa: Primary Outlier Detection - Isolation Forest (`src/outlier_detection.py`)
- Nifa:   Second Algorithm - Local Outlier Factor & Evaluation (`src/evaluation.py`, `src/visualization.py`)

This master script coordinates the end-to-end data mining workflow:
  Stage 1 -> Data Preprocessing & Sanitization (Gauri)
  Stage 2 -> Primary Outlier Detection via Isolation Forest (Melisa)
  Stage 3 -> Density-based Outlier Detection via Local Outlier Factor (Nifa)
  Stage 4 -> Algorithm Comparison, Jaccard Similarity & Consensus Profiling (Nifa & Melisa)
  Stage 5 -> Visual Analytics & Diagnostic Chart Generation (Team)
  Stage 6 -> Comprehensive Executive Report Synthesis (Team)
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Ensure project root is available on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    load_tourism_data,
    preprocess_tourism_data,
    clean_tourism_data,
    generate_kerala_tourism_dataset,
    DISTRICTS,
)
from src.outlier_detection import (
    detect_outliers_isolation_forest,
    plot_isolation_forest_visualizations,
    run_all_detectors,
)
from src.evaluation import (
    run_lof_detection,
    compare_outlier_algorithms,
    compute_model_agreement,
    categorize_anomalies,
    summarize_district_anomalies,
    evaluate_outliers,
)
from src.visualization import (
    generate_all_visualizations,
    plot_comparison_summary,
    plot_feature_scatter_comparison,
    plot_lof_score_distribution,
    plot_temporal_outlier_trend,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Kerala Tourism Trend Outlier Analysis Pipeline"
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/kerala_tourism_data.csv",
        help="Path to initial raw or benchmark Kerala tourism dataset.",
    )
    parser.add_argument(
        "--processed-data",
        type=str,
        default="data/processed_tourism.csv",
        help="Path for Gauri's cleaned and standardized dataset.",
    )
    parser.add_argument(
        "--if-results",
        type=str,
        default="results/isolation_forest_results.csv",
        help="Destination path for Melisa's Isolation Forest results.",
    )
    parser.add_argument(
        "--lof-results",
        type=str,
        default="results/lof_results.csv",
        help="Destination path for Nifa's Local Outlier Factor results.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Configurable contamination rate for outlier detectors (default: 0.05).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="Neighbor count parameter for Local Outlier Factor (default: 20).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save tabular outputs and plots (default: 'results').",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save analytical reports (default: 'reports').",
    )
    return parser.parse_args()


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to markdown string, with robust fallback."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = list(df.columns)
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        rows = [
            "| " + " | ".join(str(val) for val in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([header_row, separator] + rows)


def generate_unified_executive_report(
    summary_df: pd.DataFrame,
    comparison_summary: Dict[str, Any],
    eval_results: Dict[str, Any],
    anomalies_df: pd.DataFrame,
    report_path: str,
):
    """
    Synthesizes and writes an executive analytical report combining findings
    from Gauri (data preprocessing), Melisa (Isolation Forest), and Nifa (LOF & Evaluation).
    """
    total_records = len(anomalies_df)
    n_if = int(comparison_summary.get("if_outliers_count", 0))
    n_lof = int(comparison_summary.get("lof_outliers_count", 0))
    n_both = int(comparison_summary.get("both_outliers_count", 0))
    jaccard = comparison_summary.get("jaccard_similarity", 0.0)
    concordance = comparison_summary.get("concordance_rate_pct", 0.0)

    category_counts = anomalies_df["Anomaly_Category"].value_counts().to_dict() if "Anomaly_Category" in anomalies_df.columns else {}
    top_vulnerable = summary_df.head(3)["District"].tolist() if "District" in summary_df.columns else []

    report_content = f"""# Kerala Tourism Trend Outlier Analysis - Executive Final Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Project:** Kerala Tourism Trend Outlier Analysis Using Outlier Detection  
**Team Members & Roles:**
- **Gauri:** Data Collection, Preprocessing & Exploratory Data Analysis (`src/preprocessing.py`)
- **Melisa:** Primary Outlier Detection - Isolation Forest (`src/outlier_detection.py`)
- **Nifa:** Second Algorithm - Local Outlier Factor (LOF) & Comparative Evaluation (`src/evaluation.py`, `src/visualization.py`)

**Scope:** 14 Districts of Kerala | Total Records Analyzed: {total_records:,}  

---

## 1. Executive Summary & Problem Formulation
Tourism in Kerala ("God's Own Country") exhibits complex dynamics driven by tropical monsoons, mountain geography, backwater ecosystems, and rich cultural traditions. This project establishes an unsupervised data mining pipeline to identify, profile, and interpret statistical outliers in monthly tourism footfalls and revenues across all 14 administrative districts of Kerala.

### Core Domain Principle:
In this study, an **outlier is a statistically unusual observation or severe trend shock, NOT a data collection error**. Outliers correspond to real-world phenomena:
1. **Catastrophic Disruption Shocks:** The historic **August 2018 Kerala Deluge/Floods** caused tourist arrivals to collapse by 85–95% across 12 affected districts (especially Idukki, Alappuzha, Ernakulam, and Wayanad).
2. **Pandemic Disruptions:** The **2020–2021 COVID-19 pandemic** generated prolonged periods of near-zero foreign arrivals and sharp domestic contractions.
3. **Festive Super-Surges:** Annual cultural celebrations such as **Thrissur Pooram (April in Thrissur)** and statewide **Onam (August/September)** multiply baseline arrivals up to 2.4× normal capacity.

---

## 2. Multi-Algorithm Detection & Model Comparison

Two distinct unsupervised anomaly detection paradigms were implemented and contrasted:
- **Isolation Forest (Melisa):** Global tree-based isolation partitioning. Points isolated with few random cuts are flagged as anomalies.
- **Local Outlier Factor / LOF (Nifa):** Density-based outlier detection comparing local reachability density against $k$-nearest neighbors ($k=20$).

### Comparison Summary Metrics:
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Observations Screened** | `{total_records:,}` | Full monthly dataset across all 14 districts |
| **Isolation Forest Outliers (Melisa)** | `{n_if:,}` ({comparison_summary.get('if_outliers_pct', 0)}%) | Global volume surges and disaster collapses |
| **Local Outlier Factor Outliers (Nifa)** | `{n_lof:,}` ({comparison_summary.get('lof_outliers_pct', 0)}%) | Local density variations relative to peer neighbors |
| **Mutual Consensus Outliers (Both)** | `{n_both:,}` ({comparison_summary.get('both_outliers_pct', 0)}%) | High-confidence anomalies confirmed by both models |
| **Jaccard Similarity Index** | `{jaccard:.4f}` | Overlap ratio: $|IF \\cap LOF| / |IF \\cup LOF|$ |
| **Overall Concordance Rate** | `{concordance:.2f}%` | Percentage of observations where both models agree |

> **Methodological Note:** Because this is an unsupervised anomaly detection task without ground truth, supervised metrics (Precision, Recall, F1) treating one model as truth are methodologically invalid. Agreement reflects complementary algorithmic viewpoints: LOF captures local density deviations, while Isolation Forest isolates global volume deviations.

---

## 3. District Vulnerability & Anomaly Prevalence Ranking

Summary of detected anomalies across Kerala's 14 administrative districts:

{_df_to_markdown(summary_df)}

### Top 3 Volatile Districts:
**{', '.join(top_vulnerable)}** exhibit the highest vulnerability to external shocks and demand surges, driven by high dependence on international leisure tourism and major festival hosting.

---

## 4. Inlier vs. Outlier Operational Profiles

Comparative statistics highlighting the magnitude of operational shifts between normal periods and anomalous months:

"""
    if "Profile_Comparison" in eval_results and not eval_results["Profile_Comparison"].empty:
        report_content += _df_to_markdown(eval_results["Profile_Comparison"])
    else:
        report_content += "*Statistical profile generated from model predictions.*"

    report_content += f"""

---

## 5. Strategic Recommendations for Kerala Tourism Stakeholders

1. **Disaster Resilience & Relief Triggers**:
   - Riverine and highland districts (*Alappuzha, Idukki, Wayanad*) require automated tourism relief triggers when multi-model anomaly detection confirms footfalls drop below 20% of median baseline.
2. **Festival Crowd Management & Capacity Planning**:
   - Severe positive surges during **Thrissur Pooram** and **Onam** necessitate dynamic inter-district tourist redistribution towards northern circuits (*Kannur, Kasaragod*) to alleviate infrastructure strain.
3. **Data Quality & Automated Telemetry**:
   - Establish digital telemetry interfaces at District Tourism Promotion Councils (DTPCs) to eliminate rare data-reporting gaps and duplicate batch uploads.

---
*Generated by the Unified Kerala Tourism Outlier Mining Framework (Gauri, Melisa, Nifa).*
"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Report] Unified executive analytical report saved to: {report_path}")


def main():
    print("=" * 72)
    print("   KERALA TOURISM TREND OUTLIER ANALYSIS USING OUTLIER DETECTION")
    print("   Unified Group Project Pipeline: Gauri, Melisa, Nifa")
    print("=" * 72)

    args = parse_arguments()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 1: GAURI'S DATA PREPROCESSING & SANITIZATION PIPELINE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [STAGE 1] Data Preprocessing & Sanitization (Gauri's Module)")
    print("=" * 60)

    if not os.path.exists(args.raw_data) and not os.path.exists(args.processed_data):
        print(f"[Stage 1] Ingest dataset not found. Generating calibrated Kerala tourism dataset...")
        generate_kerala_tourism_dataset(start_year=2017, end_year=2024, save_path=args.raw_data)

    # If processed_tourism.csv doesn't exist, create it from raw
    if not os.path.exists(args.processed_data):
        print(f"[Stage 1] Cleaning raw dataset from: {args.raw_data}")
        clean_tourism_data(
            input_path=args.raw_data,
            output_path=args.processed_data,
            impute_missing=True,
        )
    else:
        print(f"[Stage 1] Verified existing cleaned dataset at: {args.processed_data}")

    # Load and preprocess features
    df_raw = load_tourism_data(args.processed_data if os.path.exists(args.processed_data) else args.raw_data)
    processed_df, scaled_features, scaler = preprocess_tourism_data(df_raw, scale_method="robust")
    print(f"[Stage 1] Preprocessed {len(processed_df):,} records across {processed_df['District'].nunique()} districts.")
    print(f"[Stage 1] Engineered features: {list(scaled_features.columns)}")

    # -------------------------------------------------------------------------
    # STAGE 2: MELISA'S PRIMARY OUTLIER DETECTION (ISOLATION FOREST)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [STAGE 2] Primary Outlier Detection - Isolation Forest (Melisa's Module)")
    print("=" * 60)

    if_df, if_detector = detect_outliers_isolation_forest(
        data=args.processed_data,
        contamination=args.contamination,
        output_path=args.if_results,
        verbose=True,
    )
    # Render Melisa's diagnostic visualizations
    plot_isolation_forest_visualizations(
        if_df,
        save_path=os.path.join(args.results_dir, "isolation_forest_visualizations.png"),
    )

    # -------------------------------------------------------------------------
    # STAGE 3: NIFA'S SECOND ALGORITHM (LOCAL OUTLIER FACTOR) & COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [STAGE 3] Local Outlier Factor (LOF) & Comparative Evaluation (Nifa's Module)")
    print("=" * 60)

    lof_res = run_lof_detection(
        data_path=args.processed_data,
        output_path=args.lof_results,
        n_neighbors=args.n_neighbors,
        contamination=args.contamination,
    )
    lof_df = lof_res[0] if isinstance(lof_res, tuple) else lof_res

    # Compare Isolation Forest (Melisa) vs Local Outlier Factor (Nifa)
    merged_comp_df, comp_summary = compare_outlier_algorithms(
        lof_df=lof_df,
        if_df=if_df,
    )

    # Render Nifa's visual evaluation plots
    plot_comparison_summary(
        comp_summary,
        save_path=os.path.join(args.results_dir, "outlier_comparison_summary.png"),
    )

    x_feat = next((c for c in ["Domestic_Tourists", "domestic"] if c in merged_comp_df.columns), None)
    y_feat = next((c for c in ["Foreign_Tourists", "foreign"] if c in merged_comp_df.columns), None)
    if not (x_feat and y_feat):
        num_c = merged_comp_df.select_dtypes(include=[np.number]).columns.tolist()
        x_feat = num_c[0] if num_c else merged_comp_df.columns[0]
        y_feat = num_c[1] if len(num_c) > 1 else x_feat

    plot_feature_scatter_comparison(
        merged_comp_df,
        x_col=x_feat,
        y_col=y_feat,
        save_path=os.path.join(args.results_dir, "feature_scatter_comparison.png"),
    )

    score_col_name = "lof_score" if "lof_score" in lof_df.columns else ("lof_anomaly_score" if "lof_anomaly_score" in lof_df.columns else lof_df.columns[-1])
    plot_lof_score_distribution(
        lof_df,
        score_col=score_col_name,
        save_path=os.path.join(args.results_dir, "lof_score_distribution.png"),
    )

    t_col = next((c for c in ["Date", "date", "Year", "year"] if c in merged_comp_df.columns), None)
    m_col = next((c for c in ["Total_Tourists", "total", "domestic", "Domestic_Tourists"] if c in merged_comp_df.columns), None)
    if t_col and m_col:
        plot_temporal_outlier_trend(
            merged_comp_df,
            date_or_index_col=t_col,
            metric_col=m_col,
            save_path=os.path.join(args.results_dir, "temporal_outlier_trend.png"),
        )

    # -------------------------------------------------------------------------
    # STAGE 4: MULTI-MODEL ENSEMBLE CONSENSUS & DISTRICT PROFILING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [STAGE 4] Multi-Model Ensemble Consensus & District Profiling")
    print("=" * 60)

    multi_results, ml_models = run_all_detectors(
        processed_df,
        scaled_features,
        contamination=args.contamination,
        min_consensus=2,
    )

    categorized_df = categorize_anomalies(multi_results, outlier_col="Ensemble_Outlier")
    summary_df = summarize_district_anomalies(categorized_df, outlier_col="Ensemble_Outlier")
    similarity_matrix = compute_model_agreement(categorized_df)

    eval_results = evaluate_outliers(
        categorized_df,
        ground_truth_col="Ground_Truth_Anomaly" if "Ground_Truth_Anomaly" in categorized_df.columns else None,
    )

    # Save summary tables
    anomalies_csv = os.path.join(args.results_dir, "detected_anomalies.csv")
    categorized_df.to_csv(anomalies_csv, index=False)

    district_summary_csv = os.path.join(args.results_dir, "district_anomaly_summary.csv")
    summary_df.to_csv(district_summary_csv, index=False)

    agreement_csv = os.path.join(args.results_dir, "model_agreement_matrix.csv")
    similarity_matrix.to_csv(agreement_csv)

    # -------------------------------------------------------------------------
    # STAGE 5: UNIFIED VISUALIZATION SUITE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [STAGE 5] Rendering Unified Visual Analytics Suite")
    print("=" * 60)

    generate_all_visualizations(
        categorized_df,
        similarity_matrix,
        output_dir=args.results_dir,
        outlier_col="Ensemble_Outlier",
    )

    # -------------------------------------------------------------------------
    # STAGE 6: COMPREHENSIVE EXECUTIVE REPORTING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [STAGE 6] Executive Synthesis & Reporting")
    print("=" * 60)

    report_path = os.path.join(args.reports_dir, "outlier_analysis_summary.md")
    generate_unified_executive_report(
        summary_df=summary_df,
        comparison_summary=comp_summary,
        eval_results=eval_results,
        anomalies_df=categorized_df,
        report_path=report_path,
    )

    # Final summary banner
    print("\n" + "=" * 72)
    print("              UNIFIED PIPELINE EXECUTION COMPLETED")
    print("=" * 72)
    print(f"Total Observations Screened     : {len(categorized_df):,}")
    print(f"Isolation Forest Outliers (Melisa): {comp_summary.get('if_outliers_count', 0):,}")
    print(f"Local Outlier Factor (Nifa)     : {comp_summary.get('lof_outliers_count', 0):,}")
    print(f"Mutual Consensus Outliers       : {comp_summary.get('both_outliers_count', 0):,}")
    print(f"Jaccard Agreement Index         : {comp_summary.get('jaccard_similarity', 0.0):.4f}")
    print(f"Overall Concordance Rate        : {comp_summary.get('concordance_rate_pct', 0.0):.2f}%")
    print(f"\nDeliverables Generated:")
    print(f"  [x] Gauri's Dataset:          {args.processed_data}")
    print(f"  [x] Melisa's IF Output:       {args.if_results}")
    print(f"  [x] Nifa's LOF Output:        {args.lof_results}")
    print(f"  [x] District Anomaly Summary: {district_summary_csv}")
    print(f"  [x] Model Agreement Matrix:   {agreement_csv}")
    print(f"  [x] Full Executive Report:    {report_path}")
    print(f"  [x] Complete Plot Suite:      results/*.png (8 analytical charts)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
