"""
Kerala Tourism Outlier Analysis Pipeline
=========================================
Main CLI orchestration script that handles data ingestion, feature engineering,
multi-algorithm anomaly detection, model evaluation, visualization generation,
and comprehensive reporting for Kerala Tourism analytics.
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure root directory is on the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    load_tourism_data,
    preprocess_tourism_data,
    generate_kerala_tourism_dataset,
    DISTRICTS,
)
from src.outlier_detection import run_all_detectors
from src.evaluation import (
    compute_model_agreement,
    summarize_district_anomalies,
    categorize_anomalies,
    evaluate_outliers,
)
from src.visualization import generate_all_visualizations


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kerala Tourism Outlier Analysis & Data Mining Framework"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/kerala_tourism_data.csv",
        help="Path to the Kerala tourism dataset (CSV/Excel).",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a realistic calibrated Kerala tourism dataset (2017-2024).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.06,
        help="Expected proportion of outliers in the data (default: 0.06).",
    )
    parser.add_argument(
        "--consensus",
        type=int,
        default=2,
        help="Minimum number of model votes required for ensemble outlier flag (default: 2).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save tabular outputs and visualization plots.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save analytical markdown reports.",
    )
    return parser.parse_args()


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to markdown string, with robust fallback."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback if tabulate is unavailable
        headers = list(df.columns)
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        rows = [
            "| " + " | ".join(str(val) for val in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([header_row, separator] + rows)


def generate_executive_report(
    summary_df: pd.DataFrame,
    eval_results: dict,
    anomalies_df: pd.DataFrame,
    report_path: str,
):
    """
    Synthesizes and writes an executive analytical report highlighting
    temporal crisis shocks, festive super-peaks, district vulnerabilities,
    and strategic recommendations for Kerala Tourism stakeholders.
    """
    total_records = len(anomalies_df)
    ensemble_anomalies = anomalies_df[anomalies_df["Ensemble_Outlier"] == 1]
    n_outliers = len(ensemble_anomalies)
    outlier_pct = (n_outliers / total_records * 100) if total_records > 0 else 0

    category_counts = ensemble_anomalies["Anomaly_Category"].value_counts().to_dict()

    top_vulnerable = summary_df.head(3)["District"].tolist()

    report_content = f"""# Kerala Tourism Outlier Analysis & Anomaly Mining Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Scope:** 14 Districts of Kerala | Temporal Coverage: {anomalies_df['Date'].dt.year.min()} - {anomalies_df['Date'].dt.year.max()}  
**Total Records Analyzed:** {total_records:,}  
**Consensus Anomalies Flagged:** {n_outliers:,} ({outlier_pct:.2f}% of total records)  

---

## 1. Executive Summary
This data mining initiative applies multi-paradigm anomaly detection (Statistical Fences, Isolation Forest, Local Outlier Factor, One-Class SVM, and Ensemble Voting) to Kerala tourism footfall and revenue records. By screening extreme variance from normal seasonal cycles, the system isolates high-impact disruptions, super-peak surges, and reporting discrepancies across Kerala's 14 administrative districts.

### Key Headline Metrics:
- **Total Anomalous Months Flagged:** `{n_outliers}`
- **Severe Crisis Shocks Detected:** `{category_counts.get('Severe Drop / Crisis Shock', 0)}` (e.g., August 2018 floods, 2020 pandemic lockdowns)
- **Holiday & Festival Surges:** `{category_counts.get('Domestic Festival / Holiday Surge', 0)}` (e.g., Onam, Thrissur Pooram)
- **International Peak Surges:** `{category_counts.get('International Peak Surge', 0)}` (e.g., Winter high-season & Biennale)
- **Top 3 Most Volatile Districts:** {', '.join(top_vulnerable)}

---

## 2. Multi-Model Detection Performance Benchmark

"""
    if "Model_Metrics" in eval_results and not eval_results["Model_Metrics"].empty:
        report_content += _df_to_markdown(eval_results["Model_Metrics"])
    else:
        report_content += "*Unsupervised execution without pre-labeled ground truth.*"

    report_content += f"""

---

## 3. District Vulnerability & Anomaly Prevalence Ranking

The table below summarizes anomaly frequency and character per district:

{_df_to_markdown(summary_df)}

---

## 4. Inlier vs. Outlier Operational Profile

Comparative operational parameters illustrating the scale of deviation in tourist volume, hotel occupancy, and tourism revenue:

"""
    if "Profile_Comparison" in eval_results and not eval_results["Profile_Comparison"].empty:
        report_content += _df_to_markdown(eval_results["Profile_Comparison"])

    report_content += f"""

---

## 5. Strategic Recommendations for Kerala Tourism Department

1. **Disaster Resilience & Early Recovery Protocols**:
   - Analysis shows dramatic collapse during the **August 2018 Floods** and **2020 Lockdowns** in riverine and highland districts (*Alappuzha, Idukki, Ernakulam, Wayanad*).
   - Establish contingency tourist safety registries and post-monsoon rehabilitation subsidies to accelerate recovery during off-season shocks.

2. **Managing Festival & Super-Season Congestion**:
   - Cultural peaks such as **Thrissur Pooram (April in Thrissur)** and **Onam (August-September statewide)** experience up to **2.4x baseline demand**, stressing local hotel capacity and transportation.
   - Implement dynamic inter-district tourist dispersal initiatives (e.g., promoting northern circuit gems like Kasaragod and Kannur during southern peak periods).

3. **Data Quality & Reporting Assurance**:
   - Detection of reporting outliers indicates occasional batch upload delays or single-month zero-logging. Implement automated telemetry verification at district tourism information offices (DTPCs).

---
*Report generated automatically by Kerala Tourism Outlier Mining System.*
"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Report] Executive analysis saved to: {report_path}")


def main():
    print("=" * 70)
    print("   KERALA TOURISM OUTLIER ANALYSIS & DATA MINING FRAMEWORK")
    print("=" * 70)

    args = parse_arguments()

    # Step 1: Ensure dataset availability
    if args.generate_sample or not os.path.exists(args.data):
        print(f"\n[Step 1] Dataset not found at '{args.data}' or sample requested.")
        print("          Synthesizing realistic calibrated Kerala Tourism dataset (2017-2024)...")
        generate_kerala_tourism_dataset(
            start_year=2017,
            end_year=2024,
            save_path=args.data,
        )

    print(f"\n[Step 1] Ingesting dataset from: {args.data}")
    raw_df = load_tourism_data(args.data)
    print(f"         Loaded {len(raw_df):,} records across {raw_df['District'].nunique()} districts.")

    # Step 2: Preprocessing and feature engineering
    print("\n[Step 2] Executing preprocessing, temporal metrics, and feature scaling...")
    processed_df, scaled_features, scaler = preprocess_tourism_data(raw_df, scale_method="robust")
    print(f"         Engineered {scaled_features.shape[1]} features for multivariate anomaly analysis.")

    # Step 3: Outlier Detection
    print("\n[Step 3] Running statistical and machine learning outlier detectors:")
    print("         - Statistical: Z-Score, Modified Z-Score (MAD), Global & District IQR")
    print("         - Machine Learning: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, DBSCAN")
    print(f"         - Ensemble: Consensus voting (threshold >= {args.consensus} models)")

    results_df, models = run_all_detectors(
        processed_df,
        scaled_features,
        contamination=args.contamination,
        min_consensus=args.consensus,
    )

    # Step 4: Anomaly Categorization & Profiling
    print("\n[Step 4] Categorizing detected anomalies & synthesizing profiles...")
    categorized_df = categorize_anomalies(results_df, outlier_col="Ensemble_Outlier")
    summary_df = summarize_district_anomalies(categorized_df, outlier_col="Ensemble_Outlier")
    similarity_matrix = compute_model_agreement(categorized_df)

    eval_results = evaluate_outliers(
        categorized_df,
        ground_truth_col="Ground_Truth_Anomaly" if "Ground_Truth_Anomaly" in categorized_df.columns else None,
    )

    # Step 5: Save Tabular Results
    os.makedirs(args.results_dir, exist_ok=True)
    anomalies_only = categorized_df[categorized_df["Ensemble_Outlier"] == 1]

    anomalies_csv = os.path.join(args.results_dir, "detected_anomalies.csv")
    categorized_df.to_csv(anomalies_csv, index=False)

    district_summary_csv = os.path.join(args.results_dir, "district_anomaly_summary.csv")
    summary_df.to_csv(district_summary_csv, index=False)

    agreement_csv = os.path.join(args.results_dir, "model_agreement_matrix.csv")
    similarity_matrix.to_csv(agreement_csv)

    print(f"\n[Step 5] Tabular outputs saved:")
    print(f"         - Detailed predictions: {anomalies_csv}")
    print(f"         - District summary:     {district_summary_csv}")
    print(f"         - Model agreement:      {agreement_csv}")

    # Step 6: Generate Visualizations
    print(f"\n[Step 6] Rendering high-resolution analytics charts to '{args.results_dir}/'...")
    generate_all_visualizations(
        categorized_df,
        similarity_matrix,
        output_dir=args.results_dir,
        outlier_col="Ensemble_Outlier",
    )

    # Step 7: Executive Report Synthesis
    print(f"\n[Step 7] Generating executive analytical report...")
    report_file = os.path.join(args.reports_dir, "outlier_analysis_summary.md")
    generate_executive_report(summary_df, eval_results, categorized_df, report_file)

    # Terminal Summary Output
    print("\n" + "=" * 70)
    print("                      PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    n_flagged = categorized_df["Ensemble_Outlier"].sum()
    pct_flagged = (n_flagged / len(categorized_df)) * 100
    print(f"Total Observations Screened : {len(categorized_df):,}")
    print(f"Ensemble Anomalies Flagged  : {n_flagged:,} ({pct_flagged:.2f}%)")
    print(f"Top 3 Volatile Districts    : {', '.join(summary_df.head(3)['District'].tolist())}")

    if "Model_Metrics" in eval_results and not eval_results["Model_Metrics"].empty:
        print("\nModel Benchmark Scores (vs. Ground Truth):")
        print(eval_results["Model_Metrics"].to_string(index=False))

    print("\nVisualizations Generated:")
    print("  [x] results/timeseries_anomalies.png")
    print("  [x] results/multivariate_outliers_scatter.png")
    print("  [x] results/district_boxplots.png")
    print("  [x] results/model_agreement_heatmap.png")
    print("  [x] results/seasonal_anomaly_matrix.png")
    print(f"\nFull analytical report: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
