"""
Kerala Tourism Trend Outlier Analysis - Data Preprocessing & Feature Engineering Module
========================================================================================

Project: Kerala Tourism Trend Outlier Analysis Using Outlier Detection
Role / Contributor: Gauri (Data Collection, Data Preprocessing, and Exploratory Data Analysis)
Collaborators: Melisa (Isolation Forest & Evaluation), Nifa (Multi-model Outlier Mining)
Repository: https://github.com/elsasabumelisa-max/kerala-tourism-outlier-analysis.git

This module provides reusable, robust preprocessing functions designed to:
1. Support dynamic column names rather than assuming fixed, rigid column headers.
2. Inspect data health (missing values, data types, duplicate records).
3. Clean, sanitize, and validate authentic Kerala tourism datasets without dropping valid observations.
4. Support temporal feature engineering and feature scaling for downstream outlier detection.
5. Provide backward compatibility for pipeline orchestration (`main.py`, `melisa_outlier_detection.ipynb`).

Exported Functions:
-------------------
- `detect_column_mapping`: Dynamically resolves arbitrary input column names to canonical keys.
- `inspect_data`: Provides a diagnostic report of nulls, duplicates, and data types.
- `clean_numeric_series`: Cleans formatted strings (commas, whitespace) into numeric values.
- `clean_text_series`: Cleans whitespace, casing, and standardizes known Kerala district synonyms.
- `clean_tourism_data`: End-to-end cleaning pipeline returning a normalized DataFrame.
- `save_processed_data`: Exports cleaned DataFrame to a designated CSV path.
- `load_and_preprocess`: One-stop convenience function for loading, cleaning, and exporting.
- `load_tourism_data`: Loads dataset with date / temporal reconciliation.
- `preprocess_tourism_data`: Advanced feature engineering (MoM/YoY growth, seasonality, rolling volatility, scaling).
- `generate_kerala_tourism_dataset`: Calibrated benchmark generator with historical shocks.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    HAS_SKLEARN = True
except ImportError:
    StandardScaler = None
    RobustScaler = None
    HAS_SKLEARN = False

# Configure logging for reproducible, auditable runs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("preprocessing")

# Standard 14 administrative districts of Kerala
DISTRICTS: List[str] = [
    "Thiruvananthapuram",
    "Kollam",
    "Pathanamthitta",
    "Alappuzha",
    "Kottayam",
    "Idukki",
    "Ernakulam",
    "Thrissur",
    "Palakkad",
    "Malappuram",
    "Kozhikode",
    "Wayanad",
    "Kannur",
    "Kasaragod",
]

# Canonical column names for standardized datasets
CANONICAL_COLUMNS: List[str] = ["year", "district", "domestic", "foreign", "total"]

# Known historical / alternative spellings for Kerala's 14 districts
DISTRICT_SYNONYMS: Dict[str, str] = {
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "quilon": "Kollam",
    "kollam": "Kollam",
    "pathanamthitta": "Pathanamthitta",
    "alleppey": "Alappuzha",
    "alappuzha": "Alappuzha",
    "kottayam": "Kottayam",
    "idukki": "Idukki",
    "cochin": "Ernakulam",
    "ernakulam": "Ernakulam",
    "trichur": "Thrissur",
    "thrissur": "Thrissur",
    "palghat": "Palakkad",
    "palakkad": "Palakkad",
    "malappuram": "Malappuram",
    "calicut": "Kozhikode",
    "kozhikode": "Kozhikode",
    "wayanad": "Wayanad",
    "wynad": "Wayanad",
    "cannanore": "Kannur",
    "kannur": "Kannur",
    "kasargod": "Kasaragod",
    "kasaragod": "Kasaragod"
}

# Base tourist scale weight per district (relative to state baseline)
DISTRICT_WEIGHTS = {
    "Ernakulam": 1.95,
    "Thiruvananthapuram": 1.65,
    "Idukki": 1.40,
    "Alappuzha": 1.25,
    "Wayanad": 1.20,
    "Kozhikode": 1.05,
    "Thrissur": 1.00,
    "Kannur": 0.70,
    "Kollam": 0.65,
    "Palakkad": 0.60,
    "Kottayam": 0.55,
    "Malappuram": 0.50,
    "Kasaragod": 0.45,
    "Pathanamthitta": 0.40,
}

# Seasonal monthly multiplier across Kerala
MONTHLY_SEASONALITY = {
    1: 1.30, 2: 1.15, 3: 0.95, 4: 1.05, 5: 1.10, 6: 0.65,
    7: 0.60, 8: 0.90, 9: 1.10, 10: 1.05, 11: 1.20, 12: 1.45,
}

FEATURE_COLUMNS = [
    "Domestic_Tourists",
    "Foreign_Tourists",
    "Total_Tourists",
    "Hotel_Occupancy_Rate",
    "Tourism_Revenue_Cr",
    "Average_Stay_Days",
    "Domestic_Foreign_Ratio",
    "Foreign_Share",
    "MoM_Growth",
    "YoY_Growth",
    "Seasonality_Index",
]


# ==============================================================================
# 1. DYNAMIC COLUMN DETECTION & INSPECTION (Gauri's Core Functions)
# ==============================================================================

def detect_column_mapping(
    columns: Union[List[str], pd.Index],
    custom_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Dynamically identifies and maps varying input column names to canonical keys.

    Canonical keys produced:
      - 'year': Calendar year of data.
      - 'district': District name in Kerala.
      - 'domestic': Domestic tourist arrivals/visits count.
      - 'foreign': Foreign/international tourist arrivals/visits count.
      - 'total': Total tourist arrivals/visits count.

    Parameters
    ----------
    columns : list of str or pd.Index
        The column names present in the input dataset.
    custom_mapping : dict, optional
        A user-supplied dictionary {original_col: canonical_col} to override
        or supplement automatic detection.

    Returns
    -------
    dict
        Dictionary mapping existing column names to canonical column names.
    """
    raw_cols = list(columns)
    mapping: Dict[str, str] = {}

    patterns = {
        "year": [r"^year$", r"^yr$", r"^period$", r"^time$", r"^date$"],
        "district": [r"^district", r"^dist$", r"^region", r"^location", r"^area"],
        "domestic": [r"domestic", r"^dtv$", r"indigenous", r"national"],
        "foreign": [r"foreign", r"^ftv$", r"international", r"overseas", r"nri"],
        "total": [r"^total", r"^all_tourists", r"^grand_total", r"^combined"]
    }

    def normalize_str(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.strip().lower())

    matched_canonicals = set()

    for col in raw_cols:
        norm_col = normalize_str(col)
        matched = False
        for canon, pat_list in patterns.items():
            if canon in matched_canonicals:
                continue
            for pat in pat_list:
                if re.search(pat, col.strip().lower()) or re.search(pat, norm_col):
                    mapping[col] = canon
                    matched_canonicals.add(canon)
                    matched = True
                    break
            if matched:
                break

    if custom_mapping:
        for orig, canon in custom_mapping.items():
            if orig in raw_cols:
                mapping[orig] = canon
                matched_canonicals.add(canon)

    logger.info("Detected column mapping: %s", mapping)
    return mapping


def inspect_data(df: pd.DataFrame, print_summary: bool = True) -> Dict[str, Union[int, float, dict]]:
    """
    Performs a thorough quality and integrity inspection on the raw DataFrame.

    Checks:
    - Dimensions and shape
    - Missing/null value counts and percentages per column
    - Exact duplicate record count
    - Column data types and non-null counts
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    null_counts = df.isnull().sum().to_dict()
    null_percentages = {col: round((count / total_rows) * 100, 2) for col, count in null_counts.items()}
    duplicate_count = int(df.duplicated().sum())

    report = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": duplicate_count,
        "null_counts": null_counts,
        "null_percentages": null_percentages,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

    if print_summary:
        print("=" * 60)
        print("DATA INTEGRITY DIAGNOSTIC REPORT")
        print("=" * 60)
        print(f"Total Observations (Rows): {total_rows}")
        print(f"Total Features (Columns)  : {total_cols}")
        print(f"Duplicate Rows Identified : {duplicate_count}")
        print("-" * 60)
        print("Missing Values Check:")
        for col, count in null_counts.items():
            pct = null_percentages[col]
            status = "CLEAN" if count == 0 else f"MISSING {count} ({pct}%)"
            print(f"  - {col:25s}: {status}")
        print("-" * 60)
        print("Column Types:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col:25s}: {dtype}")
        print("=" * 60)

    return report


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Converts a pandas Series containing formatted numbers (e.g., strings with commas,
    spaces, or currency symbols) safely into standard numeric values without dropping rows.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({r"^\s*$": np.nan, r"^NA$": np.nan, r"^null$": np.nan, r"^-$": np.nan}, regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    Standardizes a text Series (such as district names) by removing excess whitespace,
    applying Title Casing, and reconciling historical or alternate spelling variations.
    """
    cleaned = series.astype(str).str.strip()

    def standardize_name(val: str) -> str:
        key = val.strip().lower()
        if key in DISTRICT_SYNONYMS:
            return DISTRICT_SYNONYMS[key]
        return val.strip().title()

    return cleaned.apply(standardize_name)


def clean_tourism_data(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
    drop_duplicates: bool = True,
    calculate_missing_totals: bool = True,
    add_derived_features: bool = True
) -> pd.DataFrame:
    """
    Primary data cleaning and normalization pipeline.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    clean_df = df.copy()

    # 1. Resolve and rename columns dynamically
    detected_mapping = detect_column_mapping(clean_df.columns, custom_mapping=column_mapping)
    clean_df = clean_df.rename(columns=detected_mapping)

    logger.info("Columns after renaming: %s", list(clean_df.columns))

    # 2. Handle duplicates
    initial_rows = len(clean_df)
    if drop_duplicates:
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)
        dropped_count = initial_rows - len(clean_df)
        if dropped_count > 0:
            logger.info("Removed %d exact duplicate row(s).", dropped_count)

    # 3. Clean district text if present
    if "district" in clean_df.columns:
        clean_df["district"] = clean_text_series(clean_df["district"])

    # 4. Clean numeric columns
    numeric_cols = [c for c in ["year", "domestic", "foreign", "total"] if c in clean_df.columns]
    for col in numeric_cols:
        clean_df[col] = clean_numeric_series(clean_df[col])

    # 5. Handle and reconcile totals
    has_domestic = "domestic" in clean_df.columns
    has_foreign = "foreign" in clean_df.columns
    has_total = "total" in clean_df.columns

    if calculate_missing_totals and has_domestic and has_foreign:
        if not has_total:
            clean_df["total"] = clean_df["domestic"] + clean_df["foreign"]
        else:
            computed_total = clean_df["domestic"] + clean_df["foreign"]
            clean_df["total"] = clean_df["total"].fillna(computed_total)
            mismatch_mask = (clean_df["total"] != computed_total) & clean_df["total"].notnull()
            if mismatch_mask.any():
                logger.warning(
                    "Reconciled %d row(s) where total did not equal domestic + foreign.",
                    mismatch_mask.sum()
                )
                clean_df.loc[mismatch_mask, "total"] = computed_total[mismatch_mask]

    # 6. Type casting
    if "year" in clean_df.columns:
        clean_df["year"] = clean_df["year"].astype("int64")

    for col in ["domestic", "foreign", "total"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna(0).round().astype("int64")

    # 7. Add derived analytical features
    if add_derived_features and "foreign" in clean_df.columns and "total" in clean_df.columns:
        clean_df["foreign_ratio"] = np.where(
            clean_df["total"] > 0,
            (clean_df["foreign"] / clean_df["total"]).round(4),
            0.0
        )

    preferred_order = ["year", "district", "domestic", "foreign", "total", "foreign_ratio"]
    ordered_cols = [c for c in preferred_order if c in clean_df.columns] + [
        c for c in clean_df.columns if c not in preferred_order
    ]
    clean_df = clean_df[ordered_cols]

    logger.info("Successfully cleaned dataset with %d rows and %d columns.", len(clean_df), len(clean_df.columns))
    return clean_df


def save_processed_data(
    df: pd.DataFrame,
    output_path: str = "data/processed_tourism.csv",
    index: bool = False
) -> str:
    """Exports processed DataFrame to CSV with directory creation."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Created output directory: %s", output_dir)

    df.to_csv(output_path, index=index, encoding="utf-8")
    abs_path = os.path.abspath(output_path)
    logger.info("Saved processed dataset to: %s (%d records)", abs_path, len(df))
    return abs_path


def load_and_preprocess(
    input_path: str = "data/kerala_tourism_raw.csv",
    output_path: Optional[str] = "data/processed_tourism.csv",
    **kwargs
) -> pd.DataFrame:
    """End-to-end wrapper: loads raw CSV, applies cleaning, and optionally saves."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    logger.info("Reading raw data from: %s", input_path)
    raw_df = pd.read_csv(input_path)
    cleaned_df = clean_tourism_data(raw_df, **kwargs)

    if output_path:
        save_processed_data(cleaned_df, output_path=output_path)

    return cleaned_df


# ==============================================================================
# 2. COMPATIBILITY & ADVANCED PIPELINE FUNCTIONS (For Melisa, Nifa, & main.py)
# ==============================================================================

def load_tourism_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw or cleaned Kerala tourism dataset with verification and date normalization.
    Ensures backward compatibility with main.py and Melisa's notebooks.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tourism dataset not found at: {filepath}")

    logger.info("Loading tourism dataset from %s", filepath)
    df = pd.read_csv(filepath)

    # Reconcile Date / Year / Month columns
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        if "Year" not in df.columns:
            df["Year"] = df["Date"].dt.year
        if "Month" not in df.columns:
            df["Month"] = df["Date"].dt.month
    elif "Year" in df.columns and "Month" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
        )
    elif "year" in df.columns:
        # Standard district-year table
        df["Year"] = df["year"]

    # Reconcile Total if missing
    if "Total_Tourists" not in df.columns:
        if "total" in df.columns:
            df["Total_Tourists"] = df["total"]
        elif "Domestic_Tourists" in df.columns and "Foreign_Tourists" in df.columns:
            df["Total_Tourists"] = df["Domestic_Tourists"] + df["Foreign_Tourists"]
        elif "domestic" in df.columns and "foreign" in df.columns:
            df["Total_Tourists"] = df["domestic"] + df["foreign"]

    return df


def preprocess_tourism_data(
    df: pd.DataFrame,
    scale_method: str = "robust",
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Cleans, imputes, and engineers advanced features for outlier mining:
      1. Ratios: Domestic-to-Foreign, Foreign Share
      2. Temporal dynamics: Month-over-Month (MoM) & Year-over-Year (YoY) Growth rates
      3. Seasonality index relative to district baseline
      4. Rolling statistics (3-month moving mean & volatility)
      5. Feature scaling (StandardScaler or RobustScaler)
    """
    data = df.copy()

    # Reconcile column names if using lowercase canonicals
    col_map = {
        "year": "Year", "district": "District", "domestic": "Domestic_Tourists",
        "foreign": "Foreign_Tourists", "total": "Total_Tourists"
    }
    for old_c, new_c in col_map.items():
        if old_c in data.columns and new_c not in data.columns:
            data[new_c] = data[old_c]

    # Ensure Date exists if Year and Month are available
    if "Date" not in data.columns and "Year" in data.columns and "Month" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Year"].astype(str) + "-" + data["Month"].astype(str).str.zfill(2) + "-01"
        )

    # Sort strictly by district and date if available
    sort_cols = [c for c in ["District", "Date", "Year"] if c in data.columns]
    if sort_cols:
        data = data.sort_values(by=sort_cols).reset_index(drop=True)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isna().sum() > 0:
            if "District" in data.columns:
                data[col] = data.groupby("District")[col].transform(
                    lambda x: x.fillna(x.median() if not x.isna().all() else 0)
                )
            else:
                data[col] = data[col].fillna(data[col].median())

    if "Domestic_Tourists" not in data.columns:
        data["Domestic_Tourists"] = data.get("Total_Tourists", 0)
    if "Foreign_Tourists" not in data.columns:
        data["Foreign_Tourists"] = 0
    if "Total_Tourists" not in data.columns:
        data["Total_Tourists"] = data["Domestic_Tourists"] + data["Foreign_Tourists"]

    if "Hotel_Occupancy_Rate" not in data.columns:
        data["Hotel_Occupancy_Rate"] = 55.0
    if "Average_Stay_Days" not in data.columns:
        data["Average_Stay_Days"] = 2.5
    if "Tourism_Revenue_Cr" not in data.columns:
        data["Tourism_Revenue_Cr"] = (
            data["Domestic_Tourists"] * data["Average_Stay_Days"] * 3800
            + data["Foreign_Tourists"] * data["Average_Stay_Days"] * 9500
        ) / 10000000.0

    # 1. Tourist Structure Ratios
    data["Domestic_Foreign_Ratio"] = data["Domestic_Tourists"] / (data["Foreign_Tourists"] + 1.0)
    data["Foreign_Share"] = data["Foreign_Tourists"] / (data["Total_Tourists"] + 1.0)

    # 2. Growth Rates (if District present)
    if "District" in data.columns:
        data["MoM_Growth"] = (
            data.groupby("District")["Total_Tourists"]
            .pct_change(fill_method=None)
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
        )
        data["YoY_Growth"] = (
            data.groupby("District")["Total_Tourists"]
            .pct_change(periods=min(12, max(1, len(data))), fill_method=None)
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
        )
        district_means = data.groupby("District")["Total_Tourists"].transform("mean")
        data["Seasonality_Index"] = data["Total_Tourists"] / (district_means + 1e-5)
        data["Rolling_Mean_3M"] = (
            data.groupby("District")["Total_Tourists"]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )
        data["Rolling_Std_3M"] = (
            data.groupby("District")["Total_Tourists"]
            .transform(lambda s: s.rolling(window=3, min_periods=1).std())
            .fillna(0.0)
        )
    else:
        data["MoM_Growth"] = 0.0
        data["YoY_Growth"] = 0.0
        data["Seasonality_Index"] = 1.0
        data["Rolling_Mean_3M"] = data["Total_Tourists"]
        data["Rolling_Std_3M"] = 0.0

    active_features = [col for col in FEATURE_COLUMNS if col in data.columns]
    feature_matrix = data[active_features].copy()
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if scale_method == "robust":
        if RobustScaler is not None:
            scaler = RobustScaler()
            scaled_arr = scaler.fit_transform(feature_matrix)
        else:
            med = feature_matrix.median()
            q75 = feature_matrix.quantile(0.75)
            q25 = feature_matrix.quantile(0.25)
            iqr = (q75 - q25).replace(0, 1.0)
            scaled_arr = ((feature_matrix - med) / iqr).values
            scaler = "robust_scaler_numpy"
    elif scale_method == "standard":
        if StandardScaler is not None:
            scaler = StandardScaler()
            scaled_arr = scaler.fit_transform(feature_matrix)
        else:
            mean = feature_matrix.mean()
            std = feature_matrix.std().replace(0, 1.0)
            scaled_arr = ((feature_matrix - mean) / std).values
            scaler = "standard_scaler_numpy"
    else:
        raise ValueError(f"Unknown scale method: {scale_method}")
    scaled_features = pd.DataFrame(scaled_arr, columns=active_features, index=data.index)

    return data, scaled_features, scaler


def generate_kerala_tourism_dataset(
    start_year: int = 2017,
    end_year: int = 2024,
    random_seed: int = 42,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates a realistic historical dataset of Kerala tourism footfalls, occupancies,
    and revenues across all 14 districts, embedding genuine anomalous historical events:
      - August 2018: Severe Kerala Deluge / Floods (-80% to -95% drop)
      - March 2020 - Dec 2020: COVID-19 pandemic nationwide lockdowns (-90% to -99% drop)
      - April - May 2021: Second COVID wave (-80% drop)
      - August - September 2022-2023: Revenge tourism surge during Onam (+50% spike)
      - April Thrissur Pooram festival surges
    """
    np.random.seed(random_seed)
    records = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for district in DISTRICTS:
                weight = DISTRICT_WEIGHTS.get(district, 1.0)
                season_mult = MONTHLY_SEASONALITY.get(month, 1.0)

                base_domestic = np.random.normal(loc=65000, scale=6000) * weight * season_mult
                base_foreign = np.random.normal(loc=6000, scale=1200) * weight * season_mult
                growth_factor = 1.0 + (year - start_year) * 0.04
                domestic = max(500, base_domestic * growth_factor)
                foreign = max(50, base_foreign * growth_factor)
                occupancy = np.clip(np.random.normal(58 + (season_mult - 1.0) * 25, 4.5), 15.0, 95.0)
                avg_stay = np.clip(np.random.normal(2.6, 0.4), 1.2, 5.0)
                anomaly_tag = "Normal"

                # 1. August 2018 Kerala Floods
                if year == 2018 and month == 8:
                    if district in ["Ernakulam", "Idukki", "Alappuzha", "Wayanad", "Pathanamthitta", "Thrissur"]:
                        domestic *= 0.10
                        foreign *= 0.05
                        occupancy = max(10.0, occupancy * 0.20)
                        anomaly_tag = "2018_Flood_Drop"
                    else:
                        domestic *= 0.35
                        foreign *= 0.15
                        occupancy *= 0.40
                        anomaly_tag = "2018_Flood_Drop"

                # 2. COVID First Wave
                elif year == 2020 and month in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    drop_mult = 0.02 if month in [4, 5] else 0.08
                    domestic *= drop_mult
                    foreign *= (drop_mult * 0.4)
                    occupancy = max(5.0, occupancy * 0.15)
                    anomaly_tag = "COVID_Lockdown_Wave1"

                # 3. COVID Second Wave
                elif year == 2021 and month in [4, 5, 6]:
                    domestic *= 0.18
                    foreign *= 0.03
                    occupancy = max(12.0, occupancy * 0.25)
                    anomaly_tag = "COVID_Second_Wave"

                # 4. Onam Rebound Spike
                elif year in [2022, 2023] and month == 9:
                    if district in ["Ernakulam", "Thiruvananthapuram", "Idukki", "Wayanad"]:
                        domestic *= 1.55
                        foreign *= 1.30
                        occupancy = min(98.0, occupancy * 1.35)
                        anomaly_tag = "Post_COVID_Onam_Rebound"

                # 5. Thrissur Pooram
                elif month == 4 and district == "Thrissur":
                    if year not in [2020, 2021]:
                        domestic *= 1.75
                        occupancy = min(96.0, occupancy * 1.40)
                        anomaly_tag = "Thrissur_Pooram_Surge"

                total = domestic + foreign
                revenue_cr = (domestic * avg_stay * 3800 + foreign * avg_stay * 9500) / 10000000.0

                records.append({
                    "Date": f"{year}-{month:02d}-01",
                    "Year": year,
                    "Month": month,
                    "District": district,
                    "Domestic_Tourists": int(round(domestic)),
                    "Foreign_Tourists": int(round(foreign)),
                    "Total_Tourists": int(round(total)),
                    "Hotel_Occupancy_Rate": round(float(occupancy), 2),
                    "Average_Stay_Days": round(float(avg_stay), 2),
                    "Tourism_Revenue_Cr": round(float(revenue_cr), 3),
                    "Ground_Truth_Anomaly": anomaly_tag,
                })

    df = pd.DataFrame(records)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Saved Kerala tourism dataset to: %s", save_path)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess Kerala Tourism Dataset for Outlier Analysis"
    )
    parser.add_argument(
        "--input",
        default="data/kerala_tourism_raw.csv",
        help="Path to raw input CSV (default: data/kerala_tourism_raw.csv)"
    )
    parser.add_argument(
        "--output",
        default="data/processed_tourism.csv",
        help="Path to save processed CSV (default: data/processed_tourism.csv)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress printing of data diagnosis report"
    )

    args = parser.parse_args()

    print(f"\n--- Loading and diagnosing: {args.input} ---")
    raw_data = pd.read_csv(args.input)
    if not args.no_summary:
        inspect_data(raw_data)

    print(f"\n--- Executing Preprocessing Pipeline ---")
    processed_df = clean_tourism_data(raw_data)

    print(f"\n--- Saving Processed Dataset to: {args.output} ---")
    save_processed_data(processed_df, output_path=args.output)

    print("\n--- Processed Dataset Preview (First 5 Rows) ---")
    print(processed_df.head())
    print("\nProcessing completed successfully!\n")
