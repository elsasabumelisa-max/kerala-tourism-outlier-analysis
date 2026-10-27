"""
Data Preprocessing & Feature Engineering for Kerala Tourism Outlier Analysis
=============================================================================
Provides data ingestion, sanitization, temporal feature engineering, and
a realistic Kerala tourism dataset generator calibrated with real historical events
(2018 floods, 2020-2021 pandemic lockdowns, Onam festival peaks, and winter surges).
"""

from typing import Tuple, List, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

DISTRICTS = [
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

# Base tourist scale weight per district (relative to state baseline)
DISTRICT_WEIGHTS = {
    "Ernakulam": 1.95,          # Gateway, Kochi, Biennale, Business & Heritage
    "Thiruvananthapuram": 1.65, # Capital, Kovalam, Varkala
    "Idukki": 1.40,             # Munnar, Thekkady, Hill station
    "Alappuzha": 1.25,          # Backwaters, Houseboats, Coastal
    "Wayanad": 1.20,            # Hill station, wildlife, ecotourism
    "Kozhikode": 1.05,          # Culinary, Malabar culture
    "Thrissur": 1.00,           # Cultural capital, Pooram
    "Kannur": 0.70,             # Theyyam, northern coastal
    "Kollam": 0.65,             # Ashtamudi lake, cashew hub
    "Palakkad": 0.60,           # Silent Valley, heritage
    "Kottayam": 0.55,           # Kumarakom gateway
    "Malappuram": 0.50,         # Historic, ecotourism
    "Kasaragod": 0.45,          # Bekal Fort
    "Pathanamthitta": 0.40,     # Pilgrimage hub (Sabarimala)
}

# Seasonal monthly multiplier across Kerala
# (Peak in Dec-Jan, Summer in Mar-May, Monsoon dip in Jun-Jul, Onam spike in Aug-Sep)
MONTHLY_SEASONALITY = {
    1: 1.30,   # Peak winter / New year
    2: 1.15,   # Late winter
    3: 0.95,   # Early summer
    4: 1.05,   # Summer vacation & Vishu / Thrissur Pooram
    5: 1.10,   # School vacations
    6: 0.65,   # Southwest monsoon onset (sharp dip)
    7: 0.60,   # Heavy monsoon downpours
    8: 0.90,   # Onam prep / festive pickup
    9: 1.10,   # Onam peak tourism
    10: 1.05,  # Autumn pickup
    11: 1.20,  # Winter season kickoff
    12: 1.45,  # Peak international & domestic holiday season
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
      - Synthetic recording errors (negative or zero outliers)

    Returns:
        pd.DataFrame containing historical tourism records.
    """
    np.random.seed(random_seed)
    records = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for district in DISTRICTS:
                weight = DISTRICT_WEIGHTS.get(district, 1.0)
                season_mult = MONTHLY_SEASONALITY.get(month, 1.0)

                # Baseline monthly arrivals for an average district
                base_domestic = np.random.normal(loc=65000, scale=6000) * weight * season_mult
                base_foreign = np.random.normal(loc=6000, scale=1200) * weight * season_mult

                # Add year-on-year growth (organic ~4-6% annually pre-COVID)
                growth_factor = 1.0 + (year - start_year) * 0.04
                domestic = max(500, base_domestic * growth_factor)
                foreign = max(50, base_foreign * growth_factor)

                occupancy = np.clip(
                    np.random.normal(58 + (season_mult - 1.0) * 25, 4.5), 15.0, 95.0
                )
                avg_stay = np.clip(np.random.normal(2.6, 0.4), 1.2, 5.0)

                # INJECT REAL-WORLD ANOMALIES & HISTORIC SHOCKS:
                anomaly_tag = "Normal"

                # 1. August 2018 Kerala Floods
                if year == 2018 and month == 8:
                    if district in ["Ernakulam", "Idukki", "Alappuzha", "Wayanad", "Pathanamthitta", "Thrissur"]:
                        domestic *= 0.12  # 88% crash
                        foreign *= 0.08   # 92% crash
                        occupancy = np.random.uniform(5.0, 15.0)
                        anomaly_tag = "2018_Flood_Disaster"
                    else:
                        domestic *= 0.35
                        foreign *= 0.25
                        occupancy = np.random.uniform(15.0, 25.0)
                        anomaly_tag = "2018_Flood_Disaster"

                # 2. September 2018 Post-Flood sluggish recovery
                elif year == 2018 and month == 9:
                    domestic *= 0.55
                    foreign *= 0.40
                    occupancy *= 0.65
                    anomaly_tag = "Post_Flood_Lull"

                # 3. COVID-19 Lockdown (March 2020 - December 2020)
                elif year == 2020 and month in [3, 4, 5, 6, 7, 8]:
                    domestic *= 0.03   # 97% collapse
                    foreign = np.random.randint(0, 15) # Near zero
                    occupancy = np.random.uniform(2.0, 8.0)
                    avg_stay = 1.0
                    anomaly_tag = "COVID_19_Lockdown"

                elif year == 2020 and month in [9, 10, 11, 12]:
                    domestic *= 0.20   # Minor domestic easing
                    foreign = np.random.randint(5, 50)
                    occupancy = np.random.uniform(10.0, 22.0)
                    anomaly_tag = "COVID_19_Lockdown"

                # 4. COVID-19 Second Wave (April - June 2021)
                elif year == 2021 and month in [4, 5, 6]:
                    domestic *= 0.15
                    foreign = np.random.randint(2, 25)
                    occupancy = np.random.uniform(8.0, 18.0)
                    anomaly_tag = "COVID_Second_Wave"

                # 5. Thrissur Pooram Annual Cultural Mega-Surge (April in Thrissur)
                elif district == "Thrissur" and month == 4 and year not in [2020, 2021]:
                    domestic *= 2.40  # Massive festival spike
                    foreign *= 1.80
                    occupancy = np.random.uniform(88.0, 99.0)
                    anomaly_tag = "Thrissur_Pooram_Surge"

                # 6. Post-COVID Revenge Tourism Peak (September 2022 Onam in Ernakulam/Wayanad/Idukki)
                elif year == 2022 and month == 9 and district in ["Ernakulam", "Wayanad", "Idukki", "Alappuzha"]:
                    domestic *= 1.85
                    foreign *= 1.40
                    occupancy = np.random.uniform(86.0, 98.0)
                    anomaly_tag = "Revenge_Tourism_Surge"

                # 7. Cochin Carnival / Biennale Peak (December 2022 & 2023 in Ernakulam)
                elif district == "Ernakulam" and month == 12 and year in [2022, 2023]:
                    domestic *= 1.70
                    foreign *= 2.10
                    occupancy = np.random.uniform(90.0, 99.5)
                    anomaly_tag = "Biennale_Winter_Surge"

                # 8. Occasional Random Recording Irregularity (1% probability)
                elif np.random.rand() < 0.008:
                    if np.random.rand() < 0.5:
                        domestic = max(5, domestic * 0.05)  # Data loss / Under-reporting
                        anomaly_tag = "Reporting_Drop_Error"
                    else:
                        domestic *= 2.9  # Erroneous duplicate batch upload
                        anomaly_tag = "Data_Spike_Error"

                domestic = int(round(domestic))
                foreign = int(round(foreign))
                total = domestic + foreign

                # Revenue calculation: Domestic spend ~ 3800 INR/day, Foreign ~ 9500 INR/day (in Crores)
                # Revenue (Cr) = (Domestic * Avg_Stay * 3800 + Foreign * Avg_Stay * 9500) / 10^7
                revenue_cr = round(
                    ((domestic * avg_stay * 3800) + (foreign * avg_stay * 9500)) / 10000000.0,
                    2,
                )

                records.append(
                    {
                        "Year": year,
                        "Month": month,
                        "Date": f"{year}-{month:02d}-01",
                        "District": district,
                        "Domestic_Tourists": domestic,
                        "Foreign_Tourists": foreign,
                        "Total_Tourists": total,
                        "Average_Stay_Days": round(avg_stay, 2),
                        "Hotel_Occupancy_Rate": round(occupancy, 2),
                        "Tourism_Revenue_Cr": revenue_cr,
                        "Ground_Truth_Anomaly": 0 if anomaly_tag == "Normal" else 1,
                        "Anomaly_Reason": anomaly_tag,
                    }
                )

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Date", "District"]).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[Dataset] Generated {len(df)} records saved to: {save_path}")

    return df


def load_tourism_data(filepath: str) -> pd.DataFrame:
    """
    Loads Kerala tourism dataset from a CSV or Excel file.
    Validates mandatory schema columns and parses temporal fields.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tourism dataset not found at: {filepath}")

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported format. Please provide a CSV or Excel file.")

    # Standardize column naming
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    # Check minimum required columns
    required_cols = ["District"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}' in dataset.")

    # Date handling
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
    else:
        raise KeyError("Dataset must contain either 'Date' or both 'Year' and 'Month' columns.")

    # Compute Total_Tourists if missing
    if "Total_Tourists" not in df.columns:
        if "Domestic_Tourists" in df.columns and "Foreign_Tourists" in df.columns:
            df["Total_Tourists"] = df["Domestic_Tourists"] + df["Foreign_Tourists"]
        elif "Domestic_Tourists" in df.columns:
            df["Total_Tourists"] = df["Domestic_Tourists"]

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

    Returns:
        df_processed: Original dataframe with engineered features and clean values.
        scaled_features: Scaled feature matrix ready for ML outlier models.
        scaler: The fitted scaler object.
    """
    data = df.copy()

    # Sort strictly by district and date for chronological rolling features
    data = data.sort_values(by=["District", "Date"]).reset_index(drop=True)

    # Impute missing numericals with district median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isna().sum() > 0:
            data[col] = data.groupby("District")[col].transform(
                lambda x: x.fillna(x.median() if not x.isna().all() else 0)
            )

    # Ensure fundamental columns exist
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

    # 2. Month-over-Month (MoM) Growth per district
    data["MoM_Growth"] = (
        data.groupby("District")["Total_Tourists"]
        .pct_change(fill_method=None)
        .fillna(0.0)
        .replace([np.inf, -np.inf], 0.0)
    )

    # 3. Year-over-Year (YoY) Growth per district (lag 12 months)
    data["YoY_Growth"] = (
        data.groupby("District")["Total_Tourists"]
        .pct_change(periods=12, fill_method=None)
        .fillna(0.0)
        .replace([np.inf, -np.inf], 0.0)
    )

    # 4. District Seasonality Index
    district_means = data.groupby("District")["Total_Tourists"].transform("mean")
    data["Seasonality_Index"] = data["Total_Tourists"] / (district_means + 1e-5)

    # 5. Rolling Statistics (3-Month Window)
    data["Rolling_Mean_3M"] = (
        data.groupby("District")["Total_Tourists"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    data["Rolling_Std_3M"] = (
        data.groupby("District")["Total_Tourists"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).std())
        .fillna(0.0)
    )

    # Extract available feature columns
    active_features = [col for col in FEATURE_COLUMNS if col in data.columns]
    feature_matrix = data[active_features].copy()

    # Fill any remaining NaNs or Infs
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Scaling
    if scale_method == "robust":
        scaler = RobustScaler()
    elif scale_method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scale method: {scale_method}")

    scaled_arr = scaler.fit_transform(feature_matrix)
    scaled_features = pd.DataFrame(scaled_arr, columns=active_features, index=data.index)

    return data, scaled_features, scaler
