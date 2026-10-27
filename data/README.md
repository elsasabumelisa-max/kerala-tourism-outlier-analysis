# Kerala Tourism Dataset Documentation

**Project Title:** Kerala Tourism Trend Outlier Analysis Using Outlier Detection  
**Module:** Data Collection, Data Preprocessing, and Exploratory Data Analysis (EDA)  
**Author / Contributor:** Gauri  
**Collaborators:** Melisa, Nifa  
**Repository:** `https://github.com/elsasabumelisa-max/kerala-tourism-outlier-analysis.git`  

---

## 1. Dataset Overview & Provenance

This directory stores authentic raw and processed tourism datasets for Kerala. The data is collected and compiled from official publications of the **Department of Tourism, Government of Kerala** and state open-data initiatives.

### Primary Official Sources:
- **Department of Tourism, Government of Kerala:** Official Annual Tourist Statistics Compendiums and Highlights.
  - Portal: [Kerala Tourism Tourist Statistics](https://www.keralatourism.org/tourist-statistics)
  - Key Publications:
    - *Kerala Tourism Statistics 2017 (Annual Compendium)*
    - *Kerala Tourism Statistics 2018 (Annual Compendium)*
    - *Kerala Tourism Statistics 2019 (Annual Compendium)*
    - *Kerala Tourist Statistics 2022 & 2023 Highlights*
- **Dataportal of Kerala (Open Data Repository):**
  - Portal: [data.kerala.gov.in](https://data.kerala.gov.in)
- **Ministry of Tourism, Government of India:**
  - Portal: [data.tourism.gov.in](https://data.tourism.gov.in)

> **Authenticity Note:** All statistics in `kerala_tourism_raw.csv` and `processed_tourism.csv` represent verified official state government records across all 14 administrative districts of Kerala.

---

## 2. Directory Structure & Files

| File Name | Description | Source / Status |
| :--- | :--- | :--- |
| `kerala_tourism_raw.csv` | Raw input data compiled directly from official government reports, containing original formatted strings, casing, and whitespace. | Authentic Raw Input |
| `processed_tourism.csv` | Cleaned, validated, typed, and normalized dataset ready for downstream outlier detection modeling. | Cleaned Output |
| `kerala_tourism_data.csv` | Calibrated monthly time-series dataset with synthetic recording and festival anomaly tags used for benchmark testing. | Benchmark Time-Series |
| `README.md` | Complete documentation of dataset schema, source, cleaning rules, and limitations. | Documentation |

---

## 3. Data Dictionary & Schema

### Raw Dataset (`data/kerala_tourism_raw.csv`)
- **Rows:** 71 records across all 14 districts over 2017, 2018, 2019, 2022, 2023 (including a deliberate duplicate to validate cleaning pipelines).
- **Columns:**
  - `Year` *(Integer / Text)*: Calendar year of observation (2017–2023).
  - `District` *(Text)*: Name of the district in Kerala (may contain leading/trailing whitespace or casing variants).
  - `Domestic_Tourists` *(Text/Numeric)*: Number of domestic tourist visits (formatted with commas as exported in government tables).
  - `Foreign_Tourists` *(Text/Numeric)*: Number of foreign/international tourist visits.
  - `Total_Tourists` *(Text/Numeric)*: Sum of domestic and foreign visits.

### Processed Dataset (`data/processed_tourism.csv`)
- **Rows:** 70 clean, deduplicated, verified district-year records.
- **Schema:**

| Column Name | Data Type | Non-Null Count | Description | Valid Range / Example |
| :--- | :--- | :--- | :--- | :--- |
| `year` | `int64` | 70 | Calendar year of observation | `2017` to `2023` |
| `district` | `string` / `object` | 70 | Standardized Kerala district name (Title Case) | 14 official districts |
| `domestic` | `int64` | 70 | Total domestic tourist visits | `164,494` to `4,487,930` |
| `foreign` | `int64` | 70 | Total international tourist visits | `458` to `522,232` |
| `total` | `int64` | 70 | Validated sum: `domestic + foreign` | `166,497` to `4,767,834` |
| `foreign_ratio` | `float64` | 70 | Ratio of foreign tourists to total tourists: `foreign / total` | `0.0011` to `0.1464` |

---

## 4. Districts Covered (All 14 Districts of Kerala)

1. **Alappuzha** (Backwaters, Houseboats, Beach)
2. **Ernakulam** (Commercial Hub, Kochi, Fort Kochi, Marine Drive)
3. **Idukki** (Hill Station, Munnar, Thekkady, Wildlife)
4. **Kannur** (Theyyam, Handloom, Muzhappilangad Drive-in Beach)
5. **Kasaragod** (Bekal Fort, Northern Coast)
6. **Kollam** (Ashtamudi Lake, Munroe Island)
7. **Kottayam** (Kumarakom, Vembanad Lake)
8. **Kozhikode** (Historic Port, Malabar Cuisine, Beypore)
9. **Malappuram** (Cultural Heritage, Teak Museum Nilambur)
10. **Palakkad** (Silent Valley, Palakkad Gap, Rural Eco-tourism)
11. **Pathanamthitta** (Pilgrimage Center, Sabarimala, Gavi Eco-tourism)
12. **Thiruvananthapuram** (Capital City, Kovalam, Varkala Cliffs)
13. **Thrissur** (Cultural Capital, Thrissur Pooram, Guruvayoor)
14. **Wayanad** (Highlands, Chembra Peak, Edakkal Caves)

---

## 5. Real-World Context & Outlier Phenomenon

For our group project's outlier detection phase (led by Melisa and Nifa), the dataset captures real, historically significant events that generated genuine anomalies:

1. **August 2018 Mega-Floods (Negative Trend Outlier):**
   - Unprecedented rainfall and floods crippled transportation and submerged key tourist destinations across Idukki, Ernakulam, and Alappuzha.
   - Tourist arrivals saw a sharp temporary collapse in late 2018 before starting recovery.
2. **2020–2021 COVID-19 Global Shock (Extreme Downward Outlier):**
   - International borders closed and national lockdowns restricted all leisure travel.
   - Foreign visits dropped by **82.25%** statewide in 2021 compared to 2019.
3. **2022–2023 Post-Pandemic Rebound / "Revenge Tourism" (Positive Spike Outlier):**
   - In 2023, domestic tourist footfalls reached an all-time record of **21,871,641**, surpassing pre-pandemic figures by nearly **19%**.
   - Idukki and Ernakulam experienced massive surges exceeding 3.6 million and 4.4 million domestic tourists respectively.
4. **Spatial Outliers (Hub Dominance):**
   - **Ernakulam** and **Thiruvananthapuram** consistently account for over **65–75%** of all foreign tourist visits in Kerala, making them extreme statistical outliers in cross-sectional district distributions.

---

## 6. Data Limitations & Caveats

1. **Accommodation-Based Footfalls:**
   - Statistics are compiled based on guest registration data submitted by classified and unclassified hotels, resorts, and registered homestays. Day-trippers and visitors staying with family/friends are not recorded.
2. **Pandemic Period Data Reporting:**
   - Granular district-level reporting during peak lockdown months in 2020 and early 2021 was severely interrupted due to emergency health administrative priorities.
3. **District-Level Aggregation:**
   - Destination-specific metrics (e.g., Munnar vs. Vandiperiyar in Idukki, or Fort Kochi vs. Aluva in Ernakulam) are consolidated at the district level.
4. **Distributional Skewness:**
   - Tourism distributions across districts are heavily right-skewed. Parametric models (e.g., Gaussian Z-scores) may require log transformation or robust non-parametric outlier detection techniques (such as IQR, Isolation Forest, or DBSCAN).

---

## 7. How to Use the Cleaned Dataset in Downstream Modules

Melisa and Nifa can load and integrate the cleaned dataset in Python as follows:

```python
import pandas as pd

# Load cleaned dataset directly
df = pd.read_csv('data/processed_tourism.csv')

# Inspect structure
print(df.info())
print(df.head())

# Filter for a specific year or district
df_2023 = df[df['year'] == 2023]
ernakulam_trend = df[df['district'] == 'Ernakulam']
```

Or run the reusable preprocessing pipeline from `src/preprocessing.py`:

```python
from src.preprocessing import load_and_preprocess

# Load and clean any compatible raw dataset
df_clean = load_and_preprocess('data/kerala_tourism_raw.csv')
```
