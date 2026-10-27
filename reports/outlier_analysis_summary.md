# Kerala Tourism Outlier Analysis & Anomaly Mining Report

**Generated:** 2026-10-27 21:47:51  
**Scope:** 14 Districts of Kerala | Temporal Coverage: 2017 - 2024  
**Total Records Analyzed:** 1,344  
**Consensus Anomalies Flagged:** 66 (4.91% of total records)  

---

## 1. Executive Summary
This data mining initiative applies multi-paradigm anomaly detection (Statistical Fences, Isolation Forest, Local Outlier Factor, One-Class SVM, and Ensemble Voting) to Kerala tourism footfall and revenue records. By screening extreme variance from normal seasonal cycles, the system isolates high-impact disruptions, super-peak surges, and reporting discrepancies across Kerala's 14 administrative districts.

### Key Headline Metrics:
- **Total Anomalous Months Flagged:** `66`
- **Severe Crisis Shocks Detected:** `28` (e.g., August 2018 floods, 2020 pandemic lockdowns)
- **Holiday & Festival Surges:** `15` (e.g., Onam, Thrissur Pooram)
- **International Peak Surges:** `0` (e.g., Winter high-season & Biennale)
- **Top 3 Most Volatile Districts:** Ernakulam, Thrissur, Thiruvananthapuram

---

## 2. Multi-Model Detection Performance Benchmark

| Model            |   Precision |   Recall |   F1_Score |   Balanced_Accuracy |   Total_Flagged |
|:-----------------|------------:|---------:|-----------:|--------------------:|----------------:|
| EllipticEnvelope |      1      |   0.3553 |     0.5243 |              0.6776 |              81 |
| OneClassSVM      |      0.7308 |   0.25   |     0.3725 |              0.6156 |              78 |
| LOF              |      0.6543 |   0.2325 |     0.343  |              0.6037 |              81 |
| Ensemble_Outlier |      0.7273 |   0.2105 |     0.3265 |              0.5972 |              66 |
| IsolationForest  |      0.5432 |   0.193  |     0.2848 |              0.5799 |              81 |
| IQR_District     |      1      |   0.0526 |     0.1    |              0.5263 |              12 |
| ModZScore        |      0.3846 |   0.0219 |     0.0415 |              0.5074 |              13 |
| ZScore           |      0.5714 |   0.0175 |     0.034  |              0.5074 |               7 |

---

## 3. District Vulnerability & Anomaly Prevalence Ranking

The table below summarizes anomaly frequency and character per district:

| District           |   Total_Records |   Detected_Anomalies |   Anomaly_Rate_Pct |   Drop_Shocks |   Surge_Peaks |   Avg_Volume_Deviation |
|:-------------------|----------------:|---------------------:|-------------------:|--------------:|--------------:|-----------------------:|
| Ernakulam          |              96 |                   17 |              17.71 |             4 |             3 |                28906.7 |
| Thrissur           |              96 |                    9 |               9.38 |             2 |             6 |                57585.3 |
| Thiruvananthapuram |              96 |                    9 |               9.38 |             5 |             0 |               -62404.2 |
| Wayanad            |              96 |                    6 |               6.25 |             5 |             1 |               -52703.5 |
| Idukki             |              96 |                    6 |               6.25 |             3 |             1 |               -58740   |
| Alappuzha          |              96 |                    3 |               3.12 |             1 |             2 |                68382.3 |
| Kollam             |              96 |                    3 |               3.12 |             1 |             2 |                42541   |
| Kottayam           |              96 |                    3 |               3.12 |             1 |             0 |               -29810.8 |
| Kasaragod          |              96 |                    2 |               2.08 |             1 |             0 |               -29763   |
| Kannur             |              96 |                    2 |               2.08 |             2 |             0 |               -49764.5 |
| Kozhikode          |              96 |                    2 |               2.08 |             0 |             0 |               -32312   |
| Pathanamthitta     |              96 |                    2 |               2.08 |             2 |             0 |               -29404.5 |
| Palakkad           |              96 |                    1 |               1.04 |             1 |             0 |               -41850.5 |
| Malappuram         |              96 |                    1 |               1.04 |             0 |             0 |               -15310   |

---

## 4. Inlier vs. Outlier Operational Profile

Comparative operational parameters illustrating the scale of deviation in tourist volume, hotel occupancy, and tourism revenue:

| Feature              |   Inlier_Mean |   Inlier_Std |   Outlier_Mean |   Outlier_Std |   Mean_Shift_Pct |
|:---------------------|--------------:|-------------:|---------------:|--------------:|-----------------:|
| Domestic_Tourists    |      64012.2  |     43703.1  |       95964.3  |     101391    |            49.92 |
| Foreign_Tourists     |       5851.19 |      4307.83 |        7851.27 |      10084.3  |            34.18 |
| Total_Tourists       |      69863.4  |     47727.2  |      103816    |     110753    |            48.6  |
| Hotel_Occupancy_Rate |         52.64 |        18.06 |          43.27 |         31.41 |           -17.79 |
| Tourism_Revenue_Cr   |         77.54 |        55.53 |         114.59 |        123.85 |            47.78 |
| Average_Stay_Days    |          2.49 |         0.55 |           2.48 |          0.5  |            -0.57 |

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
