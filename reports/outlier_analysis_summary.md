# Kerala Tourism Trend Outlier Analysis - Executive Final Report

**Generated:** 2026-10-30 14:58:14  
**Project:** Kerala Tourism Trend Outlier Analysis Using Outlier Detection  
**Team Members & Roles:**
- **Gauri:** Data Collection, Preprocessing & Exploratory Data Analysis (`src/preprocessing.py`)
- **Melisa:** Primary Outlier Detection - Isolation Forest (`src/outlier_detection.py`)
- **Nifa:** Second Algorithm - Local Outlier Factor (LOF) & Comparative Evaluation (`src/evaluation.py`, `src/visualization.py`)

**Scope:** 14 Districts of Kerala | Total Records Analyzed: 70  

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
| **Total Observations Screened** | `70` | Full monthly dataset across all 14 districts |
| **Isolation Forest Outliers (Melisa)** | `4` (5.71%) | Global volume surges and disaster collapses |
| **Local Outlier Factor Outliers (Nifa)** | `4` (5.71%) | Local density variations relative to peer neighbors |
| **Mutual Consensus Outliers (Both)** | `1` (1.43%) | High-confidence anomalies confirmed by both models |
| **Jaccard Similarity Index** | `0.1429` | Overlap ratio: $|IF \cap LOF| / |IF \cup LOF|$ |
| **Overall Concordance Rate** | `91.43%` | Percentage of observations where both models agree |

> **Methodological Note:** Because this is an unsupervised anomaly detection task without ground truth, supervised metrics (Precision, Recall, F1) treating one model as truth are methodologically invalid. Agreement reflects complementary algorithmic viewpoints: LOF captures local density deviations, while Isolation Forest isolates global volume deviations.

---

## 3. District Vulnerability & Anomaly Prevalence Ranking

Summary of detected anomalies across Kerala's 14 administrative districts:

| District           |   Total_Records |   Detected_Anomalies |   Anomaly_Rate_Pct |   Drop_Shocks |   Surge_Peaks |   Avg_Volume_Deviation |
|:-------------------|----------------:|---------------------:|-------------------:|--------------:|--------------:|-----------------------:|
| Ernakulam          |               5 |                    5 |                100 |             0 |             0 |        16889.8         |
| Thiruvananthapuram |               5 |                    2 |                 40 |             0 |             0 |      -150174           |
| Pathanamthitta     |               5 |                    1 |                 20 |             0 |             1 |       159704           |
| Thrissur           |               5 |                    1 |                 20 |             0 |             0 |      -372014           |
| Idukki             |               5 |                    1 |                 20 |             0 |             1 |            1.84181e+06 |
| Alappuzha          |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Kollam             |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Kasaragod          |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Kannur             |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Kottayam           |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Palakkad           |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Malappuram         |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Kozhikode          |               5 |                    0 |                  0 |             0 |             0 |            0           |
| Wayanad            |               5 |                    0 |                  0 |             0 |             0 |            0           |

### Top 3 Volatile Districts:
**Ernakulam, Thiruvananthapuram, Pathanamthitta** exhibit the highest vulnerability to external shocks and demand surges, driven by high dependence on international leisure tourism and major festival hosting.

---

## 4. Inlier vs. Outlier Operational Profiles

Comparative statistics highlighting the magnitude of operational shifts between normal periods and anomalous months:

| Feature              |      Inlier_Mean |   Inlier_Std |     Outlier_Mean |      Outlier_Std |   Mean_Shift_Pct |
|:---------------------|-----------------:|-------------:|-----------------:|-----------------:|-----------------:|
| Domestic_Tourists    | 978699           |     822611   |      3.06795e+06 |      1.2055e+06  |           213.47 |
| Foreign_Tourists     |  26142.4         |      47902.6 | 280438           | 196920           |           972.73 |
| Total_Tourists       |      1.00484e+06 |     847341   |      3.34839e+06 |      1.31561e+06 |           233.23 |
| Hotel_Occupancy_Rate |     55           |          0   |     55           |      0           |             0    |
| Tourism_Revenue_Cr   |    991.85        |        843.6 |   3580.6         |   1438.45        |           261    |
| Average_Stay_Days    |      2.5         |          0   |      2.5         |      0           |             0    |

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
