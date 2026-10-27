# Kerala Tourism Dataset Guide

This directory stores raw and processed tourism datasets for Kerala.

## Overview
Kerala is one of India's premier tourist destinations, renowned for backwaters (Alappuzha, Kumarakom), hill stations (Munnar, Wayanad), cultural hubs (Kochi, Thrissur), and coastal beaches (Kovalam, Varkala). Tourism data collected by the Department of Tourism, Government of Kerala records monthly domestic and foreign footfalls across all 14 administrative districts:

1. Thiruvananthapuram
2. Kollam
3. Pathanamthitta
4. Alappuzha
5. Kottayam
6. Idukki
7. Ernakulam
8. Thrissur
9. Palakkad
10. Malappuram
11. Kozhikode
12. Wayanad
13. Kannur
14. Kasaragod

## Expected Schema

A standard Kerala tourism tabular dataset (`kerala_tourism_data.csv`) contains the following columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Year` | Integer | Observation year (e.g., 2016 - 2024) |
| `Month` | Integer / String | Observation month (1-12 or month name) |
| `District` | String | One of the 14 districts of Kerala |
| `Domestic_Tourists` | Integer | Number of registered domestic visitors |
| `Foreign_Tourists` | Integer | Number of registered international visitors |
| `Total_Tourists` | Integer | Total visitors (`Domestic + Foreign`) |
| `Average_Stay_Days` | Float | Average duration of tourist stay (in days) |
| `Hotel_Occupancy_Rate` | Float | Percentage hotel / resort room occupancy |
| `Tourism_Revenue_Cr` | Float | Estimated direct tourism revenue in Crore INR |
| `Season` | String | Peak, Monsoon, Post-Monsoon, Summer |

## Notable Real-World Outlier Phenomena in Kerala Tourism
- **August 2018 Mega-Floods**: Historic severe deluge causing massive drops in tourist arrivals across almost all districts (especially Idukki, Ernakulam, Alappuzha).
- **2020-2021 COVID-19 Pandemic**: Near-zero foreign arrivals and sharp domestic contractions due to state lockdowns.
- **Festival Peaks (Onam & Thrissur Pooram)**: Sudden dramatic spikes in footfalls during August-September (Onam statewide) and April-May (Thrissur).
- **Winter Peak Season**: December - January surges in Fort Kochi (Biennale / Cochin Carnival) and Kovalam / Varkala.

## Generating Sample Data
If real government statistical reports are not immediately placed in this directory, the project automatically generates a realistic, statistically calibrated historical dataset using:
```bash
python main.py --generate-sample
```
or programmatically using `src.preprocessing.generate_kerala_tourism_dataset()`.
