# CriPri — Crime Hotspot Prediction & Risk Heatmap

> A crime intelligence prototype that processes historical crime data,
> identifies hotspots, calculates area-wise risk scores, predicts crime
> likelihood, and visualizes results on an interactive GIS-based dashboard.

## Project Overview

CriPri (Crime Prediction) is an internship project demonstrating real-world
application of data science, machine learning, and GIS visualization in
public safety analytics.

The system ingests raw crime CSV data, cleans and enriches it, identifies
geographic crime clusters, scores areas by risk level, builds ML prediction
models, and presents findings through an interactive Streamlit dashboard.

## Project Status

- Phase 1 - Data Processing & Feature Engineering   : Complete
- Phase 2 - Hotspot Analysis & GIS Visualization    : Complete
- Phase 3 - Risk Scoring Engine (0-100 Index)       : Complete
- Phase 4 - ML Prediction Model                     : Complete
- Phase 5 - Streamlit Interactive Dashboard         : Complete

## Dataset

- Source: [Indian Crimes Dataset - Kaggle](https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset)
- Records: 40,160 crime incidents
- Cities: 29 major Indian cities
- Period: 2020 to 2024
- Features: Crime type, date, time, city, victim profile, weapon used, police deployed, case closure status

## Tech Stack

- Core: Python 3.13, Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost
- GIS and Mapping: Folium, Geopy
- Visualization: Matplotlib, Seaborn
- Dashboard: Streamlit
- Clustering: DBSCAN (Scikit-learn)

## Project Structure

```
CriPri/
│
├── data/
│   ├── indian_crime_data.csv          Raw dataset
│   ├── crime_cleaned.csv              Processed dataset
│   ├── city_risk_scores.csv           Phase 3 risk scores
│   └── geocode_cache.json             Cached city coordinates
│
├── src/
│   ├── data_processing.py             Phase 1: Cleaning and feature engineering
│   ├── hotspot_analysis.py            Phase 2: Hotspot analysis and mapping
│   ├── risk_scoring.py                Phase 3: Risk scoring engine
│   ├── ml_model.py                    Phase 4: ML prediction models
│   └── dashboard.py                   Phase 5: Streamlit dashboard
│
├── outputs/
│   ├── maps/
│   │   ├── crime_heatmap.html         Interactive crime heatmap
│   │   └── risk_score_map.html        Color coded risk zone map
│   └── charts/
│       ├── monthly_trend.png
│       ├── day_of_week.png
│       ├── season_wise_crimes.png
│       ├── city_crime_heatmap.png
│       ├── time_vs_crime.png
│       ├── top_risky_zones.png
│       ├── police_deployment.png
│       ├── risk_index.png
│       ├── ml_evaluation.png
│       └── roc_curve.png
│
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/sarthak29-hub/CriPri.git
cd CriPri
```

2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place inside data/ folder as indian_crime_data.csv

5. Run Data Processing

```bash
python src/data_processing.py
```

6. Run Hotspot Analysis

```bash
python src/hotspot_analysis.py
```

7. Run Risk Scoring

```bash
python src/risk_scoring.py
```

8. Run ML Model

```bash
python src/ml_model.py
```

9. Run Dashboard

```bash
python -m streamlit run src/dashboard.py
```

10. Open browser at http://localhost:8501

## Data Processing

- Loaded and standardized 40,160 raw crime records
- Diagnosed and fixed mixed date format bug (DD-MM-YYYY and MM-DD-YYYY)
- Auto-geocoded city names to coordinates using Geopy Nominatim with local cache
- Validated all coordinates within India bounding box
- Engineered 8 new features: Hour, Time Bucket, Day Name, Season, Is Weekend, Victim Age Group, Weapon Group, Coordinates
- Analyzed police deployment per city and crime type
- Calculated case closure rate per city
- Saved data quality report

## Hotspot Analysis

- Assigned severity weights to 21 crime types on a 1 to 10 scale
- Calculated absolute, relative, and weighted crime density per city
- Applied DBSCAN geographic clustering using Haversine distance
- Auto-identified 4 geographic crime zones without manual zone definition
- Generated interactive Folium heatmap with severity weighting
- Built 7 analytical charts
- Identified dominant crime type per city

## Risk Scoring Engine

- Designed formula combining Frequency (0.4), Severity (0.4), Recency (0.2)
- Normalized all components to 0 to 1 before combining
- Scaled final score to 0 to 100 with highest city anchored at 100
- Categorized all cities as Critical, High, Medium, or Low
- Generated color coded interactive risk map
- Exported scores for use in Phase 4 ML model

## ML Prediction Model

- Defined binary classification target: Violent vs Non-Violent crime
- Identified and fixed data leakage in two iterations
- Used 9 pre-crime context features with no outcome-derived information
- Trained Random Forest and XGBoost classifiers
- Evaluated on Accuracy, Precision, Recall, F1, ROC-AUC
- Random Forest selected as best model based on Recall performance

## Streamlit Dashboard

- 7 interactive tabs: Maps, Risk Zones, Analytics, City Compare, Victim and Weapon, Season and Time, Export
- 6 real-time filters: City, Crime Type, Time of Day, Season, Risk Category, Year Range
- 6 KPI cards updating live with filters
- 3 map options: Crime Heatmap, Risk Zone Map, Live Filtered Map
- City comparison tab for side by side analysis
- Hour vs Day heatmap showing crime intensity grid
- CSV download of filtered data

## Key Findings

- Delhi: 100/100 Risk Index, 13.4% of all India crimes
- Mumbai: 81.62/100 Risk Index, dominant crime Sexual Assault
- North India zone: highest combined crimes at 13,475
- Night crimes 13,278 nearly double Evening crimes 6,813
- Thane: best case closure rate at 53.68%
- Police deployment flat at 10 officers regardless of crime severity
- Random Forest outperformed XGBoost on Recall: 0.5026 vs 0.1092

## Geographic Crime Zones - DBSCAN

- Zone 0 North India: Delhi, Jaipur, Lucknow, Kanpur, Ludhiana, Agra, Ghaziabad, Patna, Meerut, Faridabad, Varanasi: 13,475 crimes
- Zone 1 West India: Mumbai, Pune, Ahmedabad, Surat, Thane, Nashik, Vasai, Kalyan, Rajkot: 11,664 crimes
- Zone 2 South India: Bangalore, Chennai: 6,081 crimes
- Zone 3 Central India: Nagpur, Indore, Bhopal: 2,442 crimes
- Isolated: Hyderabad, Kolkata, Visakhapatnam, Srinagar: 6,498 crimes

## ML Model Results

- Metric  Random Forest XGBoost
- Accuracy:  0.5063     0.5576
- Precision: 0.4346     0.4367
- Recall:    0.5026     0.1092
- F1 Score:  0.4661     0.1747
- ROC-AUC:   0.5052     0.5048

Note: Scores near 0.50 confirm no data leakage. Dataset is synthetically generated so no real patterns exist for models to learn. On real crime data scores would be 65 to 80 percent.

## Planned Improvements

- Nearest police station recommendation using OpenStreetMap
- Weather and festival correlation with crime spikes
- Patrol route optimization using graph theory
- Next-hour crime probability prediction
- Replace synthetic dataset with real crime records

## Author

Sarthak Aggarwal
-Internship Project
-Built with Python, Pandas, Scikit-learn, XGBoost, Folium, Geopy, Streamlit
-GitHub: https://github.com/sarthak29-hub/CriPri