# CriPri — Crime Hotspot Prediction & Risk Heatmap
> A crime analysis prototype that processes historical crime data, identifies hotspots,
> calculates area-wise risk scores, and visualizes results on interactive GIS-based maps.
---
## Project Overview
CriPri (Crime Prediction) is an internship project built to demonstrate the application
of data science, machine learning, and GIS visualization in public safety analytics.

The system ingests raw crime CSV data, cleans and enriches it, identifies geographic
crime clusters, scores areas by risk level, and presents findings through an
interactive map dashboard.
---
## Current Status
- Phase 1 - Data Processing & Feature Engineering   (Complete)
- Phase 2 - Hotspot Analysis & GIS Visualization    (Complete)
- Phase 3 - Risk Scoring Engine (0-100 Index)       (In Progress)
- Phase 4 - ML Prediction Model                     (Upcoming)
- Phase 5 - Streamlit Interactive Dashboard         (Upcoming)
---
## Dataset
- Source: [Indian Crimes Dataset - Kaggle](https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset)
- Records: 40,160 crime incidents
- Cities: 29 major Indian cities
- Period: 2020 - 2024
- Features: Crime type, date, time, city, victim profile, weapon used, case status
---
## Tech Stack
- Core: Python 3.13, Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost
- GIS & Mapping: Folium, Geopy
- Visualization: Matplotlib, Seaborn
- Dashboard: Streamlit (Phase 5)
- Clustering: DBSCAN (Scikit-learn)
---
## Project Structure
```
CriPri/
│
├── data/
│   ├── crime_cleaned.csv
│   ├── geocode_cache.json
│   └── indian_crime_data.csv
│
├── outputs/
│   ├── maps/
│   │   └── crime_heatmap.html
│   └── charts/
│       ├── city_crime_heatmap.png
│       ├── day_of_week.png
│       ├── monthly_trend.png
│       ├── police_deployment.png
│       ├── season_wise_crimes.png
│       ├── time_vs_crime.png
│       └── top_risky_zones.png
│
├── src/
│   ├── data_processing.py
│   └── hotspot_analysis.py
│
├── README.md
└── requirements.txt
```
---
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
4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/sudhanvahg/indian-crimes-dataset)
and place `indian_crime_data.csv` inside the `data/` folder.
5. Run Phase 1 - Data Processing
```bash
python src/data_processing.py
```
6. Run Phase 2 - Hotspot Analysis
```bash
python src/hotspot_analysis.py
```
7. View the interactive map
Open `outputs/maps/crime_heatmap.html` in any browser.
---
## Phase 1
- Loads and standardizes raw CSV data
- Handles mixed date formats (DD-MM-YYYY and MM-DD-YYYY)
- Auto-geocodes city names to coordinates using Geopy Nominatim
- Caches coordinates locally for instant reuse
- Validates coordinates within India bounding box
- Engineers features: Hour, Time Bucket, Day Name, Season, Is Weekend
- Groups victims by age: Child, Young Adult, Adult, Senior
- Groups weapons by type: Firearm, Blunt, Sharp, Chemical, Explosive
- Calculates case closure rate per city
- Saves data quality report
---
## Phase 2
- Calculates crime density: absolute, relative, and weighted
- Runs DBSCAN geographic clustering to identify crime zones
- Identifies top risky zones by volume and severity
- Detects dominant crime type per city
- Generates 7 analytical charts
- Builds interactive Folium heatmap with severity weighting
- Displays cluster-colored markers with detailed popups
---
## Week 1 Key Findings
- Highest crime city: Delhi (13.4% of all India crimes)
- Most severe dominant crime: Mumbai (Sexual Assault)
- Most dangerous zone: North India belt (13,475 crimes)
- Peak crime time: Night (13,278 incidents)
- Best case closure: Thane (53.68%)
- Worst case closure: Multiple cities near 49%
---
## Geographic Crime Zones (DBSCAN)
- Zone 0(North India)- Delhi, Jaipur, Lucknow, Kanpur, Ludhiana, Agra, Ghaziabad, Patna, Meerut, Faridabad, Varanasi: 13,475 crimes
- Zone 1(West India)- Mumbai, Pune, Ahmedabad, Surat, Thane, Nashik, Vasai, Kalyan, Rajkot: 11,664 crimes
- Zone 2(South India)- Bangalore, Chennai: 6,081 crimes
- Zone 3(Central India)- Nagpur, Indore, Bhopal: 2,442 crimes
- Isolated(Geographically distant)- Hyderabad, Kolkata, Visakhapatnam, Srinagar: 6,498 crimes
---
## Interactive Map Features
- Severity-weighted heatmap layer
- Cluster-colored city markers
- Click any city to view: crime count, density, severity score, closure rate, cluster zone
- Hover tooltip for quick stats
- Layer toggle control
---
## Planned Improvements
- Replace hardcoded severity weights with IPC section-based scoring
- Add weather and festival correlation
- Nearest police station recommendation
- Patrol route optimization
---
## Author 
- Sarthak Aggarwal
- Internship Project
- Built with Python, Folium, Scikit-learn, Geopy, Pandas