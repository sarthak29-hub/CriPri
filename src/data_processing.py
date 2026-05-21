import pandas as pd
import numpy as np
import os
import time
import json
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "indian_crime_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
CACHE_PATH = os.path.join(BASE_DIR, "data", "geocode_cache.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

INDIA_LAT_MIN, INDIA_LAT_MAX =  6.0,  38.0
INDIA_LON_MIN, INDIA_LON_MAX = 68.0,  98.0

WEAPON_GROUPS = {
    "Firearm":    ["Firearm", "Gun", "Pistol", "Rifle", "Shotgun"],
    "Blunt":      ["Blunt Object", "Bat", "Rod", "Stick"],
    "Sharp":      ["Knife", "Blade", "Sword", "Sharp Object"],
    "Chemical":   ["Poison", "Acid", "Chemical"],
    "Explosive":  ["Bomb", "Explosive", "Grenade"],
    "Unknown":    ["Unknown"],
}

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def parse_mixed_dates(series):
    parsed = pd.to_datetime(series, format="%d-%m-%Y %H:%M", errors="coerce")
    failed_mask = parsed.isna()
    if failed_mask.sum() > 0:
        retry = pd.to_datetime(
            series[failed_mask],
            format="%m-%d-%Y %H:%M",
            errors="coerce"
        )
        parsed[failed_mask] = retry
    return parsed

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

def geocode_cities(city_series):
    geolocator      = Nominatim(user_agent="crime_hotspot_project_v1")
    cache           = load_cache()
    unique_cities   = city_series.dropna().unique().tolist()
    cities_to_fetch = [c for c in unique_cities if c not in cache]

    print(f"\nGeocoding cities:")
    print(f"Total unique cities : {len(unique_cities)}")
    print(f"Found in cache      : {len(unique_cities) - len(cities_to_fetch)}")
    print(f"Need API calls      : {len(cities_to_fetch)}")

    if cities_to_fetch:
        print(f"Fetching from Nominatim (1 request/sec)")
        for city in cities_to_fetch:
            try:
                location = geolocator.geocode(f"{city}, India", timeout=10)
                if location:
                    cache[city] = (
                        round(location.latitude, 4),
                        round(location.longitude, 4)
                    )
                    print(f"   {city:25s} -> {cache[city]}")
                else:
                    cache[city] = None
                    print(f"   {city:25s} -> Not found")
                time.sleep(1)
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"   {city:25s} -> Error: {e}")
                cache[city] = None
        save_cache(cache)
        print(f"Cache saved to: {CACHE_PATH}")
    else:
        print(f"All cities loaded from cache instantly")
    return {
        city: coords
        for city, coords in cache.items()
        if coords is not None
    }

def clean_data(df):
    print("\nStarting data cleaning...")
    cleaning_log = {}
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    print("Columns standardized")
    before  = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    cleaning_log["duplicates_removed"] = dropped
    print(f"Duplicates removed: {dropped} rows dropped")
    df["Date_of_Occurrence"] = parse_mixed_dates(df["Date_of_Occurrence"])
    invalid_dates = df["Date_of_Occurrence"].isna().sum()
    cleaning_log["invalid_dates_dropped"] = int(invalid_dates)
    print(f"Dates parsed | Still invalid: {invalid_dates}")
    df.dropna(subset=["Date_of_Occurrence"], inplace=True)
    df["Time_of_Occurrence_dt"] = parse_mixed_dates(df["Time_of_Occurrence"])
    df["Hour"] = df["Time_of_Occurrence_dt"].dt.hour
    print(f"Hour extracted | Missing hours: {df['Hour'].isna().sum()}")
    df["Month"]=df["Date_of_Occurrence"].dt.month
    df["Day_of_Week"]=df["Date_of_Occurrence"].dt.dayofweek
    df["Year"] = df["Date_of_Occurrence"].dt.year
    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)
    print("Core date features extracted")
    years = df["Year"].value_counts().sort_index()
    print(f"Year distribution:\n{years.to_string()}")
    day_map = {0: "Monday",   1: "Tuesday",  2: "Wednesday",3: "Thursday", 4: "Friday",   5: "Saturday",  6: "Sunday"}
    df["Day_Name"] = df["Day_of_Week"].map(day_map)
    print("Day names added")

    def get_season(month):
        if month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        elif month in [10, 11]:
            return "Post-Monsoon"
        else:
            return "Winter"
    df["Season"] = df["Month"].apply(get_season)
    print(f"Seasons assigned:\n{df['Season'].value_counts().to_string()}")

    def get_time_bucket(hour):
        if pd.isna(hour):
            return "Unknown"
        hour = int(hour)
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    df["Time_Bucket"] = df["Hour"].apply(get_time_bucket)
    print(f"Time buckets:\n{df['Time_Bucket'].value_counts().to_string()}")

    def get_age_group(age):
        if pd.isna(age):
            return "Unknown"
        age = int(age)
        if age < 18:
            return "Child"
        elif age < 30:
            return "Young Adult"
        elif age < 60:
            return "Adult"
        else:
            return "Senior"

    df["Victim_Age_Group"] = df["Victim_Age"].apply(get_age_group)
    print(f"Victim age groups:\n{df['Victim_Age_Group'].value_counts().to_string()}")
    print(f"Victim gender distribution:\n{df['Victim_Gender'].value_counts().to_string()}")

    df["Weapon_Used"] = df["Weapon_Used"].fillna("Unknown")

    def get_weapon_group(weapon):
        weapon = str(weapon)
        for group, keywords in WEAPON_GROUPS.items():
            for keyword in keywords:
                if keyword.lower() in weapon.lower():
                    return group
        return "Other"

    df["Weapon_Group"] = df["Weapon_Used"].apply(get_weapon_group)
    print(f"Weapon groups:\n{df['Weapon_Group'].value_counts().to_string()}")

    coord_lookup = geocode_cities(df["City"])

    df["Latitude"] = df["City"].map(
        lambda c: coord_lookup.get(c, (np.nan, np.nan))[0]
        if coord_lookup.get(c) else np.nan
    )
    df["Longitude"] = df["City"].map(
        lambda c: coord_lookup.get(c, (np.nan, np.nan))[1]
        if coord_lookup.get(c) else np.nan
    )

    before_validation = len(df)
    invalid_coords = (
        (df["Latitude"]  < INDIA_LAT_MIN) | (df["Latitude"]  > INDIA_LAT_MAX) |
        (df["Longitude"] < INDIA_LON_MIN) | (df["Longitude"] > INDIA_LON_MAX)
    )
    if invalid_coords.sum() > 0:
        print(f"Invalid coordinates outside India: {invalid_coords.sum()} rows")
        df = df[~invalid_coords]
    else:
        print(f"All coordinates validated within India bounding box")

    unmapped = df["Latitude"].isna().sum()
    cleaning_log["unmapped_cities"] = int(unmapped)
    if unmapped > 0:
        print(f"   Cities not geocoded ({unmapped} rows):")
        print(df[df["Latitude"].isna()]["City"].value_counts())

    df.dropna(subset=["Latitude", "Longitude"], inplace=True)

    df.drop(columns=["Time_of_Occurrence_dt"], inplace=True)

    cleaning_log["final_rows"]    = len(df)
    cleaning_log["final_columns"] = len(df.columns)

    print(f"\nCleaning complete! Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, cleaning_log

def police_deployment_analysis(df):

    print("\nPolice Deployment Analysis:")
    city_deploy = df.groupby("City")["Police_Deployed"].agg(
        Avg_Deployed  = "mean",
        Max_Deployed  = "max",
        Total_Deployed= "sum").round(2).sort_values("Avg_Deployed", ascending=False)

    print("\nAverage police deployed per city (top 10):")
    print(city_deploy.head(10).to_string())

    crime_deploy = df.groupby("Crime_Description")["Police_Deployed"].mean(
    ).round(2).sort_values(ascending=False)
    print("\nAverage police deployed per crime type:")
    print(crime_deploy.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Police Deployment Analysis", fontsize=14, fontweight="bold")

    top10_deploy = city_deploy.head(10)
    axes[0].barh(top10_deploy.index, top10_deploy["Avg_Deployed"], color="navy")
    axes[0].set_title("Avg Police Deployed per City")
    axes[0].set_xlabel("Average Officers Deployed")
    axes[0].invert_yaxis()

    crime_deploy.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="black")
    axes[1].set_title("Avg Police Deployed per Crime Type")
    axes[1].set_xlabel("Crime Type")
    axes[1].set_ylabel("Average Officers")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "police_deployment.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPolice deployment chart saved to: {path}")

    return city_deploy, crime_deploy

def case_closure_rate(df):
    closure = df.groupby("City").apply(
        lambda x: round((x["Case_Closed"] == "Yes").sum() / len(x) * 100, 2)
    ).reset_index()
    closure.columns = ["City", "Case_Closure_Rate_%"]
    closure = closure.sort_values("Case_Closure_Rate_%", ascending=False)

    print("\nCase closure rate by city (top 10):")
    print(closure.head(10).to_string())
    return closure

def save_data_quality_report(df, cleaning_log):
    report_rows = []
    for col in df.columns:
        missing     = df[col].isna().sum()
        missing_pct = round(missing / len(df) * 100, 2)
        unique_vals = df[col].nunique()
        dtype       = str(df[col].dtype)
        report_rows.append({
            "Column":        col,
            "Data_Type":     dtype,
            "Missing_Count": missing,
            "Missing_%":     missing_pct,
            "Unique_Values": unique_vals,
        })

    report_df    = pd.DataFrame(report_rows)
    summary_rows = pd.DataFrame([
        {"Column": "-- CLEANING SUMMARY --", "Data_Type": "", "Missing_Count": "",   "Missing_%": "", "Unique_Values": ""},
        {"Column": "Duplicates Removed",     "Data_Type": "", "Missing_Count": cleaning_log.get("duplicates_removed", 0),    "Missing_%": "", "Unique_Values": ""},
        {"Column": "Invalid Dates Dropped",  "Data_Type": "", "Missing_Count": cleaning_log.get("invalid_dates_dropped", 0), "Missing_%": "", "Unique_Values": ""},
        {"Column": "Unmapped Cities",        "Data_Type": "", "Missing_Count": cleaning_log.get("unmapped_cities", 0),       "Missing_%": "", "Unique_Values": ""},
        {"Column": "Final Row Count",        "Data_Type": "", "Missing_Count": cleaning_log.get("final_rows", 0),            "Missing_%": "", "Unique_Values": ""},
        {"Column": "Final Column Count",     "Data_Type": "", "Missing_Count": cleaning_log.get("final_columns", 0),         "Missing_%": "", "Unique_Values": ""},
    ])

    full_report = pd.concat([report_df, summary_rows], ignore_index=True)
    report_path = os.path.join(OUTPUT_DIR, "data_quality_report.csv")
    full_report.to_csv(report_path, index=False)
    print(f"\nData quality report saved to: {report_path}")
    return report_path

def save_clean_data(df):
    out_path = os.path.join(BASE_DIR, "data", "crime_cleaned.csv")
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to: {out_path}")

if __name__ == "__main__":
    df = load_data()
    df_clean, log = clean_data(df)
    police_deployment_analysis(df_clean)
    closure_df = case_closure_rate(df_clean)
    save_data_quality_report(df_clean, log)
    save_clean_data(df_clean)

    print("\nSample of enhanced clean rows:")
    print(df_clean[[ "City", "Crime_Description", "Hour", "Time_Bucket","Day_Name", "Season", "Victim_Age_Group", "Weapon_Group","Month", "Is_Weekend", "Latitude", "Longitude"
    ]].head(8).to_string())