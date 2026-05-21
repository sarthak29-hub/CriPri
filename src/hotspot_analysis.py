import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "crime_cleaned.csv")
MAPS_DIR   = os.path.join(BASE_DIR, "outputs", "maps")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(MAPS_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

SEVERITY_WEIGHTS = {
    "HOMICIDE":           10,
    "SEXUAL ASSAULT":      9,
    "KIDNAPPING":          8,
    "FIREARM OFFENSE":     8,
    "ARSON":               7,
    "ASSAULT":             7,
    "ROBBERY":             7,
    "EXTORTION":           6,
    "DOMESTIC VIOLENCE":   6,
    "ILLEGAL POSSESSION":  5,
    "DRUG OFFENSE":        5,
    "BURGLARY":            5,
    "FRAUD":               4,
    "IDENTITY THEFT":      4,
    "CYBERCRIME":          4,
    "VEHICLE - STOLEN":    3,
    "COUNTERFEITING":      3,
    "VANDALISM":           3,
    "SHOPLIFTING":         2,
    "TRAFFIC VIOLATION":   2,
    "PUBLIC INTOXICATION": 1,
}

def load_clean_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date_of_Occurrence"])
    df["Severity"] = df["Crime_Description"].map(SEVERITY_WEIGHTS).fillna(1)
    print(f"Data loaded: {df.shape[0]} rows | Severity scores assigned")
    return df

def calculate_crime_density(df):
    total_crimes = len(df)
    density = df.groupby("City").agg(
        Crime_Count    = ("Crime_Description", "count"),
        Total_Severity = ("Severity", "sum"),
        Avg_Severity   = ("Severity", "mean"),
        Latitude       = ("Latitude", "first"),
        Longitude      = ("Longitude", "first"),
    ).reset_index()

    density["Crime_Density"]   = (density["Crime_Count"] / total_crimes * 100).round(3)
    density["Weighted_Density"]= (density["Crime_Density"] * density["Avg_Severity"]).round(3)
    density = density.sort_values("Weighted_Density", ascending=False).reset_index(drop=True)

    print("\nCity-wise Crime Density Table:")
    print(f"{'City':<20} {'Count':>8} {'Density%':>10} {'Wtd Density':>13} {'Avg Severity':>14}")
    for _, row in density.iterrows():
        print(
            f"   {row['City']:<20}"
            f"{row['Crime_Count']:>8}"
            f"{row['Crime_Density']:>10}"
            f"{row['Weighted_Density']:>13}"
            f"{row['Avg_Severity']:>14.3f}"
        )
    return density

def run_dbscan_clustering(city_summary):
    print("\nStretch Goal: Running DBSCAN clustering")
    coords     = city_summary[["Latitude", "Longitude"]].values
    coords_rad = np.radians(coords)
    epsilon    = 300 / 6371
    db = DBSCAN(eps=epsilon,min_samples=2,algorithm="ball_tree",metric="haversine").fit(coords_rad)

    city_summary         = city_summary.copy()
    city_summary["Cluster"] = db.labels_

    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_isolated = list(db.labels_).count(-1)

    print(f"\nGeographic clusters found : {n_clusters}")
    print(f"Isolated cities: {n_isolated}")

    cluster_summary = city_summary.groupby("Cluster").agg(
        Cities       = ("City", lambda x: ", ".join(x)),
        Total_Crimes = ("Crime_Count", "sum"),
        Avg_Severity = ("Avg_Severity", "mean")
    ).reset_index()

    for _, row in cluster_summary.iterrows():
        label = "Isolated" if row["Cluster"] == -1 else f"Zone {int(row['Cluster'])}"
        print(f"\n{label}:")
        print(f"Cities       : {row['Cities']}")
        print(f"Total Crimes : {row['Total_Crimes']}")
        print(f"Avg Severity : {round(row['Avg_Severity'], 3)}")

    top2= city_summary.nlargest(2, "Crime_Count")
    city_a = (top2.iloc[0]["Latitude"], top2.iloc[0]["Longitude"])
    city_b = (top2.iloc[1]["Latitude"], top2.iloc[1]["Longitude"])
    dist= geodesic(city_a, city_b).km
    print(f"\nDistance between top 2 crime cities")
    print(f"{top2.iloc[0]['City']} to {top2.iloc[1]['City']}: {dist:.1f} km")

    return city_summary

def identify_top_risky_zones(df, city_summary):
    top10 = city_summary.head(10)
    print("\n   Top 10 Risky Zones:")
    print(f"   {'Rank':<6} {'City':<20} {'Crimes':>8} {'Severity Score':>16}")
    print("   " + "-" * 55)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"   {rank:<6} {row['City']:<20} {row['Crime_Count']:>8} {row['Total_Severity']:>16}")

    dominant = df.groupby(["City", "Crime_Description"]).size().reset_index(name="Count")
    dominant = dominant.sort_values("Count", ascending=False)
    dominant = dominant.groupby("City").first().reset_index()
    dominant = dominant.rename(columns={"Crime_Description": "Dominant_Crime"})
    print("\nDominant crime type per city:")
    print(dominant[["City", "Dominant_Crime", "Count"]].to_string(index=False))

    _print_top5_insights(city_summary, df)
    _chart_monthly_trend(df)
    _chart_day_of_week(df)
    _chart_season_wise(df)
    _chart_city_crime_heatmap(df)
    _chart_time_vs_crime(df)
    _chart_top_zones(city_summary)
    return top10, dominant

def _print_top5_insights(city_summary, df):
    print("TOP 5 MOST DANGEROUS ZONES")
    closure = df.groupby("City").apply(
        lambda x: round((x["Case_Closed"] == "Yes").sum() / len(x) * 100, 2)
    ).reset_index()
    closure.columns = ["City", "Closure_Rate"]
    dominant = df.groupby(["City", "Crime_Description"]).size().reset_index(name="Count")
    dominant = dominant.sort_values("Count", ascending=False).groupby("City").first().reset_index()

    top5 = city_summary.head(5).merge(closure, on="City").merge(
        dominant[["City", "Crime_Description"]], on="City"
    )
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"\n  Rank {rank} - {row['City']}")
        print(f"  Total Crimes     : {row['Crime_Count']}")
        print(f"  Crime Density    : {row['Crime_Density']}% of all India crimes")
        print(f"  Severity Score   : {row['Total_Severity']}")
        print(f"  Avg Severity     : {round(row['Avg_Severity'], 2)} / 10")
        print(f"  Dominant Crime   : {row['Crime_Description']}")
        print(f"  Case Closure Rate: {row['Closure_Rate']}%")
        print(f"  Weighted Density : {row['Weighted_Density']}")


def _chart_monthly_trend(df):
    monthly = df.groupby("Month").size().reset_index(name="Crime_Count")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["Month_Name"] = monthly["Month"].apply(lambda x: month_names[x - 1])
    plt.figure(figsize=(12, 5))
    plt.plot(monthly["Month_Name"], monthly["Crime_Count"],marker="o", color="crimson", linewidth=2.5, markersize=7)
    plt.fill_between(monthly["Month_Name"], monthly["Crime_Count"],alpha=0.15, color="crimson")
    plt.title("Monthly Crime Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Number of Crimes")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for i, row in monthly.iterrows():
        plt.text(i, row["Crime_Count"] + 30, str(row["Crime_Count"]),
                 ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "monthly_trend.png"), dpi=150)
    plt.close()

def _chart_day_of_week(df):
    day_order  = ["Monday", "Tuesday", "Wednesday", "Thursday","Friday", "Saturday", "Sunday"]
    day_counts = df["Day_Name"].value_counts().reindex(day_order)
    colors = ["#c0392b" if d in ["Saturday", "Sunday"] else "steelblue" for d in day_order]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(day_order, day_counts.values, color=colors, edgecolor="black")
    plt.title("Crime Distribution by Day of Week", fontsize=14, fontweight="bold")
    plt.xlabel("Day of Week")
    plt.ylabel("Number of Crimes")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, day_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30, str(val),
                 ha="center", fontsize=9)

    plt.annotate("Red bars = Weekends",xy=(0.78, 0.92), xycoords="axes fraction",fontsize=9, color="crimson")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "day_of_week.png"), dpi=150)
    plt.close()


def _chart_season_wise(df):
    season_order  = ["Summer", "Monsoon", "Post-Monsoon", "Winter"]
    season_colors = {"Summer":"#e74c3c", "Monsoon":"#2980b9","Post-Monsoon": "#27ae60","Winter":"#8e44ad"}
    season_counts = df["Season"].value_counts().reindex(season_order)
    top6_crimes = df["Crime_Description"].value_counts().head(6).index
    season_crime = df[df["Crime_Description"].isin(top6_crimes)].groupby(["Season", "Crime_Description"]).size().unstack(fill_value=0).reindex(season_order)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Season-wise Crime Analysis", fontsize=14, fontweight="bold")
    bar_colors = [season_colors[s] for s in season_order]
    axes[0].bar(season_order, season_counts.values,color=bar_colors, edgecolor="black")
    axes[0].set_title("Total Crimes by Season")
    axes[0].set_ylabel("Number of Crimes")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(season_counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontsize=9)
    season_crime.plot(kind="bar", ax=axes[1],colormap="tab10", edgecolor="black")
    axes[1].set_title("Top Crime Types by Season")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Number of Crimes")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Crime Type", bbox_to_anchor=(1.01, 1),loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "season_wise_crimes.png"),dpi=150, bbox_inches="tight")
    plt.close()


def _chart_city_crime_heatmap(df):
    top_cities = df["City"].value_counts().head(15).index
    top_crimes = df["Crime_Description"].value_counts().head(10).index
    pivot= df.groupby(["City", "Crime_Description"]).size().unstack(fill_value=0)
    pivot= pivot.loc[pivot.index.isin(top_cities),pivot.columns.isin(top_crimes)]
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5, annot=True,fmt="d", cbar_kws={"label": "Crime Count"})
    plt.title("Crime Type Distribution Across Top Cities",fontsize=14, fontweight="bold")
    plt.xlabel("Crime Type")
    plt.ylabel("City")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "city_crime_heatmap.png"), dpi=150)
    plt.close()


def _chart_time_vs_crime(df):
    top_crimes = df["Crime_Description"].value_counts().head(8).index
    df_top= df[df["Crime_Description"].isin(top_crimes)]
    pivot= df_top.groupby(["Time_Bucket", "Crime_Description"]).size().unstack(fill_value=0)
    pivot= pivot.reindex(["Morning", "Afternoon", "Evening", "Night"])
    pivot.plot(kind="bar", figsize=(14, 6), colormap="tab10", edgecolor="black")
    plt.title("Crime Type by Time of Day", fontsize=14, fontweight="bold")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Crimes")
    plt.xticks(rotation=0)
    plt.legend(title="Crime Type", bbox_to_anchor=(1.01, 1),loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "time_vs_crime.png"), dpi=150)
    plt.close()

def _chart_top_zones(city_summary):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Top Risky Zones", fontsize=15, fontweight="bold")
    top15_count    = city_summary.nlargest(15, "Crime_Count")
    top15_severity = city_summary.nlargest(15, "Total_Severity")
    axes[0].barh(top15_count["City"], top15_count["Crime_Count"], color="steelblue")
    axes[0].set_title("By Crime Volume")
    axes[0].set_xlabel("Number of Crimes")
    axes[0].invert_yaxis()
    for i, v in enumerate(top15_count["Crime_Count"]):
        axes[0].text(v + 10, i, str(v), va="center", fontsize=8)

    axes[1].barh(top15_severity["City"], top15_severity["Total_Severity"], color="crimson")
    axes[1].set_title("By Weighted Severity Score")
    axes[1].set_xlabel("Total Severity Score")
    axes[1].invert_yaxis()
    for i, v in enumerate(top15_severity["Total_Severity"]):
        axes[1].text(v + 10, i, str(v), va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "top_risky_zones.png"), dpi=150)
    plt.close()

def generate_heatmap(df, city_summary):

    closure = df.groupby("City").apply(
        lambda x: round((x["Case_Closed"] == "Yes").sum() / len(x) * 100, 2)
    ).reset_index()
    closure.columns = ["City", "Closure_Rate"]
    city_summary = city_summary.merge(closure, on="City", how="left")

    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    heat_data = df[["Latitude", "Longitude", "Severity"]].values.tolist()
    HeatMap(
        heat_data,
        name="Crime Heatmap",
        min_opacity=0.4,
        radius=35,
        blur=25,
        max_zoom=6,
        gradient={
            0.2: "blue",
            0.4: "lime",
            0.6: "orange",
            0.8: "red",
            1.0: "darkred"
        }
    ).add_to(india_map)

    cluster_colors = ["red", "blue", "green", "purple","orange", "darkred", "cadetblue", "darkgreen"]

    for _, row in city_summary.iterrows():
        cluster_id = int(row["Cluster"])
        color      = "gray" if cluster_id == -1 else cluster_colors[cluster_id % len(cluster_colors)]

        popup_html = f"""
        <div style="font-family:Arial; font-size:13px; width:210px">
            <b style="font-size:15px">{row['City']}</b><br>
            <hr style="margin:4px 0">
            Total Crimes     : <b>{row['Crime_Count']}</b><br>
            Crime Density    : <b>{row['Crime_Density']}%</b><br>
            Severity Score   : <b>{row['Total_Severity']}</b><br>
            Avg Severity     : <b>{round(row['Avg_Severity'], 2)} / 10</b><br>
            Case Closure Rate: <b>{row.get('Closure_Rate', 'N/A')}%</b><br>
            Cluster Zone     : <b>{'Isolated' if cluster_id == -1 else f'Zone {cluster_id}'}</b>
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=230),
            tooltip=f"{row['City']} | Crimes: {row['Crime_Count']} | Density: {row['Crime_Density']}%"
        ).add_to(india_map)

    folium.LayerControl().add_to(india_map)

    map_path = os.path.join(MAPS_DIR, "crime_heatmap.html")
    india_map.save(map_path)
    print(f"Heatmap saved to: {map_path}")
    return map_path

if __name__ == "__main__":
    print("HOTSPOT ANALYSIS")
    df = load_clean_data()
    city_summary = calculate_crime_density(df)
    city_summary = run_dbscan_clustering(city_summary)
    top10, dom   = identify_top_risky_zones(df, city_summary)
    generate_heatmap(df, city_summary)


    print("\nOutputs generated:")
    print("outputs/maps/crime_heatmap.html")
    print("outputs/charts/monthly_trend.png")
    print("outputs/charts/day_of_week.png")
    print("outputs/charts/season_wise_crimes.png")
    print("outputs/charts/city_crime_heatmap.png")
    print("outputs/charts/time_vs_crime.png")
    print("outputs/charts/top_risky_zones.png")