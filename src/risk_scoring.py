import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "crime_cleaned.csv")
MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")

os.makedirs(MAPS_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
SEVERITY_WEIGHTS = {
    "HOMICIDE":10,
    "SEXUAL ASSAULT": 9,
    "KIDNAPPING":8,
    "FIREARM OFFENSE":8,
    "ARSON":7,
    "ASSAULT":7,
    "ROBBERY":7,
    "EXTORTION":6,
    "DOMESTIC VIOLENCE":6,
    "ILLEGAL POSSESSION":5,
    "DRUG OFFENSE":5,
    "BURGLARY":5,
    "FRAUD":4,
    "IDENTITY THEFT":4,
    "CYBERCRIME":4,
    "VEHICLE STOLEN":3,
    "COUNTERFEITING":3,
    "VANDALISM":3,
    "SHOPLIFTING":2,
    "TRAFFIC VIOLATION":2,
    "PUBLIC INTOXICATION":1,
}

RECENCY_CUTOFF = pd.Timestamp("2024-06-01")

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date_of_Occurrence"])
    df["Severity"] = df["Crime_Description"].map(SEVERITY_WEIGHTS).fillna(1)
    print(f"Data loaded: {df.shape[0]} rows")
    return df

def frequency_score(df):
    freq = df.groupby("City").size().reset_index(name="Crime_Count")
    freq["Freq_Score"] = freq["Crime_Count"] / freq["Crime_Count"].max()
    return freq

def severity_score(df):
    sev = df.groupby("City")["Severity"].sum().reset_index(name="Total_Severity")
    sev["Sev_Score"] = sev["Total_Severity"] / sev["Total_Severity"].max()
    return sev

def recency_score(df):
    recent = df[df["Date_of_Occurrence"] >= RECENCY_CUTOFF]
    rec = recent.groupby("City").size().reset_index(name="Recent_Count")
    rec["Rec_Score"] = rec["Recent_Count"] / rec["Recent_Count"].max()
    return rec
def calculate_risk_index(df):
    freq = frequency_score(df)
    sev = severity_score(df)
    rec = recency_score(df)

    meta = df.groupby("City").agg(Latitude  = ("Latitude",  "first"),Longitude = ("Longitude", "first"),).reset_index()
    risk = freq.merge(sev,  on="City", how="left")
    risk = risk.merge(rec,  on="City", how="left")
    risk = risk.merge(meta, on="City", how="left")
    risk["Rec_Score"] = risk["Rec_Score"].fillna(0)
    risk["Recent_Count"] = risk["Recent_Count"].fillna(0)
    
    risk["Raw_Score"] = (
        0.4 * risk["Freq_Score"] +
        0.4 * risk["Sev_Score"]  +
        0.2 * risk["Rec_Score"]
    )
    risk["Risk_Index"] = (
        risk["Raw_Score"] / risk["Raw_Score"].max() * 100
    ).round(2)

    def get_risk_category(score):
        if score >= 75:
            return "Critical"
        elif score >= 50:
            return "High"
        elif score >= 25:
            return "Medium"
        else:
            return "Low"

    risk["Risk_Category"] = risk["Risk_Index"].apply(get_risk_category)

    risk = risk.sort_values("Risk_Index", ascending=False).reset_index(drop=True)
    risk.index += 1

    return risk

def print_risk_table(risk):
    print("AREA RISK INDEX  ALL CITIES RANKED")
    print(f"{'Rank':<6} {'City':<20} {'Risk Index':>12} {'Category':<12} {'Crimes':>8} {'Recent':>8}")
    for rank, row in risk.iterrows():
        print(
            f"{rank:<6} "
            f"{row['City']:<20} "
            f"{row['Risk_Index']:>12.2f} "
            f"{row['Risk_Category']:<12} "
            f"{int(row['Crime_Count']):>8} "
            f"{int(row['Recent_Count']):>8}"
        )

    print("\nRisk Category Summary:")
    cat_counts = risk["Risk_Category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"{cat:<10}: {count} cities")

def generate_risk_map(risk):
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB positron"
    )
    color_map = {
        "Critical": "#d32f2f",
        "High":     "#f57c00",
        "Medium":   "#fbc02d",
        "Low":      "#388e3c"
    }

    for _, row in risk.iterrows():
        color  = color_map.get(row["Risk_Category"], "gray")
        radius = 6 + (row["Risk_Index"] / 100) * 14

        popup_html = f"""
        <div style="font-family:Arial; font-size:13px; width:220px">
            <b style="font-size:15px">{row['City']}</b><br>
            <hr style="margin:4px 0">
            <b style="color:{color}">Risk Index: {row['Risk_Index']} / 100</b><br>
            Category     : <b>{row['Risk_Category']}</b><br>
            Total Crimes : <b>{int(row['Crime_Count'])}</b><br>
            Recent Crimes: <b>{int(row['Recent_Count'])}</b><br>
            Severity Score: <b>{int(row['Total_Severity'])}</b><br>
            <hr style="margin:4px 0">
            <small>Freq Score : {round(row['Freq_Score'], 3)}</small><br>
            <small>Sev Score  : {round(row['Sev_Score'], 3)}</small><br>
            <small>Rec Score  : {round(row['Rec_Score'], 3)}</small>
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{row['City']} | Risk: {row['Risk_Index']} | {row['Risk_Category']}"
        ).add_to(india_map)

    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                border: 2px solid #ccc; font-family: Arial; font-size: 13px;">
        <b>Risk Index Legend</b><br>
        <span style="color:#d32f2f">&#9679;</span> Critical (75-100)<br>
        <span style="color:#f57c00">&#9679;</span> High     (50-75)<br>
        <span style="color:#fbc02d">&#9679;</span> Medium   (25-50)<br>
        <span style="color:#388e3c">&#9679;</span> Low      (0-25)
    </div>
    """
    india_map.get_root().html.add_child(folium.Element(legend_html))

    map_path = os.path.join(MAPS_DIR, "risk_score_map.html")
    india_map.save(map_path)
    print(f"Risk map saved to: {map_path}")
    return map_path

def generate_risk_chart(risk):
    color_map = {
        "Critical": "#d32f2f",
        "High":     "#f57c00",
        "Medium":   "#fbc02d",
        "Low":      "#388e3c"
    }

    colors = risk["Risk_Category"].map(color_map)

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(
        risk["City"],
        risk["Risk_Index"],
        color=colors,
        edgecolor="white",
        linewidth=0.5
    )
    for bar, val in zip(bars, risk["Risk_Index"]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8
        )

    ax.set_title("Area Risk Index — All Cities Ranked (0-100)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Risk Index Score")
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", label="Critical (75-100)"),
        Patch(facecolor="#f57c00", label="High (50-75)"),
        Patch(facecolor="#fbc02d", label="Medium (25-50)"),
        Patch(facecolor="#388e3c", label="Low (0-25)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, "risk_index.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Risk chart saved to: {chart_path}")
    return chart_path

def save_risk_scores(risk):
    out_path = os.path.join(BASE_DIR, "data", "city_risk_scores.csv")
    risk.to_csv(out_path)
    print(f"Risk scores saved to: {out_path}")

if __name__ == "__main__":
    df   = load_data()
    risk = calculate_risk_index(df)

    print_risk_table(risk)
    generate_risk_map(risk)
    generate_risk_chart(risk)
    save_risk_scores(risk)

    print("\nOutputs generated:")
    print("outputs/maps/risk_score_map.html")
    print("outputs/charts/risk_index.png")
    print("data/city_risk_scores.csv")