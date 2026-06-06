# src/dashboard.py
# Phase 5 - CriPri Streamlit Dashboard - Full Enhanced Version

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import tempfile
import streamlit.components.v1 as components
from io import BytesIO

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="CriPri — Crime Intelligence Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ──────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "crime_cleaned.csv")
RISK_PATH      = os.path.join(BASE_DIR, "data", "city_risk_scores.csv")
HEATMAP_PATH   = os.path.join(BASE_DIR, "outputs", "maps", "crime_heatmap.html")
RISK_MAP_PATH  = os.path.join(BASE_DIR, "outputs", "maps", "risk_score_map.html")

# ── Constants ──────────────────────────────────────────────
SEVERITY_WEIGHTS = {
    "HOMICIDE": 10, "SEXUAL ASSAULT": 9,
    "KIDNAPPING": 8, "FIREARM OFFENSE": 8,
    "ARSON": 7, "ASSAULT": 7, "ROBBERY": 7,
    "EXTORTION": 6, "DOMESTIC VIOLENCE": 6,
    "ILLEGAL POSSESSION": 5, "DRUG OFFENSE": 5, "BURGLARY": 5,
    "FRAUD": 4, "IDENTITY THEFT": 4, "CYBERCRIME": 4,
    "VEHICLE - STOLEN": 3, "COUNTERFEITING": 3, "VANDALISM": 3,
    "SHOPLIFTING": 2, "TRAFFIC VIOLATION": 2, "PUBLIC INTOXICATION": 1,
}

RISK_COLORS = {
    "Critical": "#d32f2f",
    "High":     "#f57c00",
    "Medium":   "#fbc02d",
    "Low":      "#388e3c"
}

VIOLENT_CRIMES = {
    "HOMICIDE", "SEXUAL ASSAULT", "KIDNAPPING",
    "FIREARM OFFENSE", "ARSON", "ASSAULT",
    "ROBBERY", "EXTORTION", "DOMESTIC VIOLENCE"
}


# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df   = pd.read_csv(DATA_PATH, parse_dates=["Date_of_Occurrence"])
    risk = pd.read_csv(RISK_PATH)
    df["Severity"]   = df["Crime_Description"].map(SEVERITY_WEIGHTS).fillna(1)
    df["Is_Violent"] = df["Crime_Description"].apply(
        lambda x: 1 if x in VIOLENT_CRIMES else 0
    )
    df = df.merge(
        risk[["City", "Risk_Index", "Risk_Category"]],
        on="City", how="left"
    )
    return df, risk


# ══════════════════════════════════════════════════════════
# APPLY FILTERS
# ══════════════════════════════════════════════════════════
def apply_filters(df, city, crime, bucket, season,
                  risk_cat, year_range):
    filtered = df.copy()
    if city != "All Cities":
        filtered = filtered[filtered["City"] == city]
    if crime != "All Crime Types":
        filtered = filtered[filtered["Crime_Description"] == crime]
    if bucket != "All Times":
        filtered = filtered[filtered["Time_Bucket"] == bucket]
    if season != "All Seasons":
        filtered = filtered[filtered["Season"] == season]
    if risk_cat != "All Risk Levels":
        filtered = filtered[filtered["Risk_Category"] == risk_cat]
    filtered = filtered[
        (filtered["Year"] >= year_range[0]) &
        (filtered["Year"] <= year_range[1])
    ]
    return filtered


# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
def render_sidebar(df, risk):
    st.sidebar.markdown("""
    <div style='text-align:center; padding:10px 0'>
        <h2 style='color:#ef5350; margin:0'>CriPri</h2>
        <p style='color:#aaa; margin:0; font-size:12px'>
            Crime Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Risk Leaderboard
    st.sidebar.markdown("### 🏙️ Risk Leaderboard")
    top10 = risk.head(10)[["City", "Risk_Index", "Risk_Category"]]
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        color = RISK_COLORS.get(row["Risk_Category"], "gray")
        bar_width = int(row["Risk_Index"])
        st.sidebar.markdown(
            f'<div style="margin:3px 0; font-size:13px">'
            f'<span style="color:{color}">&#9679;</span> '
            f'<b>{i}. {row["City"]}</b>'
            f'<span style="float:right; color:{color}">'
            f'{row["Risk_Index"]:.1f}</span>'
            f'</div>'
            f'<div style="background:#333; border-radius:3px; height:4px; margin:2px 0 6px 0">'
            f'<div style="background:{color}; width:{bar_width}%; height:4px; border-radius:3px"></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters")

    # City filter with search
    cities = ["All Cities"] + sorted(df["City"].unique().tolist())
    selected_city = st.sidebar.selectbox("City", cities)

    # Crime type filter
    crimes = ["All Crime Types"] + sorted(
        df["Crime_Description"].unique().tolist()
    )
    selected_crime = st.sidebar.selectbox("Crime Type", crimes)

    # Time bucket
    buckets = ["All Times", "Morning", "Afternoon", "Evening", "Night"]
    selected_bucket = st.sidebar.selectbox("Time of Day", buckets)

    # Season
    seasons = ["All Seasons", "Summer", "Monsoon", "Post-Monsoon", "Winter"]
    selected_season = st.sidebar.selectbox("Season", seasons)

    # Risk category
    risk_cats = ["All Risk Levels", "Critical", "High", "Medium", "Low"]
    selected_risk = st.sidebar.selectbox("Risk Category", risk_cats)

    # Year range slider
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    st.sidebar.markdown("---")

    # Download button placeholder
    st.sidebar.markdown("### 📥 Export")
    st.sidebar.markdown("*Apply filters then download*")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="text-align:center; color:#666; font-size:11px">'
        'Built with Python & Streamlit<br>'
        '<a href="https://github.com/sarthak29-hub/CriPri" '
        'style="color:#4fc3f7">GitHub Repository</a>'
        '</div>',
        unsafe_allow_html=True
    )

    return (selected_city, selected_crime, selected_bucket,
            selected_season, selected_risk, year_range)


# ══════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════
def render_kpis(filtered, df):
    total    = len(filtered)
    top_city = (
        filtered["City"].value_counts().index[0]
        if total > 0 else "N/A"
    )
    dominant = (
        filtered["Crime_Description"].value_counts().index[0]
        if total > 0 else "N/A"
    )
    avg_risk = (
        filtered["Risk_Index"].mean()
        if total > 0 else 0
    )
    closure = (
        round(
            (filtered["Case_Closed"] == "Yes").sum() / total * 100, 1
        ) if total > 0 else 0
    )
    violent_pct = (
        round(filtered["Is_Violent"].mean() * 100, 1)
        if total > 0 else 0
    )
    night_pct = (
        round(
            (filtered["Time_Bucket"] == "Night").sum() / total * 100, 1
        ) if total > 0 else 0
    )

    dominant_short = dominant[:13] + "..." if len(dominant) > 13 else dominant

    # Delta vs total dataset
    pct_of_total = round(total / len(df) * 100, 1)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Crimes",      f"{total:,}",
              delta=f"{pct_of_total}% of dataset")
    c2.metric("Top Crime City",    top_city)
    c3.metric("Dominant Crime",    dominant_short)
    c4.metric("Avg Risk Index",    f"{avg_risk:.1f}",
              delta="/ 100")
    c5.metric("Case Closure",      f"{closure}%")
    c6.metric("Violent Crime %",   f"{violent_pct}%",
              delta=f"Night: {night_pct}%")


# ══════════════════════════════════════════════════════════
# MAPS TAB
# ══════════════════════════════════════════════════════════
def render_maps_tab(filtered, risk):
    map_type = st.radio(
        "Select Map",
        ["Crime Heatmap", "Risk Zone Map", "Live Filtered Map"],
        horizontal=True
    )

    if map_type == "Crime Heatmap":
        st.caption("Severity-weighted crime density across India")
        if os.path.exists(HEATMAP_PATH):
            with open(HEATMAP_PATH, "r", encoding="utf-8") as f:
                components.html(f.read(), height=580, scrolling=False)
        else:
            st.error("Run hotspot_analysis.py first to generate the map.")

    elif map_type == "Risk Zone Map":
        st.caption("Color coded risk index per city (0-100)")
        if os.path.exists(RISK_MAP_PATH):
            with open(RISK_MAP_PATH, "r", encoding="utf-8") as f:
                components.html(f.read(), height=580, scrolling=False)
        else:
            st.error("Run risk_scoring.py first to generate the map.")

    else:
        st.caption("Updates live based on your sidebar filters")
        _render_live_map(filtered, risk)


def _render_live_map(filtered, risk):
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    if len(filtered) > 0:
        heat_data = filtered[
            ["Latitude", "Longitude", "Severity"]
        ].values.tolist()
        HeatMap(
            heat_data, min_opacity=0.4,
            radius=35, blur=25, max_zoom=6,
            gradient={
                0.2: "blue",   0.4: "lime",
                0.6: "orange", 0.8: "red",
                1.0: "darkred"
            }
        ).add_to(india_map)

    for _, row in risk.iterrows():
        color  = RISK_COLORS.get(row["Risk_Category"], "gray")
        radius = 6 + (row["Risk_Index"] / 100) * 14

        city_f  = filtered[filtered["City"] == row["City"]]
        cnt     = len(city_f)
        dom     = (
            city_f["Crime_Description"].value_counts().index[0]
            if cnt > 0 else "N/A"
        )
        violent = (
            round(city_f["Is_Violent"].mean() * 100, 1)
            if cnt > 0 else 0
        )
        night   = (
            round(
                (city_f["Time_Bucket"] == "Night").sum() / cnt * 100, 1
            ) if cnt > 0 else 0
        )

        popup_html = f"""
        <div style="font-family:Arial; font-size:13px; width:230px">
            <b style="font-size:15px">{row['City']}</b>
            <hr style="margin:4px 0">
            <b style="color:{color}">
                Risk: {row['Risk_Index']:.1f} / 100 — {row['Risk_Category']}
            </b><br>
            Filtered Crimes : <b>{cnt:,}</b><br>
            Dominant Crime  : <b>{dom}</b><br>
            Violent %       : <b>{violent}%</b><br>
            Night Crimes %  : <b>{night}%</b><br>
            Total Severity  : <b>{int(row['Total_Severity'])}</b>
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius, color=color,
            fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['City']} | {row['Risk_Index']:.1f} | {row['Risk_Category']}"
        ).add_to(india_map)

    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:8px;
                border:2px solid #ccc; font-family:Arial; font-size:12px;">
        <b>Risk Index</b><br>
        <span style="color:#d32f2f">&#9679;</span> Critical (75-100)<br>
        <span style="color:#f57c00">&#9679;</span> High (50-75)<br>
        <span style="color:#fbc02d">&#9679;</span> Medium (25-50)<br>
        <span style="color:#388e3c">&#9679;</span> Low (0-25)
    </div>
    """
    india_map.get_root().html.add_child(folium.Element(legend_html))

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.html', delete=False, encoding='utf-8'
    ) as f:
        india_map.save(f.name)
        tmp_path = f.name

    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    components.html(html_content, height=560, scrolling=False)

    try:
        os.unlink(tmp_path)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# RISK ZONES TAB
# ══════════════════════════════════════════════════════════
def render_risk_tab(risk, filtered):
    st.markdown("### Area Risk Index — All Cities Ranked")

    col1, col2 = st.columns([2, 1])

    with col1:
        city_counts = filtered["City"].value_counts().reset_index()
        city_counts.columns = ["City", "Filtered_Crimes"]

        display = risk.merge(city_counts, on="City", how="left")
        display["Filtered_Crimes"] = (
            display["Filtered_Crimes"].fillna(0).astype(int)
        )

        display_cols = [
            "City", "Risk_Index", "Risk_Category",
            "Crime_Count", "Filtered_Crimes", "Total_Severity"
        ]

        def color_risk(val):
            return {
                "Critical": "background-color: #ffcdd2; color: #b71c1c",
                "High":     "background-color: #ffe0b2; color: #e65100",
                "Medium":   "background-color: #fff9c4; color: #f57f17",
                "Low":      "background-color: #c8e6c9; color: #1b5e20"
            }.get(val, "")

        styled = (
            display[display_cols]
            .style
            .map(color_risk, subset=["Risk_Category"])
            .format({"Risk_Index": "{:.2f}"})
            .bar(subset=["Risk_Index"], color="#ef9a9a")
        )
        st.dataframe(styled, width="stretch", height=500)

    with col2:
        st.markdown("**Risk Category Breakdown**")
        cat_counts = risk["Risk_Category"].value_counts()
        colors     = [
            RISK_COLORS.get(c, "gray") for c in cat_counts.index
        ]
        fig, ax = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax.pie(
            cat_counts.values,
            labels=cat_counts.index,
            autopct="%1.0f%%",
            colors=colors,
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2)
        )
        for t in texts:
            t.set_fontsize(11)
        ax.set_title("29 Cities by Risk Level", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Risk Score Distribution**")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.hist(
            risk["Risk_Index"], bins=10,
            color="#ef5350", edgecolor="white", alpha=0.8
        )
        ax2.set_xlabel("Risk Index")
        ax2.set_ylabel("Cities")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ══════════════════════════════════════════════════════════
# ANALYTICS TAB
# ══════════════════════════════════════════════════════════
def render_analytics_tab(filtered):
    if len(filtered) == 0:
        st.warning("No data matches the selected filters.")
        return

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Monthly Crime Trend**")
        monthly = filtered.groupby("Month").size().reset_index(name="Count")
        month_names = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ]
        monthly["Month_Name"] = monthly["Month"].apply(
            lambda x: month_names[x - 1]
        )
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(
            monthly["Month_Name"], monthly["Count"],
            marker="o", color="#ef5350", linewidth=2.5, markersize=6
        )
        ax.fill_between(
            monthly["Month_Name"], monthly["Count"],
            alpha=0.15, color="#ef5350"
        )
        for i, row in monthly.iterrows():
            ax.text(
                i, row["Count"] + 20, str(row["Count"]),
                ha="center", fontsize=7, color="#aaa"
            )
        ax.set_ylabel("Crimes")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        plt.xticks(rotation=45, fontsize=8, color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Crime by Time of Day**")
        bucket_order = ["Morning", "Afternoon", "Evening", "Night"]
        bucket_data  = filtered["Time_Bucket"].value_counts().reindex(
            bucket_order, fill_value=0
        )
        colors = ["#f6d860", "#f0a500", "#c0392b", "#1a237e"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(
            bucket_order, bucket_data.values,
            color=colors, edgecolor="none", width=0.6
        )
        for bar, v in zip(bars, bucket_data.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{v:,}", ha="center", fontsize=9, color="white"
            )
        ax.set_ylabel("Crimes", color="white")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Top 10 Cities by Crime Count**")
        top_cities = filtered["City"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_bar = plt.cm.RdYlGn_r(
            np.linspace(0.1, 0.9, len(top_cities))
        )
        bars = ax.barh(
            top_cities.index, top_cities.values,
            color=colors_bar, edgecolor="none"
        )
        for bar, v in zip(bars, top_cities.values):
            ax.text(
                bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=8, color="white"
            )
        ax.set_xlabel("Crimes", color="white")
        ax.invert_yaxis()
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("**Top 10 Crime Types**")
        top_crimes = filtered["Crime_Description"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_bar = plt.cm.Oranges(
            np.linspace(0.4, 0.9, len(top_crimes))
        )
        bars = ax.barh(
            top_crimes.index, top_crimes.values,
            color=colors_bar, edgecolor="none"
        )
        for bar, v in zip(bars, top_crimes.values):
            ax.text(
                bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=8, color="white"
            )
        ax.set_xlabel("Count", color="white")
        ax.invert_yaxis()
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 3 - Year trend
    st.markdown("**Year-wise Crime Trend**")
    year_counts = filtered["Year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(
        year_counts.index.astype(str), year_counts.values,
        color="#4fc3f7", edgecolor="none", width=0.5
    )
    for i, v in enumerate(year_counts.values):
        ax.text(i, v + 20, f"{v:,}", ha="center", fontsize=9, color="white")
    ax.set_ylabel("Crimes", color="white")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════
# CITY COMPARISON TAB
# ══════════════════════════════════════════════════════════
def render_comparison_tab(df, risk):
    st.markdown("### Compare Two Cities Side by Side")

    cities = sorted(df["City"].unique().tolist())
    col1, col2 = st.columns(2)

    with col1:
        city_a = st.selectbox("City A", cities, index=0)
    with col2:
        city_b = st.selectbox("City B", cities, index=1)

    if city_a == city_b:
        st.warning("Please select two different cities.")
        return

    df_a = df[df["City"] == city_a]
    df_b = df[df["City"] == city_b]
    risk_a = risk[risk["City"] == city_a].iloc[0]
    risk_b = risk[risk["City"] == city_b].iloc[0]

    # Metric comparison
    st.markdown("#### Key Metrics Comparison")
    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Total Crimes",
        f"{len(df_a):,}",
        delta=f"{city_a}",
    )
    m1.metric(
        "",
        f"{len(df_b):,}",
        delta=f"{city_b}",
    )
    m2.metric(f"{city_a} Risk", f"{risk_a['Risk_Index']:.1f}")
    m2.metric(f"{city_b} Risk", f"{risk_b['Risk_Index']:.1f}")
    m3.metric(
        f"{city_a} Closure",
        f"{round((df_a['Case_Closed']=='Yes').mean()*100,1)}%"
    )
    m3.metric(
        f"{city_b} Closure",
        f"{round((df_b['Case_Closed']=='Yes').mean()*100,1)}%"
    )
    m4.metric(
        f"{city_a} Violent%",
        f"{round(df_a['Is_Violent'].mean()*100,1)}%"
    )
    m4.metric(
        f"{city_b} Violent%",
        f"{round(df_b['Is_Violent'].mean()*100,1)}%"
    )

    # Crime type comparison chart
    st.markdown("#### Crime Type Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("#0e1117")

    for ax, city_df, city_name, color in [
        (axes[0], df_a, city_a, "#ef5350"),
        (axes[1], df_b, city_b, "#4fc3f7")
    ]:
        top = city_df["Crime_Description"].value_counts().head(10)
        ax.barh(top.index, top.values, color=color, edgecolor="none")
        ax.set_title(city_name, color="white", fontsize=12, fontweight="bold")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.set_xlabel("Count", color="white")
        ax.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Time bucket comparison
    st.markdown("#### Time of Day Comparison")
    bucket_order = ["Morning", "Afternoon", "Evening", "Night"]
    counts_a = df_a["Time_Bucket"].value_counts().reindex(
        bucket_order, fill_value=0
    )
    counts_b = df_b["Time_Bucket"].value_counts().reindex(
        bucket_order, fill_value=0
    )

    x     = np.arange(len(bucket_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.bar(x - width/2, counts_a.values, width,
           label=city_a, color="#ef5350", edgecolor="none")
    ax.bar(x + width/2, counts_b.values, width,
           label=city_b, color="#4fc3f7", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_order, color="white")
    ax.set_ylabel("Crimes", color="white")
    ax.legend(facecolor="#1e2130", labelcolor="white")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════
# VICTIM & WEAPON TAB
# ══════════════════════════════════════════════════════════
def render_victim_tab(filtered):
    if len(filtered) == 0:
        st.warning("No data matches the selected filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Victim Age Group**")
        age_counts = filtered["Victim_Age_Group"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0e1117")
        wedges, texts, autotexts = ax.pie(
            age_counts.values,
            labels=age_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2196F3","#4CAF50","#FF9800","#9C27B0"],
            wedgeprops=dict(edgecolor='#0e1117', linewidth=2)
        )
        for t in texts + autotexts:
            t.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Weapon Group Distribution**")
        weapon_counts = filtered["Weapon_Group"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = plt.cm.Blues(
            np.linspace(0.4, 0.9, len(weapon_counts))
        )
        ax.barh(
            weapon_counts.index, weapon_counts.values,
            color=colors, edgecolor="none"
        )
        for i, v in enumerate(weapon_counts.values):
            ax.text(v + 5, i, f"{v:,}", va="center",
                    fontsize=9, color="white")
        ax.set_xlabel("Count", color="white")
        ax.invert_yaxis()
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Victim Gender**")
        gender_counts = filtered["Victim_Gender"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.bar(
            gender_counts.index, gender_counts.values,
            color=["#2196F3","#E91E63","#9C27B0"],
            edgecolor="none", width=0.5
        )
        for i, v in enumerate(gender_counts.values):
            ax.text(i, v + 30, f"{v:,}", ha="center",
                    fontsize=9, color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("**Day of Week**")
        day_order = [
            "Monday","Tuesday","Wednesday",
            "Thursday","Friday","Saturday","Sunday"
        ]
        day_counts = filtered["Day_Name"].value_counts().reindex(
            day_order, fill_value=0
        )
        colors = [
            "#c0392b" if d in ["Saturday","Sunday"]
            else "#4fc3f7" for d in day_order
        ]
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.bar(day_order, day_counts.values,
               color=colors, edgecolor="none", width=0.6)
        ax.set_ylabel("Count", color="white")
        ax.annotate(
            "Red = Weekend",
            xy=(0.7, 0.92), xycoords="axes fraction",
            fontsize=8, color="#c0392b"
        )
        plt.xticks(rotation=30, fontsize=7, color="white")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════
# SEASON & TIME TAB
# ══════════════════════════════════════════════════════════
def render_season_tab(filtered):
    if len(filtered) == 0:
        st.warning("No data matches the selected filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Crime by Season**")
        season_order  = ["Summer","Monsoon","Post-Monsoon","Winter"]
        season_colors = {
            "Summer": "#e74c3c", "Monsoon": "#2980b9",
            "Post-Monsoon": "#27ae60", "Winter": "#8e44ad"
        }
        season_counts = filtered["Season"].value_counts().reindex(
            season_order, fill_value=0
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0e1117")
        bars = ax.bar(
            season_order, season_counts.values,
            color=[season_colors[s] for s in season_order],
            edgecolor="none", width=0.6
        )
        for bar, v in zip(bars, season_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20, f"{v:,}",
                ha="center", fontsize=9, color="white"
            )
        ax.set_ylabel("Crimes", color="white")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Hourly Crime Distribution**")
        hour_counts = filtered["Hour"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0e1117")
        ax.fill_between(
            hour_counts.index, hour_counts.values,
            alpha=0.4, color="#ef5350"
        )
        ax.plot(
            hour_counts.index, hour_counts.values,
            color="#ef5350", linewidth=2
        )
        ax.set_xlabel("Hour of Day", color="white")
        ax.set_ylabel("Crimes", color="white")
        ax.set_xticks(range(0, 24, 2))
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Heatmap - hour vs day
    st.markdown("**Crime Intensity: Hour vs Day of Week**")
    day_order = [
        "Monday","Tuesday","Wednesday",
        "Thursday","Friday","Saturday","Sunday"
    ]
    pivot = filtered.groupby(
        ["Day_Name", "Hour"]
    ).size().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#0e1117")
    im = ax.imshow(
        pivot.values, aspect="auto",
        cmap="RdYlGn_r", interpolation="nearest"
    )
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order, color="white", fontsize=9)
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels(
        [f"{h}:00" for h in range(24)],
        rotation=45, fontsize=7, color="white"
    )
    ax.set_xlabel("Hour of Day", color="white")
    plt.colorbar(im, ax=ax, label="Crime Count")
    ax.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════
# DOWNLOAD SECTION
# ══════════════════════════════════════════════════════════
def render_download(filtered):
    st.markdown("### Download Filtered Data")

    col1, col2 = st.columns(2)

    with col1:
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="cripri_filtered_data.csv",
            mime="text/csv"
        )

    with col2:
        st.info(
            f"Current filter returns **{len(filtered):,}** records. "
            f"CSV will contain all {len(filtered.columns)} columns."
        )


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <style>
    .stMetric {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 14px;
        border: 1px solid #2d3250;
    }
    [data-testid="stSidebar"] {
        background-color: #12141f;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 6px;
        padding: 8px 16px;
        color: #aaa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ef5350 !important;
        color: white !important;
    }
    h3 { color: #4fc3f7; }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style='display:flex; align-items:center; gap:16px; margin-bottom:8px'>
        <div>
            <h1 style='color:#ef5350; margin:0; font-size:2rem'>
                CriPri — Crime Intelligence Dashboard
            </h1>
            <p style='color:#666; margin:0; font-size:13px'>
                India Crime Analytics &nbsp;|&nbsp;
                40,160 records &nbsp;|&nbsp;
                29 cities &nbsp;|&nbsp;
                2020 - 2024 &nbsp;|&nbsp;
                Powered by ML & GIS
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Load
    df, risk = load_data()

    # Sidebar
    (city, crime, bucket,
     season, risk_cat,
     year_range) = render_sidebar(df, risk)

    # Filter
    filtered = apply_filters(
        df, city, crime, bucket, season, risk_cat, year_range
    )

    # Filter summary
    st.caption(
        f"Showing **{len(filtered):,}** of **{len(df):,}** records  |  "
        f"City: {city}  |  Crime: {crime}  |  "
        f"Time: {bucket}  |  Season: {season}  |  "
        f"Risk: {risk_cat}  |  "
        f"Years: {year_range[0]}–{year_range[1]}"
    )

    # KPIs
    render_kpis(filtered, df)
    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Maps",
        "Risk Zones",
        "Analytics",
        "City Compare",
        "Victim & Weapon",
        "Season & Time",
        "Export Data"
    ])

    with tab1:
        render_maps_tab(filtered, risk)
    with tab2:
        render_risk_tab(risk, filtered)
    with tab3:
        render_analytics_tab(filtered)
    with tab4:
        render_comparison_tab(df, risk)
    with tab5:
        render_victim_tab(filtered)
    with tab6:
        render_season_tab(filtered)
    with tab7:
        render_download(filtered)


if __name__ == "__main__":
    main()