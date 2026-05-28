import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from xgboost import XGBClassifier

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "crime_cleaned.csv")
RISK_PATH  = os.path.join(BASE_DIR, "data", "city_risk_scores.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def load_data():
    df   = pd.read_csv(DATA_PATH, parse_dates=["Date_of_Occurrence"])
    risk = pd.read_csv(RISK_PATH)[["City", "Risk_Index", "Risk_Category"]]
    df   = df.merge(risk, on="City", how="left")
    print(f"Data loaded: {df.shape[0]} rows")
    return df
def create_target(df):
    df["Week"] = df["Date_of_Occurrence"].dt.isocalendar().week.astype(int)
    df["Year"] = df["Date_of_Occurrence"].dt.year

    agg = df.groupby(["City", "Year", "Week"]).agg(
        Crime_Count   = ("Crime_Description","count"),
        Avg_Severity  = ("Severity","mean") if "Severity" in df.columns else ("Crime_Description", "count"),
        Avg_Hour      = ("Hour","mean"),
        Month         = ("Month","first"),
        Is_Weekend    = ("Is_Weekend","max"),
        Risk_Index    = ("Risk_Index","first"),
    ).reset_index()

    agg["Target"] = 1

    all_cities = df["City"].unique()
    all_years  = df["Year"].unique()
    all_weeks  = range(1, 53)

    existing = set(zip(agg["City"], agg["Year"], agg["Week"]))
    negatives = []

    for city in all_cities:
        risk_val = df[df["City"] == city]["Risk_Index"].iloc[0]
        for year in all_years:
            for week in all_weeks:
                if (city, year, week) not in existing:
                    negatives.append({
                        "City":city,
                        "Year":year,
                        "Week":week,
                        "Crime_Count": 0,
                        "Avg_Severity":0,
                        "Avg_Hour":12,
                        "Month":(week // 4) + 1,
                        "Is_Weekend":0,
                        "Risk_Index":risk_val,
                        "Target":0
                    })

    neg_df = pd.DataFrame(negatives)

    combined = pd.concat([agg, neg_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    pos = combined["Target"].sum()
    neg = (combined["Target"] == 0).sum()
    print(f"Positive samples (crime occurred): {pos}")
    print(f"Negative samples (no crime): {neg}")
    print(f"Total samples: {len(combined)}")

    return combined


def engineer_features(df):
    le = LabelEncoder()
    df["City_Encoded"] = le.fit_transform(df["City"])
    features = ["City_Encoded","Month","Week","Is_Weekend","Avg_Hour","Avg_Severity","Risk_Index","Crime_Count",]

    X = df[features].fillna(0)
    y = df["Target"]

    print(f"Features used: {features}")
    print(f"X shape: {X.shape}")
    print(f"y distribution: {y.value_counts().to_dict()}")
    return X, y, features

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("   Random Forest trained")
    return rf

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    print("XGBoost trained")
    return xgb

def evaluate_model(model, X_test, y_test, model_name):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    return {
        "Model":     model_name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
        "ROC_AUC":   round(auc,  4),
        "y_pred":    y_pred,
        "y_proba":   y_proba
    }

def generate_ml_charts(rf, xgb, rf_results, xgb_results,
                        X_test, y_test, features):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("ML Model Evaluation", fontsize=15, fontweight="bold")

    cm_rf = confusion_matrix(y_test, rf_results["y_pred"])
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
                ax=axes[0, 0], xticklabels=["No Crime", "Crime"],
                yticklabels=["No Crime", "Crime"])
    axes[0, 0].set_title("Random Forest - Confusion Matrix")
    axes[0, 0].set_ylabel("Actual")
    axes[0, 0].set_xlabel("Predicted")

    cm_xgb = confusion_matrix(y_test, xgb_results["y_pred"])
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Oranges",
                ax=axes[0, 1], xticklabels=["No Crime", "Crime"],
                yticklabels=["No Crime", "Crime"])
    axes[0, 1].set_title("XGBoost - Confusion Matrix")
    axes[0, 1].set_ylabel("Actual")
    axes[0, 1].set_xlabel("Predicted")

    metrics      = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    rf_scores    = [rf_results[m]  for m in metrics]
    xgb_scores   = [xgb_results[m] for m in metrics]
    x            = np.arange(len(metrics))
    width        = 0.35

    axes[1, 0].bar(x - width/2, rf_scores,  width, label="Random Forest", color="steelblue")
    axes[1, 0].bar(x + width/2, xgb_scores, width, label="XGBoost",       color="darkorange")
    axes[1, 0].set_title("Model Comparison - All Metrics")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    for i, (rv, xv) in enumerate(zip(rf_scores, xgb_scores)):
        axes[1, 0].text(i - width/2, rv + 0.01, f"{rv:.3f}", ha="center", fontsize=7)
        axes[1, 0].text(i + width/2, xv + 0.01, f"{xv:.3f}", ha="center", fontsize=7)

    importances = xgb.feature_importances_
    feat_imp    = pd.Series(importances, index=features).sort_values(ascending=True)
    feat_imp.plot(kind="barh", ax=axes[1, 1], color="darkorange")
    axes[1, 1].set_title("XGBoost - Feature Importance")
    axes[1, 1].set_xlabel("Importance Score")
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "ml_evaluation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nML evaluation chart saved to: {path}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test, rf_results["y_proba"],
        name="Random Forest", ax=ax2, color="steelblue"
    )
    RocCurveDisplay.from_predictions(
        y_test, xgb_results["y_proba"],
        name="XGBoost", ax=ax2, color="darkorange"
    )
    ax2.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    roc_path = os.path.join(CHARTS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to: {roc_path}")

def compare_models(rf_results, xgb_results):
    print("MODEL COMPARISON SUMMARY")
    print(f"{'Metric':<15} {'Random Forest':>15} {'XGBoost':>15}")
    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
        rf_val  = rf_results[metric]
        xgb_val = xgb_results[metric]
        better  = "<-- Better" if xgb_val > rf_val else ""
        print(f"{metric:<15} {rf_val:>15.4f} {xgb_val:>15.4f}  {better}")

    best = "XGBoost" if xgb_results["ROC_AUC"] > rf_results["ROC_AUC"] else "Random Forest"
    print(f"\nBest model by ROC-AUC: {best}")

if __name__ == "__main__":
    print("PHASE 4 - ML PREDICTION MODEL")
    df= load_data()
    df_agg= create_target(df)
    X, y, features= engineer_features(df_agg)
    X_train, X_test, y_train, y_test= split_data(X, y)

    rf= train_random_forest(X_train, y_train)
    xgb= train_xgboost(X_train, y_train)

    rf_results= evaluate_model(rf,  X_test, y_test, "Random Forest")
    xgb_results= evaluate_model(xgb, X_test, y_test, "XGBoost")

    compare_models(rf_results, xgb_results)
    generate_ml_charts(rf, xgb, rf_results, xgb_results,
                       X_test, y_test, features)

    print("PHASE 4 COMPLETE")
    print("\nOutputs generated:")
    print("outputs/charts/ml_evaluation.png")
    print("outputs/charts/roc_curve.png")