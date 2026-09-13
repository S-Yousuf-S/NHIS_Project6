"""
================================================================================
 SALESFC_DASHBOARD.PY

 Project      : Sales Forecasting Across Multiple Retail Stores
 Author       : Yousuf S. R. Sakkaf
 Description  : Streamlit dashboard serving the Tuned Random Forest Sales and
                Customers models. Manual per-store prediction (sliders/dropdowns
                for top drivers, auto-populated from store.csv) and bulk CSV
                upload, with a predicted sales+customers plot and CSV download.
 Run          : streamlit run salesfc_dashboard.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
import gdown
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titlecolor": "midnightblue", "font.family": "DejaVu Sans"})

PRIMARY_COLOR = "indigo"
SECONDARY_COLOR = "thistle"
SUCCESS_COLOR = "#2E8B57"
TITLE_COLOR = "midnightblue"

ASSORTMENT_LABELS = {"a": "Basic", "b": "Extra", "c": "Extended"}
STATE_HOLIDAY_LABELS = {"0": "None", "a": "Public Holiday", "b": "Easter Holiday", "c": "Christmas"}
SCHOOL_HOLIDAY_LABELS = {0: "No", 1: "Yes"}
STORE_TYPE_LABELS = {"a": "Store Type A", "b": "Store Type B", "c": "Store Type C", "d": "Store Type D"}
# Note: the brief never defines semantic meaning for StoreType (only "4 different store models") —
# so these are friendlier display names, not real categories, unlike Assortment/StateHoliday above.

st.set_page_config(page_title="Rossmann Sales Forecaster", page_icon="deploy_assets/forecast_logo.png", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { font-size: 18px; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
label { font-size: 1.05rem !important; }
h1 { font-size: 2.3rem !important; }
[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Loading — latest serialized artifacts + raw store attributes
# ------------------------------------------------------------------------------

SALES_MODEL_DRIVE_ID = "1nRubYM7hRuwU25EGlVpuQT2Varnbgzwr"
CUSTOMERS_MODEL_DRIVE_ID = "1jB3IeUd9_kMYThx-I9Utne81-OoDJcH3"

@st.cache_resource
def load_model_from_drive(drive_id, local_filename):
    local_path = os.path.join("deploy_assets", local_filename)
    if not os.path.exists(local_path):
        os.makedirs("deploy_assets", exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={drive_id}", local_path, quiet=False)
    return joblib.load(local_path)

@st.cache_data
def load_store_reference():
    return pd.read_csv("deploy_assets/store.csv")

sales_pipeline = load_model_from_drive(SALES_MODEL_DRIVE_ID, "final_tuned_rf_pipeline.pkl")
customers_pipeline = load_model_from_drive(CUSTOMERS_MODEL_DRIVE_ID, "final_tuned_rf_customers_pipeline.pkl")
store_ref = load_store_reference()

# ------------------------------------------------------------------------------
# Header
# ------------------------------------------------------------------------------
import base64

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = get_base64_image("deploy_assets/forecast_logo.png")

st.markdown(f"""
<div style="background:{PRIMARY_COLOR};color:white;padding:20px;border-radius:8px;text-align:center;margin-bottom:20px;
            display:flex;align-items:center;justify-content:center;gap:16px;">
    <img src="data:image/png;base64,{logo_b64}" width="56" height="56">
    <div>
        <h1 style="margin:0;">Rossmann Sales Forecaster</h1>
        <p style="margin:4px 0 0 0;font-style:italic;">Predicted Sales & Customer Volume — Tuned Random Forest</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div style="background:{SECONDARY_COLOR};border-left:6px solid {PRIMARY_COLOR};padding:12px;border-radius:6px;">
    <b>About this tool</b><br><br>
    Rossmann Pharmaceuticals runs stores across several cities. This tool forecasts
    daily sales and customer numbers up to six weeks ahead, so store managers can
    plan staffing and inventory without relying on guesswork.<br><br>
    Predictions come from a Random Forest model trained on three years of historical
    sales, promotions, holidays, and store characteristics.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#F1F3F6;border-left:6px solid #1F77B4;padding:12px;border-radius:6px;margin-top:12px;">
    <b>Model Details</b><br><br>
    <b>Sales:</b> Tuned Random Forest<br>
    R² = 0.9621 · RMSPE = 8.05%<br><br>
    <b>Customers:</b> Tuned Random Forest (same architecture)<br>
    R² = 0.9467 · RMSPE = 11.95%<br><br>
    n_estimators=100, max_depth=30,<br>min_samples_split=10, min_samples_leaf=2
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------
#  Reusable KPI card component
# -----------------------------------------------
def kpi_card(label, value, color="#2E8B57"):
    st.markdown(f"""
    <div style="background:#EAF7EA;border-left:6px solid {color};padding:14px 18px;border-radius:8px;text-align:center;">
        <div style="color:#555;font-size:14px;font-weight:600;">{label}</div>
        <div style="color:{color};font-size:28px;font-weight:bold;margin-top:4px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Feature engineering helpers — shared by both modes
# ------------------------------------------------------------------------------
def derive_calendar_features(d: date):
    return {
        "Year": d.year, "Month": d.month, "Day": d.day,
        "WeekOfYear": d.isocalendar()[1], "Quarter": (d.month - 1) // 3 + 1,
        "DayOfWeek": d.isoweekday(), "IsWeekend": int(d.isoweekday() in (6, 7)),
        "MonthPosition": "Start" if d.day <= 10 else ("Mid" if d.day <= 20 else "End"),
    }

def derive_promo2_month(d: date, promo_interval: str, promo2: int):
    if promo2 == 0 or pd.isna(promo_interval):
        return 0
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return int(month_names[d.month - 1] in str(promo_interval).split(","))

def derive_duration_fields(d: date, row):
    if pd.notna(row.get("Promo2SinceYear")) and row.get("Promo2", 0) == 1:
        promo2_weeks = max(0, (d.year - int(row["Promo2SinceYear"])) * 52 + (d.isocalendar()[1] - int(row["Promo2SinceWeek"])))
    else:
        promo2_weeks = 0
    if pd.notna(row.get("CompetitionOpenSinceYear")):
        comp_months = max(0, (d.year - int(row["CompetitionOpenSinceYear"])) * 12 + (d.month - int(row["CompetitionOpenSinceMonth"])))
        comp_known = 1
    else:
        comp_months, comp_known = 0, 0
    return promo2_weeks, comp_months, comp_known

def build_feature_row(store_row, d, promo, state_holiday, school_holiday,
                       assortment, store_type, competition_distance,
                       days_to_holiday=0, days_since_holiday=0):
    cal = derive_calendar_features(d)
    promo2_weeks, comp_months, comp_known = derive_duration_fields(d, store_row)
    is_promo2_month = derive_promo2_month(d, store_row.get("PromoInterval"), store_row.get("Promo2", 0))
    return pd.DataFrame([{
        "Store": store_row["Store"], "DayOfWeek": cal["DayOfWeek"], "Promo": promo,
        "StateHoliday": state_holiday, "SchoolHoliday": school_holiday,
        "StoreType": store_type, "Assortment": assortment,
        "CompetitionDistance": competition_distance, "Promo2": store_row.get("Promo2", 0),
        "Promo2_Duration_Weeks": promo2_weeks, "CompetitionOpen_Duration_Months": comp_months,
        "CompetitionOpenDate_Known": comp_known, "Year": cal["Year"], "Month": cal["Month"],
        "Day": cal["Day"], "WeekOfYear": cal["WeekOfYear"], "Quarter": cal["Quarter"],
        "IsWeekend": cal["IsWeekend"], "MonthPosition": cal["MonthPosition"],
        "DaysToNextHoliday": days_to_holiday, "DaysSinceLastHoliday": days_since_holiday,
        "IsPromo2Month": is_promo2_month,
    }])

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🎯 Single Store Prediction", "📂 Bulk CSV Upload"])

with tab1:
    col_input, col_result = st.columns([1, 1.4])

    with col_input:
        store_id = st.selectbox("Store", sorted(store_ref["Store"].unique()))
        store_row = store_ref[store_ref["Store"] == store_id].iloc[0]

        forecast_date = st.date_input("Forecast date", value=date.today())

        st.markdown("**Top drivers** — pre-filled from this store, adjustable for what-if scenarios")
        promo = st.selectbox("Promo running today?", ["No", "Yes"])
        promo = 1 if promo == "Yes" else 0
        assortment = st.selectbox("Assortment", options=list(ASSORTMENT_LABELS.keys()),
                                   format_func=lambda k: ASSORTMENT_LABELS[k],
                                   index=list(ASSORTMENT_LABELS.keys()).index(store_row["Assortment"]))
        store_type = st.selectbox("Store Type", options=list(STORE_TYPE_LABELS.keys()),
                                   format_func=lambda k: STORE_TYPE_LABELS[k],
                                   index=list(STORE_TYPE_LABELS.keys()).index(store_row["StoreType"]))
        competition_distance = st.slider("Competition Distance (m)", 0, 20000,
                                          int(store_row["CompetitionDistance"]) if pd.notna(store_row["CompetitionDistance"]) else 5000)

        with st.expander("Other date-dependent fields"):
            state_holiday = st.selectbox("State Holiday", options=list(STATE_HOLIDAY_LABELS.keys()),
                                          format_func=lambda k: STATE_HOLIDAY_LABELS[k])
            school_holiday = st.selectbox("School Holiday", options=[0, 1],
                                           format_func=lambda k: SCHOOL_HOLIDAY_LABELS[k])
            days_to_holiday = st.slider("Days to next holiday", 0, 60, 14)
            days_since_holiday = st.slider("Days since last holiday", 0, 60, 14)

        predict_clicked = st.button("Generate Forecast", type="primary")

    with col_result:
        if predict_clicked:
            X_row = build_feature_row(store_row, forecast_date, promo, state_holiday, school_holiday,
                                       assortment, store_type, competition_distance,
                                       days_to_holiday, days_since_holiday)
            pred_sales = sales_pipeline.predict(X_row)[0]
            pred_customers = customers_pipeline.predict(X_row)[0]

            k1, k2 = st.columns(2)
            with k1: kpi_card("Predicted Sales", f"€{pred_sales:,.0f}", "#2E8B57")
            with k2: kpi_card("Predicted Customers", f"{pred_customers:,.0f}", "#1F77B4")

            fig, ax1 = plt.subplots(figsize=(6, 5))
            ax2 = ax1.twinx()

            ax1.bar(0, pred_sales, width=0.35, color="forestgreen", alpha=0.85)
            ax2.bar(1, pred_customers, width=0.35, color="cornflowerblue", alpha=0.85)

            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(["Sales (€)", "Customers"], fontsize=11)
            ax1.set_ylabel("Predicted Sales (€)", color="forestgreen", fontsize=11)
            ax2.set_ylabel("Predicted Customers", color="cornflowerblue", fontsize=11)
            ax1.set_ylim(0, pred_sales * 1.3)
            ax2.set_ylim(0, pred_customers * 1.3)
            ax1.tick_params(axis="y", labelcolor="forestgreen", labelsize=9)
            ax2.tick_params(axis="y", labelcolor="cornflowerblue", labelsize=9)
            ax1.tick_params(axis="x", labelsize=10)
            ax1.grid(axis="y", linestyle="--", alpha=0.25)
            ax1.grid(False, axis="x")
            ax2.grid(False)
            ax1.set_title(f"Store {store_id} — {forecast_date}", color=TITLE_COLOR, fontweight="bold", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Set your inputs and click **Generate Forecast**.")

with tab2:
    with st.expander("📋 CSV format guide — click to expand", expanded=True):
        st.markdown("""
        Your CSV needs these columns:

        | Column | Example | Notes |
        |---|---|---|
        | `Store` | `38` | Must match a Store ID in the trained data |
        | `Date` | `2026-09-20` | YYYY-MM-DD format |
        | `Promo` | `1` | 1 = promo running, 0 = no promo |
        | `StateHoliday` | `0` | `0`=None, `a`=Public, `b`=Easter, `c`=Christmas |
        | `SchoolHoliday` | `0` | 1 = affected by school closure, 0 = not |

        Store-specific details (type, assortment, competition distance) are pulled in automatically — you don't need to include them.
        """)
        def build_demo_template():
            demo_stores = [1, 17, 38]
            date_range = pd.date_range("2026-09-20", periods=14, freq="D")
            rows = []
            for s in demo_stores:
                for d in date_range:
                    is_promo_day = d.weekday() in [0, 2, 4]
                    is_holiday = d == pd.Timestamp("2026-09-27")
                    rows.append({
                        "Store": s, "Date": d.strftime("%Y-%m-%d"),
                        "Promo": int(is_promo_day),
                        "StateHoliday": "a" if is_holiday else "0",
                        "SchoolHoliday": 0
                    })
            return pd.DataFrame(rows)

        template_df = build_demo_template()
        st.download_button("📥 Download a template CSV", template_df.to_csv(index=False), "template.csv")

    uploaded = st.file_uploader("Upload your CSV", type="csv")
    view_mode = st.radio("View", ["Aggregate", "Per Store"], horizontal=True)

    if uploaded is not None:
        upload_df = pd.read_csv(uploaded, parse_dates=["Date"], dtype={"StateHoliday": str})
        merged = upload_df.merge(store_ref, on="Store", how="left")

        rows = []
        for _, r in merged.iterrows():
            d = r["Date"].date()
            holidays = merged.loc[merged["Store"] == r["Store"], "StateHoliday"]
            holiday_dates = merged.loc[(merged["Store"] == r["Store"]) & (merged["StateHoliday"] != "0"), "Date"]
            days_to = (holiday_dates[holiday_dates >= r["Date"]].min() - r["Date"]).days if not holiday_dates[holiday_dates >= r["Date"]].empty else 0
            days_since = (r["Date"] - holiday_dates[holiday_dates <= r["Date"]].max()).days if not holiday_dates[holiday_dates <= r["Date"]].empty else 0
            rows.append(build_feature_row(r, d, r["Promo"], r["StateHoliday"], r["SchoolHoliday"],
                                           r["Assortment"], r["StoreType"], r["CompetitionDistance"],
                                           max(days_to, 0), max(days_since, 0)))
        X_bulk = pd.concat(rows, ignore_index=True)

        results = upload_df[["Store", "Date"]].copy()
        results["Predicted_Sales"] = sales_pipeline.predict(X_bulk)
        results["Predicted_Customers"] = customers_pipeline.predict(X_bulk)

        if view_mode == "Aggregate":
            plot_data = results.groupby("Date")[["Predicted_Sales", "Predicted_Customers"]].sum()
        else:
            chosen_store = st.selectbox("Store", results["Store"].unique())
            plot_data = results[results["Store"] == chosen_store].set_index("Date")[["Predicted_Sales", "Predicted_Customers"]]

        k1, k2, k3, k4 = st.columns(4)
        with k1: k1.metric("Total Predicted Sales", f"€{results['Predicted_Sales'].sum():,.0f}")
        with k2: k2.metric("Total Predicted Customers", f"{results['Predicted_Customers'].sum():,.0f}")
        with k3: k3.metric("Stores Covered", f"{results['Store'].nunique()}")
        with k4: k4.metric("Date Range", f"{results['Date'].min().date()} → {results['Date'].max().date()}")
        

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(plot_data.index, plot_data["Predicted_Sales"], color="forestgreen",
                  marker="o", markersize=6, linewidth=2, linestyle="-", label="Sales")
        ax1.set_ylabel("Predicted Sales (€)", color="forestgreen", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="forestgreen")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        ax2 = ax1.twinx()
        ax2.plot(plot_data.index, plot_data["Predicted_Customers"], color="cornflowerblue",
                  marker="s", markersize=6, linewidth=2, linestyle="--", label="Customers")
        ax2.set_ylabel("Predicted Customers", color="cornflowerblue", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="cornflowerblue")
        ax2.grid(False)

        date_span = (plot_data.index.max() - plot_data.index.min())
        pad = pd.Timedelta(days=1) if date_span < pd.Timedelta(days=2) else date_span * 0.1
        ax1.set_xlim(plot_data.index.min() - pad, plot_data.index.max() + pad)
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=8))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=30)

        ax1.set_title("Sales & Customer Forecast Over Time", color=TITLE_COLOR, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(results)
        st.download_button("Download predictions as CSV", results.to_csv(index=False), "predictions.csv")