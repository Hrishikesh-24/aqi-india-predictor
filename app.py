from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


DEFAULT_DATA_PATH = r"C:\Users\HP\AQi-Project\INDIA_AQI_COMPLETE_20251126.csv"

NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Month",
    "Day",
    "Hour",
    "Is_Weekend",
    "Temp_2m_C",
    "Humidity_Percent",
    "Wind_Speed_10m_kmh",
    "Precipitation_mm",
    "Pressure_MSL_hPa",
    "Cloud_Cover_Percent",
    "PM2_5_ugm3",
    "PM10_ugm3",
    "CO_ugm3",
    "NO2_ugm3",
    "SO2_ugm3",
    "O3_ugm3",
    "AOD",
]

CATEGORICAL_FEATURES = [
    "City",
    "State",
    "Season",
    "Time_of_Day",
    "Humidity_Category",
    "Wind_Category",
]

USE_COLUMNS = [
    "City",
    "State",
    "Latitude",
    "Longitude",
    "Datetime",
    "Month",
    "Day",
    "Hour",
    "Is_Weekend",
    "Season",
    "Time_of_Day",
    "Temp_2m_C",
    "Humidity_Percent",
    "Humidity_Category",
    "Wind_Speed_10m_kmh",
    "Wind_Category",
    "Precipitation_mm",
    "Pressure_MSL_hPa",
    "Cloud_Cover_Percent",
    "PM2_5_ugm3",
    "PM10_ugm3",
    "CO_ugm3",
    "NO2_ugm3",
    "SO2_ugm3",
    "O3_ugm3",
    "AOD",
    "US_AQI",
]

AQI_COLORS = {
    "Good": [35, 145, 72, 210],
    "Satisfactory": [115, 190, 70, 210],
    "Moderate": [245, 190, 65, 220],
    "Poor": [235, 120, 45, 225],
    "Very Poor": [205, 65, 65, 230],
    "Severe": [120, 45, 95, 235],
}


@dataclass
class ModelBundle:
    regression_models: dict[str, Pipeline]
    regression_metrics: pd.DataFrame
    classification_models: dict[str, Pipeline]
    classification_metrics: pd.DataFrame
    clustering_model: Pipeline
    cluster_summary: pd.DataFrame


st.set_page_config(
    page_title="India AQI Predictor",
    page_icon="AQI",
    layout="wide",
)


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def aqi_category(value: float) -> str:
    if pd.isna(value):
        return np.nan
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


def compact_category(value: float) -> str:
    if pd.isna(value):
        return np.nan
    if value <= 50:
        return "Good"
    if value <= 200:
        return "Moderate"
    return "Poor"


def load_data(path: str, max_rows: int | None) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        st.error(f"Dataset not found: {path}")
        st.stop()

    df = pd.read_csv(dataset_path, usecols=USE_COLUMNS, nrows=max_rows)
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["US_AQI", "Latitude", "Longitude", "City", "State"])
    df["AQI_Level"] = df["US_AQI"].apply(aqi_category)
    df["Pollution_Level"] = df["US_AQI"].apply(compact_category)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


@st.cache_data(show_spinner=False)
def cached_load_data(path: str, max_rows: int | None) -> pd.DataFrame:
    return load_data(path, max_rows)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("category", make_ohe(), CATEGORICAL_FEATURES),
        ]
    )


@st.cache_resource(show_spinner=False)
def train_models(path: str, max_rows: int | None, train_sample: int, category_mode: str) -> ModelBundle:
    df = load_data(path, max_rows)
    if len(df) > train_sample:
        df = df.sample(train_sample, random_state=42)

    target_class = "Pollution_Level" if category_mode == "Simple: Good / Moderate / Poor" else "AQI_Level"
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_reg = df["US_AQI"]
    y_cls = df[target_class]

    x_train, x_test, y_reg_train, y_reg_test = train_test_split(
        x, y_reg, test_size=0.2, random_state=42
    )
    x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(
        x, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    regression_specs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
    }

    classification_specs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
    }

    regression_models: dict[str, Pipeline] = {}
    regression_rows: list[dict[str, float | str]] = []
    for name, model in regression_specs.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(x_train, y_reg_train)
        pred = pipe.predict(x_test)
        regression_models[name] = pipe
        regression_rows.append(
            {
                "Model": name,
                "MAE": round(mean_absolute_error(y_reg_test, pred), 2),
                "R2 Score": round(r2_score(y_reg_test, pred), 3),
            }
        )

    classification_models: dict[str, Pipeline] = {}
    classification_rows: list[dict[str, float | str]] = []
    for name, model in classification_specs.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipe.fit(x_cls_train, y_cls_train)
        pred = pipe.predict(x_cls_test)
        classification_models[name] = pipe
        classification_rows.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_cls_test, pred), 3),
            }
        )

    clustering_features = [
        "PM2_5_ugm3",
        "PM10_ugm3",
        "CO_ugm3",
        "NO2_ugm3",
        "SO2_ugm3",
        "O3_ugm3",
        "US_AQI",
    ]
    clustering_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10)),
        ]
    )
    clusters = clustering_model.fit_predict(df[clustering_features])
    clustered = df.assign(Cluster=clusters)
    cluster_summary = (
        clustered.groupby("Cluster")
        .agg(
            Records=("Cluster", "size"),
            Avg_AQI=("US_AQI", "mean"),
            Avg_PM25=("PM2_5_ugm3", "mean"),
            Avg_PM10=("PM10_ugm3", "mean"),
            Top_City=("City", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    cluster_summary[["Avg_AQI", "Avg_PM25", "Avg_PM10"]] = cluster_summary[
        ["Avg_AQI", "Avg_PM25", "Avg_PM10"]
    ].round(2)

    return ModelBundle(
        regression_models=regression_models,
        regression_metrics=pd.DataFrame(regression_rows),
        classification_models=classification_models,
        classification_metrics=pd.DataFrame(classification_rows),
        clustering_model=clustering_model,
        cluster_summary=cluster_summary,
    )


def city_aqi_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Latest record per city":
        city_df = (
            df.sort_values("Datetime")
            .groupby(["City", "State"], as_index=False)
            .tail(1)
            .copy()
        )
        city_df["Display_AQI"] = city_df["US_AQI"].round(1)
    else:
        city_df = (
            df.groupby(["City", "State", "Latitude", "Longitude"], as_index=False)
            .agg(Display_AQI=("US_AQI", "mean"), Records=("US_AQI", "size"))
            .copy()
        )
        city_df["Display_AQI"] = city_df["Display_AQI"].round(1)
        city_df["AQI_Level"] = city_df["Display_AQI"].apply(aqi_category)

    city_df["color"] = city_df["AQI_Level"].map(AQI_COLORS)
    city_df["radius"] = np.clip(city_df["Display_AQI"] * 120, 3000, 45000)
    return city_df


def render_map(city_df: pd.DataFrame) -> None:
    view_state = pdk.ViewState(latitude=22.7, longitude=79.0, zoom=4.2, pitch=0)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=city_df,
        get_position=["Longitude", "Latitude"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "html": "<b>{City}, {State}</b><br/>AQI: {Display_AQI}<br/>Level: {AQI_Level}",
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True)


def model_input_form(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    city_list = sorted(df["City"].unique())
    city = st.selectbox("City", city_list, key=f"{key_prefix}_city")
    selected = df[df["City"] == city].sort_values("Datetime").tail(1).iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        pm25 = st.number_input("PM2.5 ug/m3", value=float(selected["PM2_5_ugm3"]), min_value=0.0, key=f"{key_prefix}_pm25")
        pm10 = st.number_input("PM10 ug/m3", value=float(selected["PM10_ugm3"]), min_value=0.0, key=f"{key_prefix}_pm10")
        co = st.number_input("CO ug/m3", value=float(selected["CO_ugm3"]), min_value=0.0, key=f"{key_prefix}_co")
    with c2:
        no2 = st.number_input("NO2 ug/m3", value=float(selected["NO2_ugm3"]), min_value=0.0, key=f"{key_prefix}_no2")
        so2 = st.number_input("SO2 ug/m3", value=float(selected["SO2_ugm3"]), min_value=0.0, key=f"{key_prefix}_so2")
        o3 = st.number_input("O3 ug/m3", value=float(selected["O3_ugm3"]), min_value=0.0, key=f"{key_prefix}_o3")
    with c3:
        temp = st.number_input("Temperature C", value=float(selected["Temp_2m_C"]), key=f"{key_prefix}_temp")
        humidity = st.number_input("Humidity %", value=float(selected["Humidity_Percent"]), min_value=0.0, max_value=100.0, key=f"{key_prefix}_humidity")
        wind = st.number_input("Wind speed km/h", value=float(selected["Wind_Speed_10m_kmh"]), min_value=0.0, key=f"{key_prefix}_wind")

    row = selected.copy()
    updates = {
        "PM2_5_ugm3": pm25,
        "PM10_ugm3": pm10,
        "CO_ugm3": co,
        "NO2_ugm3": no2,
        "SO2_ugm3": so2,
        "O3_ugm3": o3,
        "Temp_2m_C": temp,
        "Humidity_Percent": humidity,
        "Wind_Speed_10m_kmh": wind,
    }
    for key, value in updates.items():
        row[key] = value

    return pd.DataFrame([row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]])


def main() -> None:
    st.title("India AQI Predictor and Pollution Analyzer")

    with st.sidebar:
        st.header("Project Controls")
        data_path = st.text_input("Dataset path", DEFAULT_DATA_PATH)
        max_rows_choice = st.selectbox(
            "Rows to load",
            ["100,000", "250,000", "Full dataset"],
            index=1,
        )
        max_rows = None if max_rows_choice == "Full dataset" else int(max_rows_choice.replace(",", ""))
        train_sample = st.slider("Training sample size", 5000, 80000, 25000, step=5000)
        category_mode = st.selectbox(
            "Classification labels",
            ["Simple: Good / Moderate / Poor", "Detailed: 6 AQI levels"],
        )

    with st.spinner("Loading dataset..."):
        df = cached_load_data(data_path, max_rows)

    aqi_mean = df["US_AQI"].mean()
    latest_date = df["Datetime"].max()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Records loaded", f"{len(df):,}")
    k2.metric("Cities", df["City"].nunique())
    k3.metric("Average AQI", f"{aqi_mean:.1f}")
    k4.metric("Latest timestamp", latest_date.strftime("%d-%m-%Y %H:%M") if pd.notna(latest_date) else "NA")

    tabs = st.tabs(
        [
            "India AQI Map",
            "Regression Models",
            "Classification Models",
            "Clustering",
            "Dataset Explorer",
        ]
    )

    with tabs[0]:
        st.subheader("India city AQI map")
        map_mode = st.radio(
            "Map value",
            ["Latest record per city", "Average AQI per city"],
            horizontal=True,
        )
        city_df = city_aqi_frame(df, map_mode)
        render_map(city_df)
        st.dataframe(
            city_df[["City", "State", "Display_AQI", "AQI_Level", "Latitude", "Longitude"]]
            .sort_values("Display_AQI", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with st.spinner("Training models for this session..."):
        bundle = train_models(data_path, max_rows, train_sample, category_mode)

    with tabs[1]:
        st.subheader("AQI value prediction")
        st.dataframe(bundle.regression_metrics, use_container_width=True, hide_index=True)
        st.divider()
        st.caption("Enter pollutant and weather values. Each regression model predicts AQI separately.")
        input_df = model_input_form(df, "regression")
        cols = st.columns(3)
        for idx, (name, model) in enumerate(bundle.regression_models.items()):
            prediction = float(model.predict(input_df)[0])
            cols[idx].metric(name, f"{prediction:.1f} AQI", aqi_category(prediction))

    with tabs[2]:
        st.subheader("Pollution level prediction")
        st.dataframe(bundle.classification_metrics, use_container_width=True, hide_index=True)
        st.divider()
        st.caption("Each classifier predicts pollution level separately from the same input values.")
        input_df = model_input_form(df, "classification")
        cols = st.columns(3)
        for idx, (name, model) in enumerate(bundle.classification_models.items()):
            prediction = model.predict(input_df)[0]
            cols[idx].metric(name, str(prediction))

    with tabs[3]:
        st.subheader("Pollution pattern analyzer")
        st.dataframe(bundle.cluster_summary, use_container_width=True, hide_index=True)
        st.caption("K-Means groups similar pollution patterns using AQI and pollutant features.")

    with tabs[4]:
        st.subheader("Dataset explorer")
        city_filter = st.multiselect("Filter cities", sorted(df["City"].unique()))
        view_df = df if not city_filter else df[df["City"].isin(city_filter)]
        st.dataframe(view_df.head(1000), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
